//! GGUF (llama.cpp) v2/v3: header, typed key/value metadata, tensor infos,
//! then aligned tensor data.
//!
//! Unknown tensor types (e.g. vendor types like PrismML's `PQ2_0`) are kept:
//! their size is inferred from the distance to the next tensor.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use mb_ir::{AuxFiles, DType, MetaType, MetaValue, Metadata, RawModel, SourceFormat, TensorInfo};

use crate::{invalid, io_err, map_file, Error, LoadedModel, Result, TensorToWrite};

pub const MAGIC: &[u8; 4] = b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
const ALIGNMENT_KEY: &str = "general.alignment";
/// Limit on nested metadata arrays, to reject hostile files early.
const MAX_DEPTH: u32 = 8;

pub fn has_magic(path: &Path) -> bool {
    let mut buf = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok_and(|_| &buf == MAGIC)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    path: &'a Path,
}

impl<'a> Cursor<'a> {
    fn err(&self, msg: impl Into<String>) -> Error {
        invalid(
            self.path,
            "GGUF",
            format!("at byte {}: {}", self.pos, msg.into()),
        )
    }

    fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let end = usize::try_from(n)
            .ok()
            .and_then(|n| self.pos.checked_add(n))
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| self.err(format!("unexpected end of file reading {n} bytes")))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self
            .take(N as u64)?
            .try_into()
            .expect("take returns N bytes"))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// Reads a count and checks it could fit in the remaining bytes at `min_size` each.
    fn count(&mut self, min_size: u64) -> Result<u64> {
        let n = self.u64()?;
        let remaining = (self.buf.len() - self.pos) as u64;
        if n.saturating_mul(min_size) > remaining {
            return Err(self.err(format!(
                "count {n} cannot fit in remaining {remaining} bytes"
            )));
        }
        Ok(n)
    }

    fn string(&mut self) -> Result<String> {
        let n = self.count(1)?;
        let bytes = self.take(n)?;
        // Tokenizer vocabularies occasionally hold invalid UTF-8; keep going lossily.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: MetaType, depth: u32) -> Result<MetaValue> {
        Ok(match ty {
            MetaType::U8 => MetaValue::U8(self.array::<1>()?[0]),
            MetaType::I8 => MetaValue::I8(self.array::<1>()?[0] as i8),
            MetaType::U16 => MetaValue::U16(u16::from_le_bytes(self.array()?)),
            MetaType::I16 => MetaValue::I16(i16::from_le_bytes(self.array()?)),
            MetaType::U32 => MetaValue::U32(self.u32()?),
            MetaType::I32 => MetaValue::I32(i32::from_le_bytes(self.array()?)),
            MetaType::U64 => MetaValue::U64(self.u64()?),
            MetaType::I64 => MetaValue::I64(i64::from_le_bytes(self.array()?)),
            MetaType::F32 => MetaValue::F32(f32::from_le_bytes(self.array()?)),
            MetaType::F64 => MetaValue::F64(f64::from_le_bytes(self.array()?)),
            MetaType::Bool => MetaValue::Bool(self.array::<1>()?[0] != 0),
            MetaType::String => MetaValue::String(self.string()?),
            MetaType::Array => {
                if depth >= MAX_DEPTH {
                    return Err(self.err("metadata arrays nested too deeply"));
                }
                let elem = self.meta_type()?;
                let n = self.count(1)?;
                let values = (0..n)
                    .map(|_| self.value(elem, depth + 1))
                    .collect::<Result<_>>()?;
                MetaValue::Array { elem, values }
            }
        })
    }

    fn meta_type(&mut self) -> Result<MetaType> {
        let id = self.u32()?;
        MetaType::from_id(id).ok_or_else(|| self.err(format!("unknown metadata type {id}")))
    }
}

fn align(n: u64, alignment: u64) -> u64 {
    n.div_ceil(alignment) * alignment
}

/// Parses a GGUF file already mapped into memory.
pub fn parse(buf: &[u8], path: &Path) -> Result<(Metadata, Vec<TensorInfo>)> {
    let mut c = Cursor { buf, pos: 0, path };
    if c.take(4)? != MAGIC {
        return Err(c.err("bad magic"));
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(c.err(format!("unsupported GGUF version {version}")));
    }
    // Minimum sizes: a tensor info is at least 8+4+4+8 bytes, a KV pair at least 8+4+1.
    let n_tensors = c.count(24)?;
    let n_kv = c.count(13)?;

    let mut kv = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.meta_type()?;
        kv.push((key, c.value(ty, 0)?));
    }
    let alignment = kv
        .iter()
        .find(|(k, _)| k == ALIGNMENT_KEY)
        .and_then(|(_, v)| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(c.err(format!("invalid alignment {alignment}")));
    }

    struct Pending {
        name: String,
        dtype: DType,
        shape: Vec<u64>,
        rel_offset: u64,
    }
    let mut pending = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()?;
        if n_dims > 8 {
            return Err(c.err(format!("{name}: {n_dims} dims")));
        }
        let mut shape = (0..n_dims).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        shape.reverse(); // ggml `ne` is innermost-first; store row-major.
        let dtype = DType::from_ggml(c.u32()?);
        let rel_offset = c.u64()?;
        pending.push(Pending {
            name,
            dtype,
            shape,
            rel_offset,
        });
    }

    let data_start = align(c.pos as u64, alignment);
    let data_len = (buf.len() as u64)
        .checked_sub(data_start)
        .ok_or_else(|| c.err("tensor data section missing"))?;

    // Offsets sorted, so unknown-type tensors can be sized by the gap to the next one.
    let mut offsets: Vec<u64> = pending.iter().map(|p| p.rel_offset).collect();
    offsets.sort_unstable();
    let next_offset = |off: u64| {
        let i = offsets.partition_point(|&o| o <= off);
        offsets.get(i).copied().unwrap_or(data_len)
    };

    let tensors = pending
        .into_iter()
        .map(|p| {
            let n: u64 = p.shape.iter().product();
            let (n_bytes, exact) = match p.dtype.storage_bytes(n) {
                Some(b) => (b, true),
                None if p.dtype.block_layout().is_some() => {
                    return Err(c.err(format!(
                        "{}: {n} elements not a multiple of the {} block size",
                        p.name, p.dtype
                    )))
                }
                None => (
                    next_offset(p.rel_offset).saturating_sub(p.rel_offset),
                    false,
                ),
            };
            if p.rel_offset % alignment != 0
                || p.rel_offset
                    .checked_add(n_bytes)
                    .is_none_or(|e| e > data_len)
            {
                return Err(c.err(format!(
                    "{}: data at {}+{n_bytes} is misaligned or out of bounds",
                    p.name, p.rel_offset
                )));
            }
            Ok(TensorInfo {
                name: p.name,
                dtype: p.dtype,
                shape: p.shape,
                file: 0,
                offset: data_start + p.rel_offset,
                n_bytes,
                bytes_exact: exact,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((Metadata::Gguf { version, kv }, tensors))
}

pub fn open(path: &Path) -> Result<LoadedModel> {
    let map = map_file(path)?;
    let (metadata, tensors) = parse(&map, path)?;
    let chat_template = match &metadata {
        Metadata::Gguf { kv, .. } => kv
            .iter()
            .find(|(k, _)| k == "tokenizer.chat_template")
            .and_then(|(_, v)| v.as_str().map(str::to_owned)),
        Metadata::Hf { .. } => None,
    };
    Ok(LoadedModel {
        raw: RawModel {
            format: SourceFormat::Gguf,
            root: path.to_owned(),
            files: vec![path.to_owned()],
            metadata,
            aux: AuxFiles {
                chat_template,
                ..Default::default()
            },
            tensors,
        },
        maps: vec![map],
    })
}

fn write_string(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn write_value(w: &mut impl Write, v: &MetaValue) -> std::io::Result<()> {
    match v {
        MetaValue::U8(x) => w.write_all(&[*x]),
        MetaValue::I8(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::U16(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::I16(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::U32(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::I32(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::U64(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::I64(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::F32(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::F64(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::Bool(x) => w.write_all(&[u8::from(*x)]),
        MetaValue::String(s) => write_string(w, s),
        MetaValue::Array { elem, values } => {
            w.write_all(&(*elem as u32).to_le_bytes())?;
            w.write_all(&(values.len() as u64).to_le_bytes())?;
            values.iter().try_for_each(|v| write_value(w, v))
        }
    }
}

/// Writes a GGUF v3 file. Tensor data is streamed in the given order.
/// Alignment comes from `general.alignment` in `kv` if present.
pub fn write(path: &Path, kv: &[(String, MetaValue)], tensors: &[TensorToWrite<'_>]) -> Result<()> {
    let alignment = kv
        .iter()
        .find(|(k, _)| k == ALIGNMENT_KEY)
        .and_then(|(_, v)| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);
    for (_, v) in kv {
        if let MetaValue::Array { elem, values } = v {
            if values.iter().any(|x| x.meta_type() != *elem) {
                return Err(invalid(
                    path,
                    "GGUF metadata",
                    "array elements must match the declared element type",
                ));
            }
        }
    }
    let mut ids = Vec::with_capacity(tensors.len());
    for t in tensors {
        t.check_size()?;
        ids.push(t.dtype.ggml_id().ok_or_else(|| Error::Tensor {
            name: t.name.clone(),
            msg: format!("{} cannot be stored in GGUF", t.dtype),
        })?);
    }

    let file = File::create(path).map_err(io_err(path))?;
    let mut w = BufWriter::new(file);
    let result = (|| -> std::io::Result<()> {
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, v) in kv {
            write_string(&mut header, k)?;
            header.extend_from_slice(&(v.meta_type() as u32).to_le_bytes());
            write_value(&mut header, v)?;
        }
        let mut offset = 0u64;
        for (t, id) in tensors.iter().zip(&ids) {
            write_string(&mut header, &t.name)?;
            header.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for d in t.shape.iter().rev() {
                header.extend_from_slice(&d.to_le_bytes());
            }
            header.extend_from_slice(&id.to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
            offset = align(offset + t.data.len() as u64, alignment);
        }
        w.write_all(&header)?;
        let zeros = vec![0u8; alignment as usize];
        let pad = |len: u64| (align(len, alignment) - len) as usize;
        w.write_all(&zeros[..pad(header.len() as u64)])?;
        for t in tensors {
            w.write_all(&t.data)?;
            w.write_all(&zeros[..pad(t.data.len() as u64)])?;
        }
        w.flush()
    })();
    result.map_err(io_err(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mb_ir::GgmlType;
    use std::borrow::Cow;

    fn kv() -> Vec<(String, MetaValue)> {
        vec![
            (
                "general.architecture".into(),
                MetaValue::String("llama".into()),
            ),
            ("llama.block_count".into(), MetaValue::U32(1)),
            ("llama.rope.freq_base".into(), MetaValue::F32(10000.0)),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array {
                    elem: MetaType::String,
                    values: vec![MetaValue::String("a".into()), MetaValue::String("b".into())],
                },
            ),
        ]
    }

    #[test]
    fn round_trip_with_known_and_unknown_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        let q8 = DType::Ggml(GgmlType::from_name("Q8_0").unwrap());
        let custom = DType::Ggml(GgmlType(200));
        let tensors = [
            TensorToWrite {
                name: "a".into(),
                dtype: DType::F32,
                shape: vec![3, 2],
                data: Cow::Owned(vec![7; 24]),
            },
            TensorToWrite {
                name: "q".into(),
                dtype: q8,
                shape: vec![2, 32],
                data: Cow::Owned(vec![9; 68]),
            },
            TensorToWrite {
                name: "c".into(),
                dtype: custom,
                shape: vec![128],
                data: Cow::Owned(vec![5; 40]),
            },
        ];
        write(&path, &kv(), &tensors).unwrap();

        let model = open(&path).unwrap();
        let Metadata::Gguf {
            version,
            kv: read_kv,
        } = &model.raw.metadata
        else {
            panic!()
        };
        assert_eq!(*version, 3);
        assert_eq!(read_kv, &kv());

        let t = &model.raw.tensors;
        assert_eq!(t[0].shape, vec![3, 2]);
        assert_eq!(model.tensor_bytes(&t[0]).unwrap(), &[7; 24]);
        assert_eq!((t[1].dtype, t[1].n_bytes, t[1].bytes_exact), (q8, 68, true));
        assert_eq!(model.tensor_bytes(&t[1]).unwrap(), &[9; 68]);
        // Unknown type: sized to the end of the data (40 bytes + 24 padding).
        assert_eq!((t[2].dtype, t[2].bytes_exact), (custom, false));
        assert_eq!(t[2].n_bytes, 64);
        assert!(t.iter().all(|t| (t.offset) % DEFAULT_ALIGNMENT == 0));
    }

    #[test]
    fn rejects_truncated_and_hostile_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // absurd tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse(&bytes, &path).is_err());
        assert!(parse(b"GGUF", &path).is_err());
        assert!(parse(b"NOPE\x03\0\0\0", &path).is_err());
    }
}

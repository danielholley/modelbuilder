//! Hugging Face safetensors: `u64` LE header length, a JSON header, then raw data.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use mb_ir::{AuxFiles, DType, Metadata, RawModel, SourceFormat, TensorInfo};
use serde_json::{Map, Value};

use crate::{invalid, io_err, map_file, LoadedModel, Result, TensorToWrite};

/// Headers larger than this are rejected rather than parsed (safetensors itself caps at 100 MB).
const MAX_HEADER: u64 = 100 << 20;

pub struct ParsedHeader {
    pub tensors: Vec<TensorInfo>,
    pub metadata: BTreeMap<String, String>,
}

/// Parses the header of one safetensors file. `file` is recorded in each [`TensorInfo`].
pub fn parse_header(bytes: &[u8], path: &Path, file: usize) -> Result<ParsedHeader> {
    let bad = |msg: String| invalid(path, "safetensors header", msg);
    let len_bytes: [u8; 8] = bytes
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| bad("file shorter than 8 bytes".into()))?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER || 8 + header_len > bytes.len() as u64 {
        return Err(bad(format!(
            "header length {header_len} exceeds file size {}",
            bytes.len()
        )));
    }
    let data_start = 8 + header_len;
    let header: Map<String, Value> =
        serde_json::from_slice(&bytes[8..data_start as usize]).map_err(|e| bad(e.to_string()))?;
    let data_len = bytes.len() as u64 - data_start;

    let mut tensors = Vec::with_capacity(header.len());
    let mut metadata = BTreeMap::new();
    for (name, entry) in header {
        if name == "__metadata__" {
            if let Value::Object(m) = entry {
                for (k, v) in m {
                    let v = v
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| v.to_string());
                    metadata.insert(k, v);
                }
            }
            continue;
        }
        let field = |k: &str| {
            entry
                .get(k)
                .ok_or_else(|| bad(format!("{name}: missing {k}")))
        };
        let dtype_s = field("dtype")?.as_str().unwrap_or_default();
        let dtype = DType::from_safetensors(dtype_s)
            .ok_or_else(|| bad(format!("{name}: unknown dtype {dtype_s:?}")))?;
        let shape = field("shape")?
            .as_array()
            .and_then(|a| a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>())
            .ok_or_else(|| bad(format!("{name}: bad shape")))?;
        let (start, end) = match field("data_offsets")?.as_array().map(Vec::as_slice) {
            Some([s, e]) => (s.as_u64(), e.as_u64()),
            _ => (None, None),
        };
        let (Some(start), Some(end)) = (start, end) else {
            return Err(bad(format!("{name}: bad data_offsets")));
        };
        if start > end || end > data_len {
            return Err(bad(format!(
                "{name}: data_offsets [{start}, {end}) outside data section of {data_len} bytes"
            )));
        }
        let n_bytes = end - start;
        let n: u64 = shape.iter().product();
        if dtype.storage_bytes(n) != Some(n_bytes) {
            return Err(bad(format!(
                "{name}: {dtype} {shape:?} does not match {n_bytes} bytes"
            )));
        }
        tensors.push(TensorInfo {
            name,
            dtype,
            shape,
            file,
            offset: data_start + start,
            n_bytes,
            bytes_exact: true,
        });
    }
    tensors.sort_by_key(|t| t.offset);
    Ok(ParsedHeader { tensors, metadata })
}

fn read_json(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path).map_err(io_err(path))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| invalid(path, "JSON", e.to_string()))
}

fn shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if let Some(idx) = read_json(&index)? {
        let map = idx
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(&index, "index", "missing weight_map"))?;
        let names: BTreeSet<&str> = map.values().filter_map(Value::as_str).collect();
        return Ok(names.into_iter().map(|n| dir.join(n)).collect());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(io_err(dir))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    Ok(files)
}

fn read_aux(dir: &Path) -> Result<AuxFiles> {
    let read_text = |name: &str| -> Result<Option<String>> {
        let p = dir.join(name);
        if p.is_file() {
            std::fs::read_to_string(&p).map(Some).map_err(io_err(&p))
        } else {
            Ok(None)
        }
    };
    let chat_template = match read_text("chat_template.jinja")? {
        Some(t) => Some(t),
        None => read_json(&dir.join("chat_template.json"))?
            .and_then(|v| v.get("chat_template")?.as_str().map(str::to_owned)),
    };
    Ok(AuxFiles {
        tokenizer_config: read_json(&dir.join("tokenizer_config.json"))?,
        generation_config: read_json(&dir.join("generation_config.json"))?,
        model_card: read_text("README.md")?,
        chat_template,
    })
}

/// Opens an HF model directory.
pub fn open(dir: &Path) -> Result<LoadedModel> {
    let config_path = dir.join("config.json");
    let config = read_json(&config_path)?
        .ok_or_else(|| invalid(dir, "model directory", "missing config.json"))?;
    let files = shard_files(dir)?;
    if files.is_empty() {
        return Err(invalid(dir, "model directory", "no .safetensors files"));
    }

    let mut maps = Vec::with_capacity(files.len());
    let mut tensors = Vec::new();
    let mut safetensors_metadata = BTreeMap::new();
    for (i, path) in files.iter().enumerate() {
        let map = map_file(path)?;
        let parsed = parse_header(&map, path, i)?;
        tensors.extend(parsed.tensors);
        safetensors_metadata.extend(parsed.metadata);
        maps.push(map);
    }

    Ok(LoadedModel {
        raw: RawModel {
            format: SourceFormat::HfSafetensors,
            root: dir.to_owned(),
            files,
            metadata: Metadata::Hf {
                config,
                safetensors_metadata,
            },
            aux: read_aux(dir)?,
            tensors,
        },
        maps,
    })
}

/// Writes one safetensors file, streaming tensor data in the given order.
pub fn write(
    path: &Path,
    tensors: &[TensorToWrite<'_>],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let mut header = Map::new();
    if !metadata.is_empty() {
        let m: Map<String, Value> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().into()))
            .collect();
        header.insert("__metadata__".into(), Value::Object(m));
    }
    let mut offset = 0u64;
    for t in tensors {
        t.check_size()?;
        let dtype = t
            .dtype
            .safetensors_name()
            .ok_or_else(|| crate::Error::Tensor {
                name: t.name.clone(),
                msg: format!("{} cannot be stored in safetensors", t.dtype),
            })?;
        let end = offset + t.data.len() as u64;
        header.insert(
            t.name.clone(),
            serde_json::json!({ "dtype": dtype, "shape": t.shape, "data_offsets": [offset, end] }),
        );
        offset = end;
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    // Pad so tensor data starts 8-byte aligned; the format allows trailing spaces.
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }

    let file = File::create(path).map_err(io_err(path))?;
    let mut w = BufWriter::new(file);
    let result = (|| {
        w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&header_bytes)?;
        for t in tensors {
            w.write_all(&t.data)?;
        }
        w.flush()
    })();
    result.map_err(io_err(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn write_then_parse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let a = TensorToWrite {
            name: "a".into(),
            dtype: DType::F32,
            shape: vec![2, 3],
            data: Cow::Owned(vec![1; 24]),
        };
        let b = TensorToWrite {
            name: "b".into(),
            dtype: DType::Bf16,
            shape: vec![5],
            data: Cow::Owned(vec![2; 10]),
        };
        let meta = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        write(&path, &[a, b], &meta).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let parsed = parse_header(&bytes, &path, 0).unwrap();
        assert_eq!(parsed.metadata, meta);
        assert_eq!(parsed.tensors.len(), 2);
        let b = &parsed.tensors[1];
        assert_eq!(
            (b.name.as_str(), b.dtype, b.n_bytes),
            ("b", DType::Bf16, 10)
        );
        assert_eq!(&bytes[b.offset as usize..][..10], &[2; 10]);
        assert_eq!(parsed.tensors[0].offset % 8, 0);
    }

    #[test]
    fn rejects_size_mismatch_and_bad_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let t = TensorToWrite {
            name: "a".into(),
            dtype: DType::F32,
            shape: vec![4],
            data: Cow::Owned(vec![0; 3]),
        };
        assert!(write(&path, &[t], &BTreeMap::new()).is_err());

        let header = br#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0; 8]); // only 8 of the 16 declared bytes
        assert!(parse_header(&bytes, &path, 0).is_err());
    }
}

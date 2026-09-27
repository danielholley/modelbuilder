//! Readers and writers for model checkpoints.
//!
//! Readers memory-map tensor files and parse only headers, so opening a
//! 100B+ parameter model costs a few MB of RAM. Tensor bytes are borrowed
//! straight from the maps via [`LoadedModel::tensor_bytes`].

pub mod gguf;
pub mod safetensors;

use std::fs::File;
use std::path::{Path, PathBuf};

use mb_ir::{DType, RawModel, TensorInfo};
use memmap2::Mmap;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: invalid {what}: {msg}")]
    Invalid {
        path: PathBuf,
        what: &'static str,
        msg: String,
    },
    #[error("{0}: not a recognized model (expected a GGUF file or a directory with config.json and .safetensors files)")]
    Unrecognized(PathBuf),
    #[error("tensor {name}: {msg}")]
    Tensor { name: String, msg: String },
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_owned(),
        source,
    }
}

pub(crate) fn invalid(path: &Path, what: &'static str, msg: impl Into<String>) -> Error {
    Error::Invalid {
        path: path.to_owned(),
        what,
        msg: msg.into(),
    }
}

pub(crate) fn map_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(io_err(path))?;
    // SAFETY: the map is read-only. Checkpoint files must not be modified while
    // open; modelbuilder never writes to its inputs.
    unsafe { Mmap::map(&file) }.map_err(io_err(path))
}

/// An opened checkpoint: its tensor index plus the memory maps backing it.
pub struct LoadedModel {
    pub raw: RawModel,
    maps: Vec<Mmap>,
}

impl LoadedModel {
    /// Borrows a tensor's bytes from the memory map. Touching the slice pages
    /// in only that tensor.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let map = self.maps.get(t.file).ok_or_else(|| Error::Tensor {
            name: t.name.clone(),
            msg: format!("file index {} out of range", t.file),
        })?;
        let start = usize::try_from(t.offset).ok();
        let end = start.and_then(|s| s.checked_add(usize::try_from(t.n_bytes).ok()?));
        match (start, end) {
            (Some(s), Some(e)) if e <= map.len() => Ok(&map[s..e]),
            _ => Err(Error::Tensor {
                name: t.name.clone(),
                msg: format!(
                    "data range {}+{} exceeds file size {}",
                    t.offset,
                    t.n_bytes,
                    map.len()
                ),
            }),
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.raw.tensors.iter().find(|t| t.name == name)
    }
}

/// Opens a checkpoint, detecting its format from the path.
///
/// Accepts a `.gguf` file, a directory containing one `.gguf` file, or an HF
/// directory with `config.json` and `.safetensors` shards.
pub fn open(path: &Path) -> Result<LoadedModel> {
    if path.is_file() {
        return match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => gguf::open(path),
            Some("safetensors") => safetensors::open(path.parent().unwrap_or(Path::new("."))),
            _ if gguf::has_magic(path) => gguf::open(path),
            _ => Err(Error::Unrecognized(path.to_owned())),
        };
    }
    if path.join("config.json").is_file() {
        return safetensors::open(path);
    }
    let ggufs: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(io_err(path))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    match ggufs.as_slice() {
        [one] => gguf::open(one),
        _ => Err(Error::Unrecognized(path.to_owned())),
    }
}

/// A tensor to be written. Data is borrowed so it can stream from an input map.
pub struct TensorToWrite<'a> {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: std::borrow::Cow<'a, [u8]>,
}

impl TensorToWrite<'_> {
    fn check_size(&self) -> Result<()> {
        let n: u64 = self.shape.iter().product();
        match self.dtype.storage_bytes(n) {
            Some(b) if b != self.data.len() as u64 => Err(Error::Tensor {
                name: self.name.clone(),
                msg: format!(
                    "{} {:?} needs {b} bytes, got {}",
                    self.dtype,
                    self.shape,
                    self.data.len()
                ),
            }),
            _ => Ok(()),
        }
    }
}

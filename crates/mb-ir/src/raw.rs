use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    /// Hugging Face layout: `config.json` + one or more `.safetensors` shards.
    HfSafetensors,
    Gguf,
}

/// One entry of the tensor index. No tensor data is held here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    /// Logical shape in row-major order (outermost dimension first), matching
    /// PyTorch/HF conventions. GGUF readers reverse ggml's `ne` order to match.
    pub shape: Vec<u64>,
    /// Index into [`RawModel::files`].
    pub file: usize,
    /// Absolute byte offset of the tensor data within its file.
    pub offset: u64,
    /// Size of the tensor data in bytes.
    pub n_bytes: u64,
    /// False when `n_bytes` had to be inferred (e.g. an unknown GGUF type,
    /// sized from the gap to the next tensor, which may include padding).
    pub bytes_exact: bool,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// GGUF metadata value type ids, as stored on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum MetaType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl MetaType {
    pub fn from_id(id: u32) -> Option<Self> {
        use MetaType::*;
        Some(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }
}

/// A GGUF metadata value. Integer widths are kept exactly so a read/write
/// round trip produces the types llama.cpp expects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array {
        elem: MetaType,
        values: Vec<MetaValue>,
    },
}

impl MetaValue {
    pub fn meta_type(&self) -> MetaType {
        match self {
            Self::U8(_) => MetaType::U8,
            Self::I8(_) => MetaType::I8,
            Self::U16(_) => MetaType::U16,
            Self::I16(_) => MetaType::I16,
            Self::U32(_) => MetaType::U32,
            Self::I32(_) => MetaType::I32,
            Self::U64(_) => MetaType::U64,
            Self::I64(_) => MetaType::I64,
            Self::F32(_) => MetaType::F32,
            Self::F64(_) => MetaType::F64,
            Self::Bool(_) => MetaType::Bool,
            Self::String(_) => MetaType::String,
            Self::Array { .. } => MetaType::Array,
        }
    }

    fn as_i128(&self) -> Option<i128> {
        Some(match *self {
            Self::U8(v) => v.into(),
            Self::I8(v) => v.into(),
            Self::U16(v) => v.into(),
            Self::I16(v) => v.into(),
            Self::U32(v) => v.into(),
            Self::I32(v) => v.into(),
            Self::U64(v) => v.into(),
            Self::I64(v) => v.into(),
            _ => return None,
        })
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_i128().and_then(|v| u64::try_from(v).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(v.into()),
            Self::F64(v) => Some(v),
            _ => self.as_i128().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            Self::Array { values, .. } => Some(values),
            _ => None,
        }
    }

    /// A compact JSON rendering for reports. Long arrays are summarized.
    pub fn to_display_json(&self) -> serde_json::Value {
        use serde_json::Value as J;
        match self {
            Self::F32(v) => J::from(f64::from(*v)),
            Self::F64(v) => J::from(*v),
            Self::Bool(b) => J::from(*b),
            Self::String(s) => J::from(s.as_str()),
            Self::Array { elem, values } if values.len() > 16 => {
                J::from(format!("[{} x {:?}]", values.len(), elem))
            }
            Self::Array { values, .. } => {
                J::Array(values.iter().map(Self::to_display_json).collect())
            }
            other => match other.as_i128() {
                Some(v) if v >= 0 => J::from(v as u64),
                Some(v) => J::from(v as i64),
                None => J::Null,
            },
        }
    }
}

/// Format-specific model metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Metadata {
    /// Parsed `config.json`, plus any `__metadata__` from safetensors headers.
    Hf {
        config: serde_json::Value,
        safetensors_metadata: BTreeMap<String, String>,
    },
    /// GGUF key/value pairs, in file order.
    Gguf {
        version: u32,
        kv: Vec<(String, MetaValue)>,
    },
}

/// Side files that carry provenance rather than weights.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct AuxFiles {
    pub tokenizer_config: Option<serde_json::Value>,
    pub generation_config: Option<serde_json::Value>,
    /// `README.md` model card, if present.
    pub model_card: Option<String>,
    /// Chat template shipped as a separate file (`chat_template.jinja`/`.json`).
    pub chat_template: Option<String>,
}

/// What a format reader produces: the tensor index and metadata, before normalization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct RawModel {
    pub format: SourceFormat,
    /// The path the model was opened from.
    pub root: PathBuf,
    /// Files holding tensor data, referenced by [`TensorInfo::file`].
    pub files: Vec<PathBuf>,
    pub metadata: Metadata,
    pub aux: AuxFiles,
    pub tensors: Vec<TensorInfo>,
}

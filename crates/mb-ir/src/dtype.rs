use std::fmt;

use serde::{Deserialize, Serialize};

/// Storage type of a tensor as it sits on disk.
///
/// Plain element types come from safetensors (and GGUF's unquantized types);
/// block-quantized GGUF types are carried as [`DType::Ggml`] so unknown or
/// vendor-specific types (e.g. PrismML's `PQ2_0`) survive a round trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    F16,
    Bf16,
    F32,
    F64,
    /// A GGML/GGUF tensor type, which may be block-quantized.
    Ggml(GgmlType),
}

impl DType {
    /// Parses a safetensors dtype string.
    pub fn from_safetensors(s: &str) -> Option<Self> {
        Some(match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" => Self::F8E8M0,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            _ => return None,
        })
    }

    /// The safetensors dtype string, or `None` for types safetensors can't hold.
    pub fn safetensors_name(self) -> Option<&'static str> {
        Some(match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Ggml(_) => return None,
        })
    }

    /// Maps a GGML type id to a `DType`, using plain element types where they exist.
    pub fn from_ggml(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::Bf16,
            _ => Self::Ggml(GgmlType(id)),
        }
    }

    /// The GGML type id for this dtype, if GGUF can store it.
    pub fn ggml_id(self) -> Option<u32> {
        Some(match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Bf16 => 30,
            Self::Ggml(t) => t.0,
            _ => return None,
        })
    }

    /// `(elements per block, bytes per block)`. Plain types have a block of one element.
    /// `None` for GGML types whose layout we don't know.
    pub fn block_layout(self) -> Option<(u64, u64)> {
        Some(match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => (1, 1),
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => (1, 2),
            Self::U32 | Self::I32 | Self::F32 => (1, 4),
            Self::U64 | Self::I64 | Self::F64 => (1, 8),
            Self::Ggml(t) => return t.block_layout(),
        })
    }

    /// Storage bytes for `n_elements`, if the layout is known and the count is block-aligned.
    pub fn storage_bytes(self, n_elements: u64) -> Option<u64> {
        let (elems, bytes) = self.block_layout()?;
        (n_elements % elems == 0).then(|| n_elements / elems * bytes)
    }

    /// Nominal storage bits per element (including block scales), if known.
    pub fn bits_per_element(self) -> Option<f64> {
        let (elems, bytes) = self.block_layout()?;
        Some(bytes as f64 * 8.0 / elems as f64)
    }

    /// True for block-quantized types, as opposed to plain element types.
    pub fn is_block_quantized(self) -> bool {
        matches!(self, Self::Ggml(_))
    }

    pub fn name(self) -> String {
        match self {
            Self::Ggml(t) => t.to_string(),
            other => other
                .safetensors_name()
                .expect("non-ggml dtypes have safetensors names")
                .to_string(),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// A GGML tensor type id. Ids outside the known table are kept verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct GgmlType(pub u32);

/// `(id, name, elements per block, bytes per block)` for upstream ggml types.
const GGML_TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
    // PrismML vendor types (PrismML-Eng/llama.cpp fork, not upstream). Ternary
    // {-1,0,+1} with one FP16 scale per 128 weights. Layouts match PrismML's
    // model card and the sizes measured from tensor offsets in the released GGUFs.
    (142, "PQ2_0", 128, 34),  // 2-bit slot per trit: 2.125 bits/weight
    (143, "PTQ1_0", 128, 28), // dense trit packing: 1.75 bits/weight
];

impl GgmlType {
    fn entry(self) -> Option<&'static (u32, &'static str, u64, u64)> {
        GGML_TYPES.iter().find(|e| e.0 == self.0)
    }

    pub fn known_name(self) -> Option<&'static str> {
        self.entry().map(|e| e.1)
    }

    pub fn block_layout(self) -> Option<(u64, u64)> {
        self.entry().map(|e| (e.2, e.3))
    }

    pub fn from_name(name: &str) -> Option<Self> {
        GGML_TYPES
            .iter()
            .find(|e| e.1.eq_ignore_ascii_case(name))
            .map(|e| Self(e.0))
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.known_name() {
            Some(n) => write!(f, "{n}"),
            None => write!(f, "ggml_type_{}", self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_types_round_trip_through_ggml_ids() {
        for d in [DType::F32, DType::F16, DType::Bf16, DType::I8] {
            assert_eq!(DType::from_ggml(d.ggml_id().unwrap()), d);
        }
    }

    #[test]
    fn block_sizes() {
        let q8 = DType::Ggml(GgmlType::from_name("Q8_0").unwrap());
        assert_eq!(q8.storage_bytes(64), Some(68));
        assert_eq!(q8.storage_bytes(33), None);
        assert_eq!(q8.bits_per_element(), Some(8.5));
        let tq2 = DType::Ggml(GgmlType::from_name("TQ2_0").unwrap());
        assert!((tq2.bits_per_element().unwrap() - 2.0625).abs() < 1e-9);
    }

    #[test]
    fn unknown_ggml_types_are_preserved() {
        let t = DType::from_ggml(4242);
        assert_eq!(t, DType::Ggml(GgmlType(4242)));
        assert_eq!(t.block_layout(), None);
        assert_eq!(t.to_string(), "ggml_type_4242");
    }
}

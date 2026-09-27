//! Checkpoint surgery: writes a new checkpoint with a feature added.
//!
//! Surgery never modifies its inputs. Tensors are streamed from the memory
//! maps of the source checkpoints; only small transformed tensors (norms)
//! are materialized.

pub mod export;
pub mod hf_export;
pub mod mtp;

use std::path::PathBuf;

use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum SurgeryError {
    #[error(transparent)]
    Format(#[from] mb_formats::Error),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("incompatible models: {0}")]
    Incompatible(String),
    #[error("output: {0}")]
    Output(String),
    #[error("{name}: {source}")]
    Dequant {
        name: String,
        #[source]
        source: mb_formats::dequant::DequantError,
    },
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct WrittenTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Where it came from: `target:<name>` or `reference:<name>`.
    pub source: String,
    /// What was done to it on the way (`copied`, `norm + 1 → F32`, ...).
    pub transform: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct SurgeryReport {
    pub output: PathBuf,
    pub bytes: u64,
    pub tensors: Vec<WrittenTensor>,
    /// Metadata keys added, changed or filtered.
    pub metadata: Vec<String>,
    pub notes: Vec<String>,
}

/// Round-to-nearest-even conversion from `f32` to BF16 bits.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040; // keep it a (quiet) NaN
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::f32_to_bf16;

    #[test]
    fn bf16_rounding() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-2.0), 0xc000);
        // 1 + 2^-8 is exactly halfway between two BF16 values: ties to even (1.0).
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3f80);
        // Just above halfway rounds up.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8) + 2f32.powi(-20)), 0x3f81);
        assert!(f32::from_bits(u32::from(f32_to_bf16(f32::NAN)) << 16).is_nan());
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
    }
}

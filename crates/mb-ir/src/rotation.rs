//! Orthogonal rotations folded into stored weights.
//!
//! PrismML's Bonsai 2 checkpoints store every foldable weight in a blockwise
//! Hadamard basis. The runtime (PrismML-Eng/llama.cpp, `build_lora_mm` in
//! `src/llama-graph.cpp`) computes
//!
//! ```text
//! y = W_stored · H(s ⊙ x)
//! ```
//!
//! where `H` is the normalized Sylvester-Walsh-Hadamard matrix applied to each
//! `block_size` chunk of the input dimension and `s` is a ±1 sign vector
//! chosen by the weight's input width. `H` is symmetric and orthogonal, so the
//! original (primal) weights are recovered per row as `w = s ⊙ (H · w_stored)`.
//! Embedding rows stored with the inverse transform (`inverse_weight_names`)
//! are restored the same way: the runtime computes `h = s ⊙ (H z)` after lookup.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::ConfigView;

const PREFIX: &str = "prism.hadamard.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum SignMode {
    /// No sign flips (`s = 1`).
    Identity,
    /// Per-width ±1 vectors in `prism.hadamard.sign_values`.
    Explicit,
}

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum RotationError {
    #[error("{0} is not declared as rotated")]
    NotRotated(String),
    #[error("row length {len} is not a multiple of block size {block}")]
    BadLength { len: usize, block: usize },
    #[error("no sign vector for input width {0}")]
    NoSigns(usize),
    #[error("invalid rotation metadata: {0}")]
    Metadata(String),
}

/// An orthogonal rotation folded into the stored weights. Surgery has to keep
/// new and modified tensors in the same basis and keep the metadata in sync.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct WeightRotation {
    pub scheme: String,
    pub version: Option<u64>,
    pub block_size: Option<u64>,
    pub sign_mode: SignMode,
    /// Input widths that have their own sign vector.
    pub sign_widths: Vec<u64>,
    /// Weight matrices stored in the rotated basis.
    pub rotated_tensors: usize,
    /// Tensors stored with the inverse rotation (e.g. input embeddings).
    pub inverse_tensors: usize,
    /// `ssm_out` inputs are kept in grouped (training) V-head order; the
    /// runtime permutes activations from llama.cpp's tiled order first.
    pub gdn_v_grouped: bool,
    /// Metadata key prefix that declares the rotation.
    pub metadata_prefix: String,
    #[serde(skip)]
    weight_names: BTreeSet<String>,
    #[serde(skip)]
    inverse_names: BTreeSet<String>,
    #[serde(skip)]
    signs: BTreeMap<u64, Vec<f32>>,
}

impl WeightRotation {
    /// Reads `prism.hadamard.*` metadata. Returns `None` when the checkpoint
    /// declares no rotation.
    pub fn detect(cfg: &ConfigView) -> Option<Self> {
        let key = |k: &str| cfg.gguf_raw(&format!("{PREFIX}{k}"));
        let version = key("version")?.as_u64();
        let names = |k: &str| -> BTreeSet<String> {
            key(k)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default()
        };
        let ints = |k: &str| -> Vec<i64> {
            key(k)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as i64))
                        .collect()
                })
                .unwrap_or_default()
        };
        let sign_mode = match key("sign_mode").and_then(|v| v.as_str()) {
            Some("explicit") => SignMode::Explicit,
            _ => SignMode::Identity,
        };

        // Sign vectors are concatenated in `sign_values`, one per entry of `sign_widths`.
        let mut signs = BTreeMap::new();
        let widths = ints("sign_widths");
        let values = ints("sign_values");
        let mut off = 0usize;
        for w in &widths {
            let w = *w as usize;
            if let Some(chunk) = values.get(off..off + w) {
                signs.insert(w as u64, chunk.iter().map(|&v| v as f32).collect());
            }
            off += w;
        }

        let weight_names = names("weight_names");
        let inverse_names = names("inverse_weight_names");
        Some(Self {
            scheme: key("transform")
                .and_then(|v| v.as_str())
                .unwrap_or("hadamard")
                .to_string(),
            version,
            block_size: key("block_size").and_then(|v| v.as_u64()),
            sign_mode,
            sign_widths: signs.keys().copied().collect(),
            rotated_tensors: weight_names.len(),
            inverse_tensors: inverse_names.len(),
            gdn_v_grouped: key("gdn_v_grouped")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            metadata_prefix: PREFIX.trim_end_matches('.').to_string(),
            weight_names,
            inverse_names,
            signs,
        })
    }

    /// Whether a tensor is stored in the rotated basis (forward or inverse).
    pub fn is_rotated(&self, name: &str) -> bool {
        self.weight_names.contains(name) || self.inverse_names.contains(name)
    }

    /// Converts one stored row (along the input dimension) back to the primal
    /// basis in place: `w = s ⊙ (H · w_stored)`.
    ///
    /// For `ssm_out` with [`Self::gdn_v_grouped`], the result is in grouped
    /// (HF training) V-head order, not llama.cpp's tiled order.
    pub fn to_primal(&self, name: &str, row: &mut [f32]) -> Result<(), RotationError> {
        if !self.is_rotated(name) {
            return Err(RotationError::NotRotated(name.to_string()));
        }
        let block = self
            .block_size
            .ok_or_else(|| RotationError::Metadata("missing block_size".into()))?
            as usize;
        if block == 0 || !block.is_power_of_two() {
            return Err(RotationError::Metadata(format!(
                "block size {block} is not a power of two"
            )));
        }
        if row.len() % block != 0 {
            return Err(RotationError::BadLength {
                len: row.len(),
                block,
            });
        }
        for chunk in row.chunks_exact_mut(block) {
            fwht_normalized(chunk);
        }
        if self.sign_mode == SignMode::Explicit {
            let s = self
                .signs
                .get(&(row.len() as u64))
                .ok_or(RotationError::NoSigns(row.len()))?;
            for (x, s) in row.iter_mut().zip(s) {
                *x *= s;
            }
        }
        Ok(())
    }

    /// The inverse of [`Self::to_primal`]: a primal row back to the stored
    /// basis, `w_stored = H · (s ⊙ w)` (H is symmetric and orthogonal, and
    /// `s ⊙ s = 1`). New or retrained weights go through this before they are
    /// re-quantized, so they stay in the checkpoint's rotated basis.
    pub fn from_primal(&self, name: &str, row: &mut [f32]) -> Result<(), RotationError> {
        if !self.is_rotated(name) {
            return Err(RotationError::NotRotated(name.to_string()));
        }
        let block = self
            .block_size
            .ok_or_else(|| RotationError::Metadata("missing block_size".into()))?
            as usize;
        if block == 0 || !block.is_power_of_two() {
            return Err(RotationError::Metadata(format!(
                "block size {block} is not a power of two"
            )));
        }
        if row.len() % block != 0 {
            return Err(RotationError::BadLength {
                len: row.len(),
                block,
            });
        }
        if self.sign_mode == SignMode::Explicit {
            let s = self
                .signs
                .get(&(row.len() as u64))
                .ok_or(RotationError::NoSigns(row.len()))?;
            for (x, s) in row.iter_mut().zip(s) {
                *x *= s;
            }
        }
        for chunk in row.chunks_exact_mut(block) {
            fwht_normalized(chunk);
        }
        Ok(())
    }
}

/// In-place fast Walsh-Hadamard transform, normalized by `1/sqrt(n)`.
///
/// This multiplies by the Sylvester-ordered matrix `H[r][c] = (-1)^popcount(r & c) / sqrt(n)`,
/// the same matrix PrismML's runtime builds in `llama-model.cpp`. `n` must be a
/// power of two. The transform is its own inverse.
pub fn fwht_normalized(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x {
        *v *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetaType, MetaValue, Metadata};

    /// The explicit matrix, built exactly as PrismML's runtime does.
    fn prism_matrix(n: usize) -> Vec<Vec<f32>> {
        let scale = 1.0 / (n as f32).sqrt();
        (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| {
                        if (r & c).count_ones() % 2 == 1 {
                            -scale
                        } else {
                            scale
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn fwht_matches_prism_matrix() {
        for n in [2, 8, 64] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let m = prism_matrix(n);
            let expected: Vec<f32> = m
                .iter()
                .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
                .collect();
            let mut got = x.clone();
            fwht_normalized(&mut got);
            for (g, e) in got.iter().zip(&expected) {
                assert!((g - e).abs() < 1e-5, "n={n}: {g} vs {e}");
            }
            fwht_normalized(&mut got); // involution
            for (g, e) in got.iter().zip(&x) {
                assert!((g - e).abs() < 1e-5);
            }
        }
    }

    fn rotation(block: u32, signs: &[i32]) -> WeightRotation {
        let s = |v: &str| MetaValue::String(v.into());
        let arr = |elem, values| MetaValue::Array { elem, values };
        let meta = Metadata::Gguf {
            version: 3,
            kv: vec![
                ("prism.hadamard.version".into(), MetaValue::U32(1)),
                ("prism.hadamard.block_size".into(), MetaValue::U32(block)),
                (
                    "prism.hadamard.transform".into(),
                    s("normalized-sylvester-walsh-hadamard"),
                ),
                ("prism.hadamard.sign_mode".into(), s("explicit")),
                (
                    "prism.hadamard.weight_names".into(),
                    arr(MetaType::String, vec![s("blk.0.attn_q.weight")]),
                ),
                (
                    "prism.hadamard.sign_widths".into(),
                    arr(MetaType::I32, vec![MetaValue::I32(signs.len() as i32)]),
                ),
                (
                    "prism.hadamard.sign_values".into(),
                    arr(
                        MetaType::I32,
                        signs.iter().map(|&v| MetaValue::I32(v)).collect(),
                    ),
                ),
            ],
        };
        WeightRotation::detect(&ConfigView::new(&meta)).unwrap()
    }

    /// Folding as the exporter does (`w_stored = H(s ⊙ w)`) and then running
    /// the runtime's matmul must give the original dot product, and
    /// `to_primal` must recover `w`.
    #[test]
    fn to_primal_inverts_the_fold() {
        let signs = [1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1];
        let rot = rotation(8, &signs);
        let w: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).cos()).collect();
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();

        let mut stored: Vec<f32> = w.iter().zip(&signs).map(|(w, &s)| w * s as f32).collect();
        stored.chunks_exact_mut(8).for_each(fwht_normalized);

        let mut xr: Vec<f32> = x.iter().zip(&signs).map(|(x, &s)| x * s as f32).collect();
        xr.chunks_exact_mut(8).for_each(fwht_normalized);
        let runtime: f32 = stored.iter().zip(&xr).map(|(a, b)| a * b).sum();
        let primal: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((runtime - primal).abs() < 1e-5);

        let folded = stored.clone();
        rot.to_primal("blk.0.attn_q.weight", &mut stored).unwrap();
        for (a, b) in stored.iter().zip(&w) {
            assert!((a - b).abs() < 1e-5);
        }
        // And from_primal folds it back.
        rot.from_primal("blk.0.attn_q.weight", &mut stored).unwrap();
        for (a, b) in stored.iter().zip(&folded) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn errors() {
        let rot = rotation(8, &[1; 16]);
        let mut row = vec![0.0; 16];
        assert!(matches!(
            rot.to_primal("other.weight", &mut row),
            Err(RotationError::NotRotated(_))
        ));
        let mut short = vec![0.0; 12];
        assert!(matches!(
            rot.to_primal("blk.0.attn_q.weight", &mut short),
            Err(RotationError::BadLength { .. })
        ));
        let mut other_width = vec![0.0; 32];
        assert_eq!(
            rot.to_primal("blk.0.attn_q.weight", &mut other_width),
            Err(RotationError::NoSigns(32))
        );
    }
}

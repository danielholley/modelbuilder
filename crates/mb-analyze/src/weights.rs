//! Weight statistics, computed by streaming tensor data a chunk of rows at a time.
//!
//! Two views of each tensor:
//! - **Stored** values (as decoded from the file): zero fraction and ternary
//!   structure, i.e. what the quantizer produced.
//! - **Primal** values (rotation undone with [`WeightRotation::to_primal`] when
//!   the checkpoint folds one in): moments, max, kurtosis and input-channel
//!   outliers, i.e. properties of the function the weights compute.
//!
//! K/V spectra use the stored basis: an orthogonal rotation of the input
//! dimension leaves singular values unchanged.

use mb_formats::dequant::{dequantize, DequantError};
use mb_formats::LoadedModel;
use mb_ir::{AttentionKind, Mixer, ModelIr, RotationError, TensorInfo, TensorKind, TensorRole};
use nalgebra::DMatrix;
use serde::Serialize;

/// Group size used to test for ternary structure (matches Bonsai's g128).
const TERNARY_GROUP: usize = 128;
/// Rows are decoded in chunks of about this many elements.
const CHUNK_ELEMS: usize = 1 << 22;

#[derive(Debug, thiserror::Error)]
pub enum WeightStatsError {
    #[error(transparent)]
    Format(#[from] mb_formats::Error),
    #[error("{name}: {source}")]
    Dequant {
        name: String,
        #[source]
        source: DequantError,
    },
    #[error("{name}: {source}")]
    Rotation {
        name: String,
        #[source]
        source: RotationError,
    },
}

#[derive(Clone, Debug, Default)]
pub struct WeightStatsOptions {
    /// Only tensors whose name contains one of these substrings (all if empty).
    pub only: Vec<String>,
    /// Compute K/V singular-value spectra for attention layers.
    pub kv_spectra: bool,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct TensorStats {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub kind: TensorKind,
    /// Whether primal statistics were computed after undoing a folded rotation.
    pub unrotated: bool,
    // Primal basis.
    pub rms: f64,
    pub mean: f64,
    pub max_abs: f64,
    /// Excess kurtosis (0 for a Gaussian). Large values mean heavy outliers.
    pub kurtosis: f64,
    /// Largest input-channel RMS divided by the median input-channel RMS
    /// (2-D weights only). Large values flag outlier channels.
    pub channel_outlier_ratio: Option<f64>,
    // Stored basis.
    pub zero_fraction: f64,
    /// Fraction of 128-wide row groups whose nonzero values share one magnitude.
    pub ternary_group_fraction: Option<f64>,
}

/// Singular-value summary of one attention layer's K and V projections.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct KvSpectrum {
    pub layer: u32,
    /// Rows of K and of V (`kv_heads × head_dim` each), and the input width.
    pub k_rows: usize,
    pub v_rows: usize,
    pub cols: usize,
    pub k: SpectrumSummary,
    pub v: SpectrumSummary,
    /// `[K; V]` stacked: the rank a shared KV latent (MLA-style) would need.
    pub kv: SpectrumSummary,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct SpectrumSummary {
    /// Smallest rank capturing 90%, 95% and 99% of the squared singular values.
    pub energy_rank_90: usize,
    pub energy_rank_95: usize,
    pub energy_rank_99: usize,
    /// exp(entropy) of the normalized singular values (Roy & Vetterli).
    pub effective_rank: f64,
    pub full_rank: usize,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct WeightStatsReport {
    pub tensors: Vec<TensorStats>,
    pub kv_spectra: Vec<KvSpectrum>,
    pub skipped: Vec<(String, String)>,
    pub notes: Vec<String>,
}

#[derive(Default)]
struct Moments {
    n: u64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    max_abs: f64,
}

impl Moments {
    fn push(&mut self, x: f32) {
        let x = f64::from(x);
        let x2 = x * x;
        self.n += 1;
        self.s1 += x;
        self.s2 += x2;
        self.s3 += x2 * x;
        self.s4 += x2 * x2;
        self.max_abs = self.max_abs.max(x.abs());
    }

    fn finish(&self) -> (f64, f64, f64) {
        let n = self.n.max(1) as f64;
        let mean = self.s1 / n;
        let var = (self.s2 / n - mean * mean).max(0.0);
        let m4 = self.s4 / n - 4.0 * mean * self.s3 / n + 6.0 * mean * mean * self.s2 / n
            - 3.0 * mean.powi(4);
        let kurtosis = if var > 0.0 {
            m4 / (var * var) - 3.0
        } else {
            0.0
        };
        ((self.s2 / n).sqrt(), mean, kurtosis)
    }
}

/// Row width (innermost dimension) and number of rows.
fn row_geometry(t: &TensorInfo) -> (usize, usize) {
    let width = t.shape.last().copied().unwrap_or(1).max(1) as usize;
    (width, (t.n_elements() as usize) / width)
}

fn is_ternary_group(g: &[f32]) -> bool {
    let mut mag = None;
    g.iter().filter(|v| **v != 0.0).all(|v| {
        let a = v.abs();
        *mag.get_or_insert(a) == a
    })
}

fn tensor_stats(
    model: &LoadedModel,
    ir: &ModelIr,
    t: &TensorInfo,
    role: &TensorRole,
) -> Result<TensorStats, WeightStatsError> {
    let (width, rows) = row_geometry(t);
    let bytes = model.tensor_bytes(t)?;
    let row_bytes = bytes.len() / rows.max(1);
    let rotation = ir
        .weight_rotation
        .as_ref()
        .filter(|r| r.is_rotated(&t.name));
    let two_d = t.shape.len() == 2;

    let mut m = Moments::default();
    let mut zeros = 0u64;
    let (mut groups, mut ternary_groups) = (0u64, 0u64);
    let mut col_ss = if two_d { vec![0f64; width] } else { Vec::new() };
    let rows_per_chunk = (CHUNK_ELEMS / width).max(1);
    let mut buf = Vec::with_capacity(rows_per_chunk * width);

    for chunk in bytes.chunks(rows_per_chunk * row_bytes) {
        buf.clear();
        dequantize(t.dtype, chunk, &mut buf).map_err(|source| WeightStatsError::Dequant {
            name: t.name.clone(),
            source,
        })?;
        for row in buf.chunks_exact_mut(width) {
            zeros += row.iter().filter(|v| **v == 0.0).count() as u64;
            if width % TERNARY_GROUP == 0 {
                for g in row.chunks_exact(TERNARY_GROUP) {
                    groups += 1;
                    ternary_groups += u64::from(is_ternary_group(g));
                }
            }
            if let Some(r) = rotation {
                r.to_primal(&t.name, row)
                    .map_err(|source| WeightStatsError::Rotation {
                        name: t.name.clone(),
                        source,
                    })?;
            }
            for (i, &v) in row.iter().enumerate() {
                m.push(v);
                if two_d {
                    col_ss[i] += f64::from(v) * f64::from(v);
                }
            }
        }
    }

    let channel_outlier_ratio = two_d.then(|| {
        let mut rms: Vec<f64> = col_ss.iter().map(|s| (s / rows as f64).sqrt()).collect();
        rms.sort_by(f64::total_cmp);
        let median = rms[rms.len() / 2];
        if median > 0.0 {
            rms[rms.len() - 1] / median
        } else {
            0.0
        }
    });
    let (rms, mean, kurtosis) = m.finish();
    Ok(TensorStats {
        name: t.name.clone(),
        dtype: t.dtype.to_string(),
        shape: t.shape.clone(),
        kind: role.kind,
        unrotated: rotation.is_some(),
        rms,
        mean,
        max_abs: m.max_abs,
        kurtosis,
        channel_outlier_ratio,
        zero_fraction: zeros as f64 / t.n_elements().max(1) as f64,
        ternary_group_fraction: (groups > 0).then(|| ternary_groups as f64 / groups as f64),
    })
}

/// Squared singular values of a row-major `rows × cols` matrix, largest first,
/// via the eigenvalues of the smaller Gram matrix.
pub fn squared_singular_values(data: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let a = DMatrix::from_row_iterator(rows, cols, data.iter().map(|&v| f64::from(v)));
    let gram = if rows <= cols {
        &a * a.transpose()
    } else {
        a.transpose() * &a
    };
    let mut ev: Vec<f64> = gram
        .symmetric_eigenvalues()
        .iter()
        .map(|&e| e.max(0.0))
        .collect();
    ev.sort_by(|x, y| y.total_cmp(x));
    ev
}

pub fn summarize_spectrum(sq: &[f64]) -> SpectrumSummary {
    let total: f64 = sq.iter().sum();
    let rank_at = |p: f64| {
        let mut acc = 0.0;
        sq.iter()
            .position(|&e| {
                acc += e;
                acc >= p * total
            })
            .map_or(sq.len(), |i| i + 1)
    };
    let sigma: Vec<f64> = sq.iter().map(|e| e.sqrt()).collect();
    let sum: f64 = sigma.iter().sum();
    let entropy: f64 = sigma
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| {
            let p = s / sum;
            -p * p.ln()
        })
        .sum();
    SpectrumSummary {
        energy_rank_90: rank_at(0.90),
        energy_rank_95: rank_at(0.95),
        energy_rank_99: rank_at(0.99),
        effective_rank: if sum > 0.0 { entropy.exp() } else { 0.0 },
        full_rank: sq.len(),
    }
}

fn decode_all(model: &LoadedModel, t: &TensorInfo) -> Result<Vec<f32>, WeightStatsError> {
    let mut out = Vec::with_capacity(t.n_elements() as usize);
    dequantize(t.dtype, model.tensor_bytes(t)?, &mut out).map_err(|source| {
        WeightStatsError::Dequant {
            name: t.name.clone(),
            source,
        }
    })?;
    Ok(out)
}

fn kv_spectra(
    model: &LoadedModel,
    ir: &ModelIr,
    report: &mut WeightStatsReport,
) -> Result<(), WeightStatsError> {
    for layer in &ir.layers {
        let Mixer::Attention(a) = &layer.mixer else {
            continue;
        };
        if a.kind == AttentionKind::Mla {
            report.skipped.push((
                format!("layer {} K/V spectrum", layer.index),
                "MLA layers already use a low-rank KV latent".into(),
            ));
            continue;
        }
        let find = |kind| {
            ir.tensors()
                .find(|(t, r)| {
                    r.layer == Some(layer.index) && r.kind == kind && t.name.ends_with("weight")
                })
                .map(|(t, _)| t)
        };
        let (Some(k), Some(v)) = (find(TensorKind::AttnK), find(TensorKind::AttnV)) else {
            report.skipped.push((
                format!("layer {} K/V spectrum", layer.index),
                "no separate K and V projections (fused QKV is not split yet)".into(),
            ));
            continue;
        };
        let (kd, vd) = (decode_all(model, k)?, decode_all(model, v)?);
        let ((cols, k_rows), (_, v_rows)) = (row_geometry(k), row_geometry(v));
        let mut stacked = kd.clone();
        stacked.extend_from_slice(&vd);
        report.kv_spectra.push(KvSpectrum {
            layer: layer.index,
            k_rows,
            v_rows,
            cols,
            k: summarize_spectrum(&squared_singular_values(&kd, k_rows, cols)),
            v: summarize_spectrum(&squared_singular_values(&vd, v_rows, cols)),
            kv: summarize_spectrum(&squared_singular_values(&stacked, k_rows + v_rows, cols)),
        });
    }
    Ok(())
}

/// Streams every selected tensor once and computes its statistics.
pub fn weight_stats(
    model: &LoadedModel,
    ir: &ModelIr,
    opts: &WeightStatsOptions,
) -> Result<WeightStatsReport, WeightStatsError> {
    let selected =
        |name: &str| opts.only.is_empty() || opts.only.iter().any(|s| name.contains(s.as_str()));
    let mut report = WeightStatsReport {
        tensors: Vec::new(),
        kv_spectra: Vec::new(),
        skipped: Vec::new(),
        notes: Vec::new(),
    };
    if let Some(r) = &ir.weight_rotation {
        report.notes.push(format!(
            "{} tensors are stored in a rotated basis ({}); their moments and channel statistics are computed after undoing it. Zero fraction and ternary structure use the stored values.",
            r.rotated_tensors + r.inverse_tensors,
            r.scheme
        ));
        if r.gdn_v_grouped {
            report.notes.push(
                "ssm_out input channels are reported in grouped (HF training) V-head order.".into(),
            );
        }
    }
    for (t, role) in ir.tensors().filter(|(t, _)| selected(&t.name)) {
        match tensor_stats(model, ir, t, role) {
            Ok(s) => report.tensors.push(s),
            Err(WeightStatsError::Dequant {
                source: DequantError::Unsupported(d),
                ..
            }) => {
                report
                    .skipped
                    .push((t.name.clone(), format!("decoding {d} is not supported")));
            }
            Err(e) => return Err(e),
        }
    }
    if opts.kv_spectra {
        kv_spectra(model, ir, &mut report)?;
        report.notes.push(
            "K/V spectra are computed in the stored basis; orthogonal weight rotations do not change singular values.".into(),
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_of_a_known_sample() {
        let mut m = Moments::default();
        for x in [1.0, -1.0, 1.0, -1.0] {
            m.push(x);
        }
        let (rms, mean, kurt) = m.finish();
        assert_eq!((rms, mean, m.max_abs), (1.0, 0.0, 1.0));
        assert!((kurt + 2.0).abs() < 1e-12); // two-point distribution
    }

    #[test]
    fn ternary_groups() {
        assert!(is_ternary_group(&[0.5, -0.5, 0.0, 0.5]));
        assert!(is_ternary_group(&[0.0, 0.0]));
        assert!(!is_ternary_group(&[0.5, 0.25]));
    }

    #[test]
    fn spectrum_of_a_rank_two_matrix() {
        // rows = combinations of two orthogonal directions, scaled 3 and 1.
        let (rows, cols) = (6, 8);
        let u: Vec<f32> = (0..cols)
            .map(|c| if c % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let w: Vec<f32> = (0..cols).map(|c| if c < 4 { 1.0 } else { -1.0 }).collect();
        let mut data = Vec::new();
        for r in 0..rows {
            // Energy split ≈ 97% / 3%, so rank 1 covers 95% but not 99%.
            let a = 3.0 * (r as f32 + 1.0);
            let b = if r % 2 == 0 { 2.0 } else { -2.0 };
            data.extend((0..cols).map(|c| a * u[c] + b * w[c]));
        }
        let sq = squared_singular_values(&data, rows, cols);
        assert!(sq[2] < 1e-6 * sq[0]);
        let s = summarize_spectrum(&sq);
        assert_eq!(
            (s.energy_rank_95, s.energy_rank_99, s.full_rank),
            (1, 2, rows)
        );
        assert!(s.effective_rank > 1.0 && s.effective_rank < 2.0);
    }

    #[test]
    fn singular_values_are_rotation_invariant() {
        let (rows, cols) = (5, 16);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut rotated = data.clone();
        rotated.chunks_exact_mut(8).for_each(mb_ir::fwht_normalized);
        let (a, b) = (
            squared_singular_values(&data, rows, cols),
            squared_singular_values(&rotated, rows, cols),
        );
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4 * a[0], "{x} vs {y}");
        }
    }
}

//! GQA/MHA → multi-head latent attention (MLA) initialization, following
//! MHA2MLA (Ji et al., 2025, arXiv 2502.14837):
//!
//! - **Partial RoPE (§3.1):** each KV head keeps RoPE on `rope_dim` of its
//!   dimensions (`rope_dim / 2` frequency pairs), chosen without data:
//!   `high` keeps the highest frequencies (S_high), `uniform` spaces them
//!   evenly over the spectrum (S_uniform). The other dimensions lose RoPE in
//!   both Q and K.
//! - **Joint SVD (§3.2, SVD_joint):** the RoPE-free key rows of all KV heads
//!   and the value rows are stacked, `W = [W_k,nope; W_v]`, and factored as
//!   `W ≈ kv_b · kv_a` with rank `kv_rank` (balanced: `kv_b = U Σ^½`,
//!   `kv_a = Σ^½ Vᵀ`). The cache then holds the `kv_rank` latent plus
//!   `n_kv × rope_dim` RoPE key values per token and layer.
//!
//! The input is a Hugging Face checkpoint in the primal basis (e.g. from
//! `modelbuilder export-hf`). The output directory links every source file
//! (the original weights are the teacher during recovery training), adds the
//! latent projections in `mla.safetensors` and records the conversion under
//! `modelbuilder_mla` in `config.json`. The Python side's `trunk_distill`
//! stage reads that key and trains the converted attention against the
//! original. With `kv_rank` at full rank and every rotary dimension kept,
//! the conversion is exact.
//!
//! Every attention layer is converted: partial RoPE is applied through the
//! model's one rotary embedding, so a layer left out would lose RoPE
//! dimensions too.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mb_formats::{safetensors, LoadedModel, TensorToWrite};
use mb_ir::DType;
use nalgebra::DMatrix;
use serde::Serialize;
use serde_json::{json, Value};

use crate::SurgeryError;

pub const CONFIG_KEY: &str = "modelbuilder_mla";
pub const TENSORS_FILE: &str = "mla.safetensors";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeSelect {
    /// The highest-frequency pairs (MHA2MLA's S_high).
    High,
    /// Pairs spaced evenly over the spectrum (S_uniform).
    Uniform,
}

impl std::str::FromStr for RopeSelect {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "high" => Ok(Self::High),
            "uniform" => Ok(Self::Uniform),
            _ => Err(format!("unknown RoPE selection `{s}`; use high or uniform")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MlaOptions {
    pub kv_rank: usize,
    /// RoPE dimensions kept per KV head (even).
    pub rope_dim: usize,
    pub select: RopeSelect,
    pub overwrite: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlaLayer {
    pub layer: u32,
    /// Fraction of `‖[W_k,nope; W_v]‖²_F` the rank-`kv_rank` factorization keeps.
    pub energy_kept: f64,
    /// Relative Frobenius error of the factorization.
    pub rel_error: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlaReport {
    pub output: PathBuf,
    pub layers: Vec<MlaLayer>,
    /// Cached values per token and layer, before and after.
    pub cache_before: usize,
    pub cache_after: usize,
    /// Frequency indices (in `0..rotary_dim/2`) that keep RoPE.
    pub rope_freqs: Vec<usize>,
    pub notes: Vec<String>,
}

fn find<'a>(t: &'a Value, k: &str) -> Option<&'a Value> {
    t.get(k)
        .or_else(|| t.get("rope_parameters").and_then(|r| r.get(k)))
        .or_else(|| t.get("rope_scaling").and_then(|r| r.get(k)))
}

/// The frequency indices to keep.
pub fn select_freqs(half: usize, keep: usize, select: RopeSelect) -> Vec<usize> {
    match select {
        // inv_freq[i] = base^(-2i/rotary_dim) falls with i: the first pairs rotate fastest.
        RopeSelect::High => (0..keep).collect(),
        RopeSelect::Uniform => (0..keep).map(|j| j * half / keep.max(1)).collect(),
    }
}

/// `W ≈ B · A` with rank `r`, balanced (`B = U Σ^½`, `A = Σ^½ Vᵀ`), via the
/// eigendecomposition of the smaller Gram matrix. Returns (A, B, energy kept).
pub fn factor(w: &DMatrix<f64>, r: usize) -> (DMatrix<f64>, DMatrix<f64>, f64) {
    let (m, n) = w.shape();
    let total: f64 = w.iter().map(|x| x * x).sum();
    let left = m <= n;
    let gram = if left {
        w * w.transpose()
    } else {
        w.transpose() * w
    };
    let eig = gram.symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let mut a = DMatrix::zeros(r, n);
    let mut b = DMatrix::zeros(m, r);
    let mut kept = 0.0;
    for (j, &i) in order.iter().take(r).enumerate() {
        let s2 = eig.eigenvalues[i].max(0.0);
        let s = s2.sqrt();
        if s <= f64::EPSILON * total.sqrt() {
            continue;
        }
        kept += s2;
        let vec = eig.eigenvectors.column(i);
        if left {
            // u = vec; vᵀ = uᵀ W / σ.
            let vt = vec.transpose() * w / s;
            b.set_column(j, &(vec * s.sqrt()));
            a.set_row(j, &(vt * s.sqrt()));
        } else {
            // v = vec; u = W v / σ.
            let u = w * vec / s;
            a.set_row(j, &(vec.transpose() * s.sqrt()));
            b.set_column(j, &(u * s.sqrt()));
        }
    }
    let energy = if total > 0.0 { kept / total } else { 1.0 };
    (a, b, energy)
}

fn matrix(model: &LoadedModel, name: &str) -> Result<DMatrix<f64>, SurgeryError> {
    let t = model
        .tensor(name)
        .ok_or_else(|| SurgeryError::Incompatible(format!("no tensor {name}")))?;
    let mut v = Vec::new();
    mb_formats::dequant::dequantize(t.dtype, model.tensor_bytes(t)?, &mut v).map_err(|e| {
        SurgeryError::Dequant {
            name: name.into(),
            source: e,
        }
    })?;
    let (rows, cols) = match t.shape[..] {
        [r, c] => (r as usize, c as usize),
        [r] => (r as usize, 1),
        _ => {
            return Err(SurgeryError::Incompatible(format!(
                "{name}: expected a matrix, found {:?}",
                t.shape
            )))
        }
    };
    Ok(DMatrix::from_row_iterator(
        rows,
        cols,
        v.into_iter().map(f64::from),
    ))
}

fn rows_of(m: &DMatrix<f64>, rows: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(rows.len(), m.ncols(), |i, j| m[(rows[i], j)])
}

fn f32_bytes(m: &DMatrix<f64>) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.len() * 4);
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.extend((m[(i, j)] as f32).to_le_bytes());
        }
    }
    out
}

#[cfg(unix)]
fn link(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dst)
}

#[cfg(not(unix))]
fn link(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::copy(src, dst).map(|_| ())
}

/// Converts every attention layer of the HF checkpoint in `src` to MLA and
/// writes the result to `out`.
pub fn convert(
    model: &LoadedModel,
    src: &Path,
    opts: &MlaOptions,
    out: &Path,
) -> Result<MlaReport, SurgeryError> {
    let io = |e: std::io::Error| SurgeryError::Output(e.to_string());
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(src.join("config.json")).map_err(|e| {
            SurgeryError::Incompatible(format!("{}: {e}", src.join("config.json").display()))
        })?,
    )
    .map_err(|e| SurgeryError::Incompatible(format!("config.json: {e}")))?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    if t.get(CONFIG_KEY).is_some() {
        return Err(SurgeryError::Incompatible(
            "the checkpoint is already converted to MLA".into(),
        ));
    }
    let need = |k: &str| {
        t.get(k)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| SurgeryError::Incompatible(format!("config has no `{k}`")))
    };
    let n_head = need("num_attention_heads")?;
    let hidden = need("hidden_size")?;
    let n_kv = t
        .get("num_key_value_heads")
        .and_then(Value::as_u64)
        .map_or(n_head, |v| v as usize);
    let hd = t
        .get("head_dim")
        .and_then(Value::as_u64)
        .map_or(hidden / n_head.max(1), |v| v as usize);
    let factor_ = find(t, "partial_rotary_factor")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let rot = ((hd as f64) * factor_).round() as usize;
    if rot % 2 != 0 || rot == 0 || rot > hd {
        return Err(SurgeryError::Incompatible(format!(
            "rotary dim {rot} (head dim {hd} × partial_rotary_factor {factor_}) must be even and in 1..={hd}"
        )));
    }
    let r = opts.rope_dim;
    if r % 2 != 0 || r == 0 || r > rot {
        return Err(SurgeryError::Incompatible(format!(
            "rope_dim {r} must be even and in 2..={rot} (the model's rotary dims per head)"
        )));
    }
    let kv_rows = n_kv * (hd - r) + n_kv * hd;
    if opts.kv_rank == 0 || opts.kv_rank > kv_rows.min(hidden) {
        return Err(SurgeryError::Incompatible(format!(
            "kv_rank {} must be in 1..={} (min of the {kv_rows} stacked K/V rows and hidden {hidden})",
            opts.kv_rank,
            kv_rows.min(hidden)
        )));
    }

    // Attention layers: every tensor named `…layers.{i}.self_attn.k_proj.weight`.
    let mut layers: Vec<(u32, String)> = model
        .raw
        .tensors
        .iter()
        .filter_map(|x| {
            let prefix = x.name.strip_suffix("k_proj.weight")?;
            let before = prefix.strip_suffix("self_attn.")?;
            let idx = before.strip_suffix('.')?.rsplit('.').next()?.parse().ok()?;
            Some((idx, prefix.to_string()))
        })
        .collect();
    layers.sort();
    if layers.is_empty() {
        return Err(SurgeryError::Incompatible(
            "no attention layers (`…self_attn.k_proj.weight`)".into(),
        ));
    }

    let freqs = select_freqs(rot / 2, r / 2, opts.select);
    let mut rope_dims: Vec<usize> = freqs.iter().flat_map(|&i| [i, i + rot / 2]).collect();
    rope_dims.sort_unstable();
    let nope_dims: Vec<usize> = (0..hd).filter(|d| !rope_dims.contains(d)).collect();
    let head_rows = |dims: &[usize]| -> Vec<usize> {
        (0..n_kv)
            .flat_map(|h| dims.iter().map(move |&d| h * hd + d))
            .collect()
    };
    let (nope_rows, rope_rows) = (head_rows(&nope_dims), head_rows(&rope_dims));

    if out.exists() {
        if !opts.overwrite {
            return Err(SurgeryError::Output(format!(
                "{} exists (pass overwrite)",
                out.display()
            )));
        }
        if std::fs::canonicalize(out).ok() == std::fs::canonicalize(src).ok() {
            return Err(SurgeryError::Output(
                "the output is the input directory".into(),
            ));
        }
        std::fs::remove_dir_all(out).map_err(io)?;
    }
    std::fs::create_dir_all(out).map_err(io)?;

    let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut report_layers = Vec::new();
    for (idx, prefix) in &layers {
        let k = matrix(model, &format!("{prefix}k_proj.weight"))?;
        let v = matrix(model, &format!("{prefix}v_proj.weight"))?;
        if k.shape() != (n_kv * hd, hidden) || v.shape() != (n_kv * hd, hidden) {
            return Err(SurgeryError::Incompatible(format!(
                "layer {idx}: K {:?} / V {:?} don't match {n_kv} KV heads × {hd} × hidden {hidden}",
                k.shape(),
                v.shape()
            )));
        }
        let knope = rows_of(&k, &nope_rows);
        let mut w = DMatrix::zeros(kv_rows, hidden);
        w.rows_mut(0, knope.nrows()).copy_from(&knope);
        w.rows_mut(knope.nrows(), v.nrows()).copy_from(&v);
        let (a, b, energy) = factor(&w, opts.kv_rank);
        let err = (&b * &a - &w).norm() / w.norm().max(f64::MIN_POSITIVE);
        report_layers.push(MlaLayer {
            layer: *idx,
            energy_kept: energy,
            rel_error: err,
        });
        let krope = rows_of(&k, &rope_rows);
        let base = format!("{prefix}mla.");
        tensors.push((
            format!("{base}kv_a.weight"),
            vec![opts.kv_rank as u64, hidden as u64],
            f32_bytes(&a),
        ));
        tensors.push((
            format!("{base}kv_b.weight"),
            vec![kv_rows as u64, opts.kv_rank as u64],
            f32_bytes(&b),
        ));
        tensors.push((
            format!("{base}k_rope.weight"),
            vec![rope_rows.len() as u64, hidden as u64],
            f32_bytes(&krope),
        ));
        let kb = model.tensor(&format!("{prefix}k_proj.bias")).is_some();
        let vb = model.tensor(&format!("{prefix}v_proj.bias")).is_some();
        if kb || vb {
            let col = |name: &str, present: bool| -> Result<Vec<f64>, SurgeryError> {
                if present {
                    Ok(matrix(model, name)?.iter().copied().collect())
                } else {
                    Ok(vec![0.0; n_kv * hd])
                }
            };
            let kbias = col(&format!("{prefix}k_proj.bias"), kb)?;
            let vbias = col(&format!("{prefix}v_proj.bias"), vb)?;
            let kvb: Vec<f64> = nope_rows
                .iter()
                .map(|&i| kbias[i])
                .chain(vbias.iter().copied())
                .collect();
            let rb: Vec<f64> = rope_rows.iter().map(|&i| kbias[i]).collect();
            let bytes = |x: &[f64]| -> Vec<u8> {
                x.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect()
            };
            tensors.push((
                format!("{base}kv_b.bias"),
                vec![kvb.len() as u64],
                bytes(&kvb),
            ));
            tensors.push((
                format!("{base}k_rope.bias"),
                vec![rb.len() as u64],
                bytes(&rb),
            ));
        }
    }
    let to_write: Vec<TensorToWrite> = tensors
        .iter()
        .map(|(n, s, d)| TensorToWrite {
            name: n.clone(),
            dtype: DType::F32,
            shape: s.clone(),
            data: Cow::Borrowed(d),
        })
        .collect();
    let meta = BTreeMap::from([
        ("format".to_string(), "pt".to_string()),
        ("modelbuilder".to_string(), "mla-init".to_string()),
    ]);
    safetensors::write(&out.join(TENSORS_FILE), &to_write, &meta)?;

    // Link every other source file (weights, tokenizer, manifests).
    let src_abs = std::fs::canonicalize(src).map_err(io)?;
    for e in std::fs::read_dir(&src_abs).map_err(io)? {
        let e = e.map_err(io)?;
        let name = e.file_name();
        if name == "config.json" || name == TENSORS_FILE || !e.path().is_file() {
            continue;
        }
        link(&e.path(), &out.join(&name)).map_err(io)?;
    }
    let mla = json!({
        "layers": layers.iter().map(|(i, _)| i).collect::<Vec<_>>(),
        "kv_rank": opts.kv_rank,
        "rope_dim": r,
        "rope_select": opts.select,
        "rotary_dim": rot,
        "head_dim": hd,
        "num_key_value_heads": n_kv,
        "rope_freqs": freqs,
        "rope_dims": rope_dims,
        "nope_dims": nope_dims,
        "tensors": TENSORS_FILE,
        "method": "MHA2MLA partial RoPE + joint SVD (arXiv 2502.14837)",
    });
    let mut new_cfg = cfg.clone();
    let target = if new_cfg.get("text_config").is_some() {
        &mut new_cfg["text_config"]
    } else {
        &mut new_cfg
    };
    target[CONFIG_KEY] = mla;
    std::fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&new_cfg).expect("config serializes"),
    )
    .map_err(io)?;

    let before = 2 * n_kv * hd;
    let after = opts.kv_rank + n_kv * r;
    let worst = report_layers
        .iter()
        .map(|l| l.energy_kept)
        .fold(1.0, f64::min);
    Ok(MlaReport {
        output: out.to_owned(),
        layers: report_layers,
        cache_before: before,
        cache_after: after,
        rope_freqs: freqs,
        notes: vec![
            format!(
                "Converted {} attention layers: the cache holds {after} values per token and layer instead of {before} ({:.1}× smaller).",
                layers.len(),
                before as f64 / after as f64
            ),
            format!("Lowest K/V energy kept by the rank-{} factorization: {:.2}%.", opts.kv_rank, 100.0 * worst),
            format!("RoPE kept on {r} of {rot} rotary dimensions per head ({:?} selection); the rest lose RoPE in Q and K.", opts.select),
            "Original weights are linked, not copied: they are the teacher for recovery training (`trunk_distill` with trainable `self_attn.mla.`).".into(),
            "No runtime runs this layout yet: llama.cpp and vLLM run MLA only in DeepSeek's architecture.".into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_is_exact_at_full_rank_and_optimal_when_truncated() {
        let (m, n) = (6, 9);
        let w = DMatrix::from_fn(m, n, |i, j| {
            ((i * 7 + j * 3) % 11) as f64 - 5.0 + (i * j) as f64 * 0.1
        });
        for wt in [w.clone(), w.transpose()] {
            let full = wt.nrows().min(wt.ncols());
            let (a, b, e) = factor(&wt, full);
            assert!((&b * &a - &wt).norm() / wt.norm() < 1e-9);
            assert!((e - 1.0).abs() < 1e-9);
            // Truncated: the error equals the dropped singular values (Eckart–Young).
            let sv = wt.clone().svd(false, false).singular_values;
            let mut s: Vec<f64> = sv.iter().copied().collect();
            s.sort_by(|x, y| y.total_cmp(x));
            let (a, b, _) = factor(&wt, 2);
            let err = (&b * &a - &wt).norm();
            let want = s[2..].iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((err - want).abs() < 1e-7 * wt.norm(), "{err} vs {want}");
        }
    }

    #[test]
    fn frequency_selection() {
        assert_eq!(select_freqs(32, 4, RopeSelect::High), [0, 1, 2, 3]);
        assert_eq!(select_freqs(32, 4, RopeSelect::Uniform), [0, 8, 16, 24]);
        assert_eq!(select_freqs(4, 4, RopeSelect::Uniform), [0, 1, 2, 3]);
    }
}

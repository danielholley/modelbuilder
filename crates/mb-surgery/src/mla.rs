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
    /// Pairs given explicitly, e.g. ranked by query/key norms on calibration
    /// text (S_2-norm, `modelbuilder_train mla-rope-scores`).
    Given,
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
    /// Frequency indices to keep (overrides `select`).
    pub freqs: Option<Vec<usize>>,
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
        RopeSelect::Uniform | RopeSelect::Given => {
            (0..keep).map(|j| j * half / keep.max(1)).collect()
        }
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
            // trunk layers only: an MTP block (`mtp.layers.N`) isn't converted
            if x.name.starts_with("mtp.") || x.name.contains(".mtp.") {
                return None;
            }
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

    let (freqs, select) = match &opts.freqs {
        Some(f) => {
            let mut f = f.clone();
            f.sort_unstable();
            f.dedup();
            if f.len() != r / 2 || f.iter().any(|&i| i >= rot / 2) {
                return Err(SurgeryError::Incompatible(format!(
                    "freqs must be {} distinct indices below {} (rope_dim {r} / 2, rotary dim {rot} / 2)",
                    r / 2,
                    rot / 2
                )));
            }
            (f, RopeSelect::Given)
        }
        None => (select_freqs(rot / 2, r / 2, opts.select), opts.select),
    };
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
        if name == "model.safetensors.index.json" {
            // the shard index lists the latent too, so the directory reads as one checkpoint
            let mut idx: Value =
                serde_json::from_str(&std::fs::read_to_string(e.path()).map_err(io)?)
                    .map_err(|e| SurgeryError::Incompatible(format!("shard index: {e}")))?;
            if let Some(map) = idx.get_mut("weight_map").and_then(Value::as_object_mut) {
                for (n, _, _) in &tensors {
                    map.insert(n.clone(), json!(TENSORS_FILE));
                }
            }
            std::fs::write(
                out.join(&name),
                serde_json::to_string_pretty(&idx).expect("json"),
            )
            .map_err(io)?;
            continue;
        }
        link(&e.path(), &out.join(&name)).map_err(io)?;
    }
    let mla = json!({
        "layers": layers.iter().map(|(i, _)| i).collect::<Vec<_>>(),
        "kv_rank": opts.kv_rank,
        "rope_dim": r,
        "rope_select": select,
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
            format!("RoPE kept on {r} of {rot} rotary dimensions per head ({select:?} selection); the rest lose RoPE in Q and K."),
            "Original weights are linked, not copied: they are the teacher for recovery training (`trunk_distill` with trainable `self_attn.mla.`).".into(),
            "Upstream llama.cpp and vLLM run MLA only in DeepSeek's architecture; `surgery mla-gguf` writes this layout for the patched fork (patches/prism-llama.cpp/).".into(),
        ],
    })
}

/// Writes an MLA conversion into a copy of the GGUF it came from, in the
/// layout of the fork patch `patches/prism-llama.cpp/0003-mla.patch`: for
/// each converted layer the latent projections (`attn_mla_kv_a`, `_kv_b`,
/// `_k_rope`, F32), the K-norm weights split into RoPE and other dims, the
/// head permutation `[rope dims; nope dims]` applied to Q, and RoPE frequency
/// factors, with `attn_k`/`attn_v`/`attn_k_norm` removed. `mla_dir` is a
/// `surgery mla` output (its `config.json` and `mla.safetensors`); `updates`,
/// if given, holds trained latent tensors (from `job trunk-distill
/// --mla-rank`) that replace the initial ones. Other tensors are copied byte
/// for byte.
pub fn write_gguf(
    model: &LoadedModel,
    ir: &mb_ir::ModelIr,
    mla_dir: &Path,
    updates: Option<&LoadedModel>,
    out: &Path,
    overwrite: bool,
) -> Result<crate::SurgeryReport, SurgeryError> {
    use mb_ir::{MetaType, MetaValue, Metadata};
    let (mut kv, arch) = crate::gguf_edit::gguf_kv(ir)?;
    crate::gguf_edit::check_output(ir, out, overwrite)?;
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(mla_dir.join("config.json"))
            .map_err(|e| SurgeryError::Incompatible(format!("{}: {e}", mla_dir.display())))?,
    )
    .map_err(|e| SurgeryError::Incompatible(e.to_string()))?;
    let m = cfg
        .get("text_config")
        .and_then(|t| t.get(CONFIG_KEY))
        .or_else(|| cfg.get(CONFIG_KEY))
        .ok_or_else(|| {
            SurgeryError::Incompatible(format!("{} has no {CONFIG_KEY}", mla_dir.display()))
        })?;
    let ints = |k: &str| -> Vec<usize> {
        m[k].as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default()
    };
    let (layers, rope_dims, nope_dims, freqs) = (
        ints("layers"),
        ints("rope_dims"),
        ints("nope_dims"),
        ints("rope_freqs"),
    );
    let num = |k: &str| m[k].as_u64().unwrap_or(0) as usize;
    let (rank, r, rot, hd, n_kv) = (
        num("kv_rank"),
        num("rope_dim"),
        num("rotary_dim"),
        num("head_dim"),
        num("num_key_value_heads"),
    );
    if layers.is_empty() || r == 0 || rope_dims.len() != r || nope_dims.len() != hd - r {
        return Err(SurgeryError::Incompatible(format!(
            "{CONFIG_KEY} is incomplete"
        )));
    }
    let latent = safetensors::open(mla_dir).ok();
    let source = |name_end: &str| -> Result<DMatrix<f64>, SurgeryError> {
        for src in updates.into_iter().chain(latent.as_ref()) {
            if let Some(t) = src.raw.tensors.iter().find(|t| t.name.ends_with(name_end)) {
                return matrix(src, &t.name);
            }
        }
        Err(SurgeryError::Incompatible(format!(
            "no tensor ending in {name_end}"
        )))
    };
    let get_meta = |k: &str| match &ir.raw.metadata {
        Metadata::Gguf { kv, .. } => kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()),
        _ => None,
    };
    let base = match get_meta(&format!("{arch}.rope.freq_base")) {
        Some(MetaValue::F32(f)) => f64::from(f),
        Some(MetaValue::F64(f)) => f,
        _ => 10_000.0,
    };
    let n_layers = get_meta(&format!("{arch}.block_count"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // per layer: (name, shape, type, bytes) of the tensors added
    type Added = Vec<(String, Vec<u64>, DType, Vec<u8>)>;
    let mut added: BTreeMap<u32, Added> = BTreeMap::new();
    for &l in &layers {
        let b = |s: &str| format!("blk.{l}.{s}");
        let hfb = |s: &str| format!("layers.{l}.self_attn.mla.{s}");
        if updates.is_some_and(|u| {
            u.raw
                .tensors
                .iter()
                .any(|t| t.name.ends_with(&hfb("kv_b.bias")))
        }) || latent.as_ref().is_some_and(|u| {
            u.raw
                .tensors
                .iter()
                .any(|t| t.name.ends_with(&hfb("kv_b.bias")))
        }) {
            return Err(SurgeryError::Unsupported(
                "K/V biases have no MLA runtime path yet".into(),
            ));
        }
        let kv_a = source(&hfb("kv_a.weight"))?;
        let kv_b = source(&hfb("kv_b.weight"))?;
        let k_rope = source(&hfb("k_rope.weight"))?;
        let k_norm: Vec<f64> = matrix(model, &b("attn_k_norm.weight"))?
            .iter()
            .copied()
            .collect();
        let vec_bytes =
            |v: &[f64]| -> Vec<u8> { v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect() };
        let order: Vec<usize> = rope_dims.iter().chain(&nope_dims).copied().collect();
        let mut perm = DMatrix::zeros(hd, hd);
        for (o, &i) in order.iter().enumerate() {
            perm[(o, i)] = 1.0;
        }
        let ff_q: Vec<f64> = (0..rot / 2)
            .map(|i| if freqs.contains(&i) { 1.0 } else { 1e30 })
            .collect();
        let ff_k: Vec<f64> = freqs
            .iter()
            .enumerate()
            .map(|(j, &k)| {
                base.powf(-2.0 * j as f64 / r as f64) / base.powf(-2.0 * k as f64 / rot as f64)
            })
            .collect();
        let shape = |m: &DMatrix<f64>| vec![m.nrows() as u64, m.ncols() as u64];
        added.insert(
            l as u32,
            vec![
                (
                    b("attn_mla_kv_a.weight"),
                    shape(&kv_a),
                    DType::F32,
                    f32_bytes(&kv_a),
                ),
                (
                    b("attn_mla_kv_b.weight"),
                    shape(&kv_b),
                    DType::F32,
                    f32_bytes(&kv_b),
                ),
                (
                    b("attn_mla_k_rope.weight"),
                    shape(&k_rope),
                    DType::F32,
                    f32_bytes(&k_rope),
                ),
                (
                    b("attn_mla_k_rope_norm.weight"),
                    vec![r as u64],
                    DType::F32,
                    vec_bytes(&rope_dims.iter().map(|&d| k_norm[d]).collect::<Vec<_>>()),
                ),
                (
                    b("attn_mla_k_nope_norm.weight"),
                    vec![(hd - r) as u64],
                    DType::F32,
                    vec_bytes(&nope_dims.iter().map(|&d| k_norm[d]).collect::<Vec<_>>()),
                ),
                (
                    b("attn_mla_perm.weight"),
                    vec![hd as u64, hd as u64],
                    DType::F32,
                    f32_bytes(&perm),
                ),
                (
                    b("attn_mla_ff_q.weight"),
                    vec![(rot / 2) as u64],
                    DType::F32,
                    vec_bytes(&ff_q),
                ),
                (
                    b("attn_mla_ff_k.weight"),
                    vec![(r / 2) as u64],
                    DType::F32,
                    vec_bytes(&ff_k),
                ),
            ],
        );
        if kv_b.nrows() != n_kv * (hd - r) + n_kv * hd || kv_a.nrows() != rank {
            return Err(SurgeryError::Incompatible(format!(
                "layer {l}: latent shapes don't match {CONFIG_KEY}"
            )));
        }
    }

    let dropped = |name: &str| {
        crate::gguf_edit::layer_of(name).is_some_and(|(l, rest)| {
            added.contains_key(&l)
                && matches!(
                    rest,
                    "attn_k.weight"
                        | "attn_v.weight"
                        | "attn_k_norm.weight"
                        | "attn_k.bias"
                        | "attn_v.bias"
                )
        })
    };
    for (_, v) in kv.iter_mut() {
        if let MetaValue::Array { values, .. } = v {
            values.retain(|x| !x.as_str().is_some_and(dropped));
        }
    }
    let mut ranks = vec![0u32; n_layers.max(layers.iter().max().map_or(0, |l| l + 1))];
    for &l in &layers {
        ranks[l] = rank as u32;
    }
    crate::gguf_edit::set(
        &mut kv,
        &format!("{arch}.attention.mla_rank"),
        MetaValue::Array {
            elem: MetaType::U32,
            values: ranks.iter().map(|&x| MetaValue::U32(x)).collect(),
        },
    );
    crate::gguf_edit::set(
        &mut kv,
        &format!("{arch}.attention.mla_rope_dim"),
        MetaValue::U32(r as u32),
    );
    crate::gguf_edit::set(
        &mut kv,
        &format!("{arch}.attention.mla_rope_freqs"),
        MetaValue::Array {
            elem: MetaType::I32,
            values: freqs.iter().map(|&x| MetaValue::I32(x as i32)).collect(),
        },
    );

    let mut tensors: Vec<TensorToWrite> = Vec::new();
    let mut written = Vec::new();
    for t in &ir.raw.tensors {
        if dropped(&t.name) {
            continue;
        }
        tensors.push(TensorToWrite {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: Cow::Borrowed(model.tensor_bytes(t)?),
        });
        if t.name.ends_with("attn_q_norm.weight") {
            if let Some((l, _)) = crate::gguf_edit::layer_of(&t.name) {
                for (name, shape, dtype, data) in added.get(&l).into_iter().flatten() {
                    written.push(crate::WrittenTensor {
                        name: name.clone(),
                        dtype: dtype.to_string(),
                        shape: shape.clone(),
                        source: format!("mla:{}", mla_dir.display()),
                        transform: "MLA latent layout".into(),
                    });
                    tensors.push(TensorToWrite {
                        name: name.clone(),
                        dtype: *dtype,
                        shape: shape.clone(),
                        data: Cow::Borrowed(data),
                    });
                }
            }
        }
    }
    if written.len() != added.values().map(Vec::len).sum::<usize>() {
        return Err(SurgeryError::Incompatible(
            "a converted layer has no attn_q_norm in the GGUF".into(),
        ));
    }
    mb_formats::gguf::write(out, &kv, &tensors)?;
    Ok(crate::SurgeryReport {
        output: out.to_owned(),
        bytes: std::fs::metadata(out).map_err(|e| SurgeryError::Output(e.to_string()))?.len(),
        tensors: written,
        metadata: vec![format!("{arch}.attention.mla_rank / mla_rope_dim / mla_rope_freqs")],
        notes: vec![
            format!("{} layers cache {} values per token instead of {}.", layers.len(), rank + n_kv * (r + 1), 2 * n_kv * hd),
            "Runs with patches/prism-llama.cpp/ 0003 (qwen35 graphs) and 0005 (qwen3 graphs). The latent tensors are F32; quantizing them is future work.".into(),
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

//! Export a `qwen35` GGUF (e.g. PrismML's Ternary-Bonsai-2-27B) as a
//! Hugging Face checkpoint that transformers' `Qwen3_5ForCausalLM` loads, so
//! the trunk can run in PyTorch (frozen for MTP alignment, or trained for QAT).
//!
//! It inverts, tensor by tensor, what the PrismML fork's converter does
//! (`conversion/qwen.py`: `Qwen3NextModel.modify_tensors` and
//! `_LinearAttentionVReorderBase.modify_tensors`):
//!
//! - names: `blk.N.attn_qkv` → `model.layers.N.linear_attn.in_proj_qkv`, …;
//! - zero-centered RMSNorm weights: the converter adds 1 (except the gated
//!   `linear_attn.norm`), so 1 is subtracted;
//! - `A_log`: stored as `-exp(A_log)`, so `A_log = ln(-a)`;
//! - `conv1d`: squeezed from `[C, 1, K]`, so the middle axis comes back;
//! - linear-attention V heads: reordered from HF's grouped order to ggml's
//!   tiled order when `num_k_heads != num_v_heads`, so they are reordered back
//!   (rows of `in_proj_qkv`'s V part, `in_proj_z`, `in_proj_a/b`; entries of
//!   `A_log` and `dt_bias`; `conv1d`'s V channels; `out_proj`'s columns unless
//!   the Hadamard fold kept them grouped, `prism.hadamard.gdn_v_grouped`).
//!
//! PQ2_0/PTQ1_0 weights are decoded and the folded rotation is undone
//! (`WeightRotation::to_primal`), so the export is in the primal basis.
//! Tensors are streamed a chunk of rows at a time into sharded safetensors.
//!
//! The architecture comes from the reference HF model's `config.json` (for
//! Bonsai 2: Qwen3.8-27B, whose architecture it keeps), checked against the
//! GGUF's shapes. Tokenizer files are copied from the reference.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use mb_formats::dequant::dequantize;
use mb_formats::safetensors::{write_streaming, TensorHeader};
use mb_formats::LoadedModel;
use mb_ir::{DType, ModelIr, TensorInfo};
use serde_json::{json, Value};

use crate::{f32_to_bf16, SurgeryError, SurgeryReport, WrittenTensor};

const CHUNK_ELEMS: usize = 1 << 22;
const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
];

#[derive(Clone, Debug)]
pub struct HfExportOptions {
    /// Reference HF model directory: its `config.json` (and tokenizer files)
    /// describe the architecture the GGUF was converted from.
    pub reference: PathBuf,
    /// BF16 or F32 for the large weights. F32 source tensors (norms, `A_log`,
    /// `dt_bias`, `conv1d`) stay F32.
    pub dtype: DType,
    /// Target shard size in bytes.
    pub shard_bytes: u64,
    /// Only these decoder layers (`start..end`); `None` for all. A partial
    /// export is for validation (e.g. streaming one layer at a time).
    pub layers: Option<(u32, u32)>,
    /// Include the embedding, final norm and LM head.
    pub globals: bool,
    pub overwrite: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Elem {
    Copy,
    /// Zero-centered RMSNorm: the GGUF stores `w + 1`.
    MinusOne,
    /// `A_log`: the GGUF stores `-exp(A_log)`.
    LogNeg,
}

#[derive(Clone, Debug)]
struct Mapped<'a> {
    src: &'a TensorInfo,
    name: String,
    shape: Vec<u64>,
    dtype: DType,
    /// Source row for each output row (`None`: identity).
    rows: Option<Vec<usize>>,
    /// Source column for each output column (`None`: identity).
    cols: Option<Vec<usize>>,
    elem: Elem,
    rotated: bool,
    note: String,
}

/// Linear-attention head layout, from the reference config.
#[derive(Clone, Copy, Debug)]
struct Gdn {
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
}

impl Gdn {
    fn reorders(&self) -> bool {
        self.nk > 0 && self.nv > 0 && self.nk != self.nv
    }
}

/// For each HF (grouped-order) index along a V-head axis, the GGUF (tiled-order)
/// index it comes from. The converter's `_reorder_v_heads` maps grouped
/// `[nk, per_k, hd]` to tiled `[per_k, nk, hd]`; this is its inverse gather.
pub fn grouped_from_tiled(nk: usize, per_k: usize, hd: usize) -> Vec<usize> {
    (0..nk * per_k * hd)
        .map(|h| {
            let (head, d) = (h / hd, h % hd);
            let (k, j) = (head / per_k, head % per_k);
            (j * nk + k) * hd + d
        })
        .collect()
}

fn text_config(reference: &Path) -> Result<(Value, Value), SurgeryError> {
    let path = reference.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| SurgeryError::Incompatible(format!("{}: {e}", path.display())))?;
    let full: Value = serde_json::from_str(&text)
        .map_err(|e| SurgeryError::Incompatible(format!("{}: {e}", path.display())))?;
    let t = full
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full.clone());
    Ok((full, t))
}

fn usize_key(t: &Value, k: &str) -> Result<usize, SurgeryError> {
    t.get(k)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| SurgeryError::Incompatible(format!("reference config has no `{k}`")))
}

fn map_tensor<'a>(
    t: &'a TensorInfo,
    gdn: Gdn,
    grouped_out: bool,
    rotated: bool,
    dtype: DType,
) -> Result<Option<(Option<u32>, Mapped<'a>)>, SurgeryError> {
    let name = t.name.as_str();
    let out_dtype = if t.dtype == DType::F32 {
        DType::F32
    } else {
        dtype
    };
    let mut m = Mapped {
        src: t,
        name: String::new(),
        shape: t.shape.clone(),
        dtype: out_dtype,
        rows: None,
        cols: None,
        elem: Elem::Copy,
        rotated,
        note: String::new(),
    };
    let global = |m: &mut Mapped<'a>, hf: &str, elem: Elem| {
        m.name = hf.into();
        m.elem = elem;
    };
    match name {
        "token_embd.weight" => {
            global(&mut m, "model.embed_tokens.weight", Elem::Copy);
            return Ok(Some((None, m)));
        }
        "output_norm.weight" => {
            global(&mut m, "model.norm.weight", Elem::MinusOne);
            return Ok(Some((None, m)));
        }
        "output.weight" => {
            global(&mut m, "lm_head.weight", Elem::Copy);
            return Ok(Some((None, m)));
        }
        _ => {}
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return Err(SurgeryError::Unsupported(format!(
            "no HF name for tensor {name}"
        )));
    };
    let (idx, suffix) = rest
        .split_once('.')
        .ok_or_else(|| SurgeryError::Unsupported(format!("no HF name for tensor {name}")))?;
    let layer: u32 = idx
        .parse()
        .map_err(|_| SurgeryError::Unsupported(format!("no HF name for tensor {name}")))?;
    if suffix.starts_with("nextn.") {
        return Ok(None); // MTP tensors: not part of the trunk
    }
    let per_k = gdn.nv.checked_div(gdn.nk).unwrap_or(0);
    let v_rows = |hd: usize| {
        gdn.reorders()
            .then(|| grouped_from_tiled(gdn.nk, per_k, hd))
    };
    let (hf, elem): (&str, Elem) = match suffix {
        "attn_norm.weight" => ("input_layernorm.weight", Elem::MinusOne),
        "post_attention_norm.weight" => ("post_attention_layernorm.weight", Elem::MinusOne),
        "ffn_gate.weight" => ("mlp.gate_proj.weight", Elem::Copy),
        "ffn_up.weight" => ("mlp.up_proj.weight", Elem::Copy),
        "ffn_down.weight" => ("mlp.down_proj.weight", Elem::Copy),
        "attn_q.weight" => ("self_attn.q_proj.weight", Elem::Copy),
        "attn_k.weight" => ("self_attn.k_proj.weight", Elem::Copy),
        "attn_v.weight" => ("self_attn.v_proj.weight", Elem::Copy),
        "attn_output.weight" => ("self_attn.o_proj.weight", Elem::Copy),
        "attn_q_norm.weight" => ("self_attn.q_norm.weight", Elem::MinusOne),
        "attn_k_norm.weight" => ("self_attn.k_norm.weight", Elem::MinusOne),
        "ssm_norm.weight" => ("linear_attn.norm.weight", Elem::Copy),
        "attn_qkv.weight" => {
            // Rows: [q (nk·hk), k (nk·hk), v (nv·hv)]; only V is reordered.
            if let Some(v) = v_rows(gdn.hv) {
                let qk = 2 * gdn.nk * gdn.hk;
                m.rows = Some((0..qk).chain(v.into_iter().map(|r| r + qk)).collect());
                m.note = "V rows tiled → grouped".into();
            }
            ("linear_attn.in_proj_qkv.weight", Elem::Copy)
        }
        "attn_gate.weight" => {
            m.rows = v_rows(gdn.hv);
            ("linear_attn.in_proj_z.weight", Elem::Copy)
        }
        "ssm_beta.weight" => {
            m.rows = v_rows(1);
            ("linear_attn.in_proj_b.weight", Elem::Copy)
        }
        "ssm_alpha.weight" => {
            m.rows = v_rows(1);
            ("linear_attn.in_proj_a.weight", Elem::Copy)
        }
        "ssm_a" => {
            m.cols = v_rows(1);
            ("linear_attn.A_log", Elem::LogNeg)
        }
        "ssm_dt.bias" => {
            m.cols = v_rows(1);
            ("linear_attn.dt_bias", Elem::Copy)
        }
        "ssm_conv1d.weight" => {
            // [channels, kernel]: channels are [q, k, v] like in_proj_qkv.
            if let Some(v) = v_rows(gdn.hv) {
                let qk = 2 * gdn.nk * gdn.hk;
                m.rows = Some((0..qk).chain(v.into_iter().map(|r| r + qk)).collect());
                m.note = "V channels tiled → grouped".into();
            }
            if let [c, k] = t.shape[..] {
                m.shape = vec![c, 1, k];
            }
            ("linear_attn.conv1d.weight", Elem::Copy)
        }
        "ssm_out.weight" => {
            if !grouped_out {
                m.cols = v_rows(gdn.hv);
            } else {
                m.note = "columns already grouped (gdn_v_grouped)".into();
            }
            ("linear_attn.out_proj.weight", Elem::Copy)
        }
        other => {
            return Err(SurgeryError::Unsupported(format!(
                "no HF name for blk.{layer}.{other}"
            )))
        }
    };
    if m.rows
        .as_ref()
        .is_some_and(|r| r.len() as u64 != t.shape[0])
    {
        return Err(SurgeryError::Incompatible(format!(
            "{name}: {} rows, but the reference config implies {}",
            t.shape[0],
            m.rows.as_ref().map_or(0, Vec::len)
        )));
    }
    if m.cols
        .as_ref()
        .is_some_and(|c| c.len() as u64 != *t.shape.last().unwrap_or(&0))
    {
        return Err(SurgeryError::Incompatible(format!(
            "{name}: {} columns, but the reference config implies {}",
            t.shape.last().unwrap_or(&0),
            m.cols.as_ref().map_or(0, Vec::len)
        )));
    }
    m.name = format!("model.layers.{layer}.{hf}");
    m.elem = elem;
    Ok(Some((Some(layer), m)))
}

fn apply_elem(elem: Elem, x: &mut [f32], name: &str) -> Result<(), SurgeryError> {
    match elem {
        Elem::Copy => {}
        Elem::MinusOne => x.iter_mut().for_each(|v| *v -= 1.0),
        Elem::LogNeg => {
            for v in x.iter_mut() {
                if *v >= 0.0 {
                    return Err(SurgeryError::Incompatible(format!(
                        "{name}: expected -exp(A_log) < 0, found {v}"
                    )));
                }
                *v = (-*v).ln();
            }
        }
    }
    Ok(())
}

/// Decodes the source rows `rows` (consecutive) into `out`, in the primal basis.
fn decode_rows(
    model: &LoadedModel,
    ir: &ModelIr,
    m: &Mapped,
    first: usize,
    count: usize,
    out: &mut Vec<f32>,
) -> Result<(), SurgeryError> {
    let t = m.src;
    let width = *t.shape.last().unwrap_or(&1) as usize;
    let total_rows = (t.n_elements() as usize) / width.max(1);
    let bytes = model.tensor_bytes(t)?;
    let row_bytes = bytes.len() / total_rows.max(1);
    let chunk = &bytes[first * row_bytes..(first + count) * row_bytes];
    out.clear();
    dequantize(t.dtype, chunk, out).map_err(|source| SurgeryError::Dequant {
        name: t.name.clone(),
        source,
    })?;
    if m.rotated {
        let rot = ir
            .weight_rotation
            .as_ref()
            .expect("rotated implies a rotation");
        for row in out.chunks_exact_mut(width) {
            rot.to_primal(&t.name, row)
                .map_err(|e| SurgeryError::Unsupported(format!("{}: {e}", t.name)))?;
        }
    }
    Ok(())
}

fn write_mapped(
    model: &LoadedModel,
    ir: &ModelIr,
    m: &Mapped,
    w: &mut dyn Write,
) -> Result<(), SurgeryError> {
    let t = m.src;
    let width = *t.shape.last().unwrap_or(&1) as usize;
    let rows = (t.n_elements() as usize) / width.max(1);
    let per_chunk = (CHUNK_ELEMS / width.max(1)).max(1);
    let mut buf = Vec::new();
    let mut row = vec![0f32; width];
    let mut bytes_out = Vec::new();
    let mut emit = |vals: &mut [f32], w: &mut dyn Write| -> Result<(), SurgeryError> {
        for r in vals.chunks_exact_mut(width) {
            if let Some(cols) = &m.cols {
                row.iter_mut().zip(cols).for_each(|(o, &c)| *o = r[c]);
                r.copy_from_slice(&row);
            }
            apply_elem(m.elem, r, &m.name)?;
        }
        bytes_out.clear();
        match m.dtype {
            DType::F32 => bytes_out.extend(vals.iter().flat_map(|x| x.to_le_bytes())),
            _ => bytes_out.extend(vals.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes())),
        }
        w.write_all(&bytes_out)
            .map_err(|e| SurgeryError::Output(e.to_string()))
    };
    let mut i = 0;
    while i < rows {
        // The longest run of output rows whose sources are consecutive.
        let src = |i: usize| m.rows.as_ref().map_or(i, |r| r[i]);
        let start = src(i);
        let mut n = 1;
        while i + n < rows && n < per_chunk && src(i + n) == start + n {
            n += 1;
        }
        decode_rows(model, ir, m, start, n, &mut buf)?;
        emit(&mut buf, w)?;
        i += n;
    }
    Ok(())
}

/// Writes `out_dir/{config.json, model-XXXXX-of-YYYYY.safetensors,
/// model.safetensors.index.json, tokenizer files}`.
pub fn export_hf(
    model: &LoadedModel,
    ir: &ModelIr,
    out_dir: &Path,
    opts: &HfExportOptions,
) -> Result<SurgeryReport, SurgeryError> {
    if !matches!(opts.dtype, DType::Bf16 | DType::F32) {
        return Err(SurgeryError::Unsupported(format!(
            "export dtype {} (use BF16 or F32)",
            opts.dtype
        )));
    }
    let arch = ir.family.clone().unwrap_or_default();
    if arch != "qwen35" {
        return Err(SurgeryError::Unsupported(format!(
            "HF export is implemented for qwen35 GGUFs (Qwen3.5-family text models), not `{arch}`"
        )));
    }
    let (full_cfg, t) = text_config(&opts.reference)?;
    let gdn = Gdn {
        nk: usize_key(&t, "linear_num_key_heads")?,
        nv: usize_key(&t, "linear_num_value_heads")?,
        hk: usize_key(&t, "linear_key_head_dim")?,
        hv: usize_key(&t, "linear_value_head_dim")?,
    };
    let n_layers = usize_key(&t, "num_hidden_layers")?;
    if ir.layers.len() != n_layers {
        return Err(SurgeryError::Incompatible(format!(
            "the GGUF has {} layers, the reference config {n_layers}",
            ir.layers.len()
        )));
    }
    let hidden = usize_key(&t, "hidden_size")? as u64;
    if ir.hidden_size != Some(hidden) {
        return Err(SurgeryError::Incompatible(format!(
            "hidden size {:?} vs the reference's {hidden}",
            ir.hidden_size
        )));
    }
    let rotation = ir.weight_rotation.as_ref();
    let grouped_out = rotation.is_some_and(|r| r.gdn_v_grouped);

    if out_dir.exists()
        && std::fs::read_dir(out_dir).is_ok_and(|mut d| d.next().is_some())
        && !opts.overwrite
    {
        return Err(SurgeryError::Output(format!(
            "{} is not empty (pass overwrite)",
            out_dir.display()
        )));
    }
    if ir.raw.files.iter().any(|f| f.starts_with(out_dir)) {
        return Err(SurgeryError::Output(
            "the output directory contains the input".into(),
        ));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| SurgeryError::Output(e.to_string()))?;
    let mut written = Vec::new();
    let result = write_all(
        model,
        ir,
        out_dir,
        opts,
        gdn,
        grouped_out,
        n_layers,
        &full_cfg,
        &t,
        &mut written,
    );
    if result.is_err() {
        // Don't leave a checkpoint that looks complete but isn't.
        for f in &written {
            let _ = std::fs::remove_file(f);
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn write_all(
    model: &LoadedModel,
    ir: &ModelIr,
    out_dir: &Path,
    opts: &HfExportOptions,
    gdn: Gdn,
    grouped_out: bool,
    n_layers: usize,
    full_cfg: &Value,
    t: &Value,
    written: &mut Vec<PathBuf>,
) -> Result<SurgeryReport, SurgeryError> {
    let rotation = ir.weight_rotation.as_ref();
    let mut mapped = Vec::new();
    for t in &ir.raw.tensors {
        let rotated = rotation.is_some_and(|r| r.is_rotated(&t.name));
        let Some((layer, m)) = map_tensor(t, gdn, grouped_out, rotated, opts.dtype)? else {
            continue;
        };
        let keep = match (layer, opts.layers) {
            (None, _) => opts.globals,
            (Some(_), None) => true,
            (Some(l), Some((a, b))) => (a..b).contains(&l),
        };
        if keep {
            mapped.push(m);
        }
    }
    if mapped.is_empty() {
        return Err(SurgeryError::Incompatible(
            "nothing to export with these options".into(),
        ));
    }

    // Shards, in file order.
    let size =
        |m: &Mapped| m.shape.iter().product::<u64>() * if m.dtype == DType::F32 { 4 } else { 2 };
    let mut shards: Vec<Vec<usize>> = vec![Vec::new()];
    let mut acc = 0u64;
    for (i, m) in mapped.iter().enumerate() {
        if acc > 0 && acc + size(m) > opts.shard_bytes {
            shards.push(Vec::new());
            acc = 0;
        }
        shards.last_mut().expect("non-empty").push(i);
        acc += size(m);
    }
    let n = shards.len();
    let mut weight_map = serde_json::Map::new();
    let mut meta = BTreeMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    meta.insert("modelbuilder.export".to_string(), "hf-primal".to_string());
    meta.insert(
        "modelbuilder.source".to_string(),
        ir.raw.root.display().to_string(),
    );
    if let Some((a, b)) = opts.layers {
        meta.insert("modelbuilder.layers".to_string(), format!("{a}..{b}"));
    }
    let mut total = 0u64;
    for (si, idx) in shards.iter().enumerate() {
        let file = format!("model-{:05}-of-{n:05}.safetensors", si + 1);
        let headers: Vec<TensorHeader> = idx
            .iter()
            .map(|&i| TensorHeader {
                name: mapped[i].name.clone(),
                dtype: mapped[i].dtype,
                shape: mapped[i].shape.clone(),
            })
            .collect();
        let mut failure = None;
        written.push(out_dir.join(&file));
        let result = write_streaming(&out_dir.join(&file), &headers, &meta, |k, w| {
            write_mapped(model, ir, &mapped[idx[k]], w).map_err(|e| {
                let msg = e.to_string();
                failure = Some(e);
                std::io::Error::other(msg)
            })
        });
        if let Some(e) = failure {
            return Err(e);
        }
        result?;
        for &i in idx {
            weight_map.insert(mapped[i].name.clone(), file.clone().into());
            total += size(&mapped[i]);
        }
    }
    let index = json!({ "metadata": { "total_size": total }, "weight_map": weight_map });
    written.push(out_dir.join("model.safetensors.index.json"));
    write_json(&out_dir.join("model.safetensors.index.json"), &index)?;

    // A text-only config for Qwen3_5ForCausalLM, from the reference's text_config.
    let mut cfg = t.clone();
    if let Value::Object(o) = &mut cfg {
        o.insert("architectures".into(), json!(["Qwen3_5ForCausalLM"]));
        o.insert("model_type".into(), json!("qwen3_5_text"));
        o.insert("tie_word_embeddings".into(), json!(false));
        o.insert(
            "dtype".into(),
            json!(if opts.dtype == DType::F32 {
                "float32"
            } else {
                "bfloat16"
            }),
        );
        // The GGUF has no MTP block; the reference's belongs to the reference.
        o.insert("mtp_num_hidden_layers".into(), json!(0));
        for k in ["bos_token_id", "eos_token_id", "pad_token_id"] {
            if let Some(v) = full_cfg.get(k) {
                o.entry(k).or_insert(v.clone());
            }
        }
    }
    written.push(out_dir.join("config.json"));
    write_json(&out_dir.join("config.json"), &cfg)?;
    let mut copied = Vec::new();
    for f in TOKENIZER_FILES {
        let src = opts.reference.join(f);
        if src.is_file() {
            written.push(out_dir.join(f));
            std::fs::copy(&src, out_dir.join(f))
                .map_err(|e| SurgeryError::Output(format!("{f}: {e}")))?;
            copied.push(*f);
        }
    }

    let tensors = mapped
        .iter()
        .map(|m| WrittenTensor {
            name: m.name.clone(),
            dtype: m.dtype.to_string(),
            shape: m.shape.clone(),
            source: format!("gguf:{}", m.src.name),
            transform: [
                Some(format!("{} → {}", m.src.dtype, m.dtype)),
                m.rotated.then(|| "rotation undone".to_string()),
                match m.elem {
                    Elem::Copy => None,
                    Elem::MinusOne => Some("norm − 1".into()),
                    Elem::LogNeg => Some("ln(−a) → A_log".into()),
                },
                (!m.note.is_empty()).then(|| m.note.clone()),
                (m.rows.is_some() && m.note.is_empty()).then(|| "V rows tiled → grouped".into()),
                (m.cols.is_some() && m.note.is_empty()).then(|| "V entries tiled → grouped".into()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(", "),
        })
        .collect();
    let mut notes = vec![
        "Primal basis: the Hadamard fold is undone, so the weights are plain HF weights (dequantized ternary values).".into(),
        "config.json is the reference's text_config for Qwen3_5ForCausalLM; load with transformers' AutoModelForCausalLM.".into(),
    ];
    if let Some((a, b)) = opts.layers {
        notes.push(format!(
            "Partial export: layers {a}..{b} only (config.json still describes all {n_layers})."
        ));
    }
    if copied.is_empty() {
        notes.push("No tokenizer files found in the reference directory.".into());
    }
    Ok(SurgeryReport {
        output: out_dir.to_owned(),
        bytes: total,
        tensors,
        metadata: vec![
            format!("{n} shard(s), config.json, model.safetensors.index.json"),
            format!("copied: {}", copied.join(", ")),
        ],
        notes,
    })
}

fn write_json(path: &Path, v: &Value) -> Result<(), SurgeryError> {
    let text = serde_json::to_string_pretty(v).expect("JSON serializes") + "\n";
    std::fs::write(path, text).map_err(|e| SurgeryError::Output(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::grouped_from_tiled;

    /// The converter's `_reorder_v_heads` (grouped → tiled), as a gather.
    fn tiled_from_grouped(nk: usize, per_k: usize, hd: usize) -> Vec<usize> {
        (0..nk * per_k * hd)
            .map(|g| {
                let (head, d) = (g / hd, g % hd);
                let (j, k) = (head / nk, head % nk);
                (k * per_k + j) * hd + d
            })
            .collect()
    }

    #[test]
    fn v_head_reorder_inverts_the_converter() {
        for (nk, per_k, hd) in [(16, 3, 128), (2, 2, 32), (4, 1, 8), (3, 5, 1)] {
            let hf: Vec<usize> = (0..nk * per_k * hd).collect();
            let gguf: Vec<usize> = tiled_from_grouped(nk, per_k, hd)
                .iter()
                .map(|&i| hf[i])
                .collect();
            let back: Vec<usize> = grouped_from_tiled(nk, per_k, hd)
                .iter()
                .map(|&i| gguf[i])
                .collect();
            assert_eq!(back, hf, "nk {nk} per_k {per_k} hd {hd}");
        }
        // Spot check against the converter's description: grouped [G0_v0, G0_v1, G1_v0, G1_v1]
        // becomes tiled [G0_v0, G1_v0, G0_v1, G1_v1].
        assert_eq!(tiled_from_grouped(2, 2, 1), [0, 2, 1, 3]);
    }
}

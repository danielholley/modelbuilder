//! Export a GGUF as the Hugging Face checkpoint it was converted from, so the
//! trunk can run in PyTorch (frozen, for draft-head training, or trained, for
//! QAT and restructuring).
//!
//! Per-tensor names and transforms come from the architecture adapter
//! ([`crate::arch`]), which inverts what llama.cpp's converter does to that
//! architecture. Quantized weights are decoded, and a folded rotation (PrismML
//! Hadamard) is undone, so the export is in the primal basis. Tensors are
//! streamed a chunk of rows at a time into sharded safetensors.
//!
//! The architecture comes from a reference HF `config.json` (the model the
//! GGUF was converted from, or one with the same architecture), checked
//! against the GGUF's shapes. Tokenizer files are copied from the reference.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use mb_formats::dequant::dequantize;
use mb_formats::safetensors::{write_streaming, TensorHeader};
use mb_formats::LoadedModel;
use mb_ir::{DType, ModelIr, TensorInfo};
use serde_json::{json, Value};

use crate::arch::{self, Dims, Elem, Mapping};
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

#[derive(Clone, Debug)]
struct Mapped<'a> {
    src: &'a TensorInfo,
    map: Mapping,
    dtype: DType,
    rotated: bool,
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
            if let Some(cols) = &m.map.cols {
                row.iter_mut().zip(cols).for_each(|(o, &c)| *o = r[c]);
                r.copy_from_slice(&row);
            }
            for v in r.iter_mut() {
                *v = m
                    .map
                    .elem
                    .to_hf(*v)
                    .map_err(|e| SurgeryError::Incompatible(format!("{}: {e}", m.src.name)))?;
            }
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
        let src = |i: usize| m.map.rows.as_ref().map_or(i, |r| r[i]);
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

/// `modelbuilder.pruned_layers` (`start..end`), written by `surgery prune`.
fn pruned_layers(ir: &ModelIr) -> Option<(usize, usize)> {
    let mb_ir::Metadata::Gguf { kv, .. } = &ir.raw.metadata else {
        return None;
    };
    let v = kv.iter().find(|(k, _)| k == "modelbuilder.pruned_layers")?.1.as_str()?;
    let (a, b) = v.split_once("..")?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Removes layers `a..b` from a reference config: `num_hidden_layers`, and
/// every per-layer array (e.g. `layer_types`).
fn prune_config(cfg: &mut Value, a: usize, b: usize) {
    let Some(o) = cfg.as_object_mut() else {
        return;
    };
    let Some(n) = o.get("num_hidden_layers").and_then(Value::as_u64) else {
        return;
    };
    for v in o.values_mut() {
        if let Value::Array(items) = v {
            if items.len() == n as usize {
                let mut i = 0;
                items.retain(|_| {
                    let keep = !(a..b).contains(&i);
                    i += 1;
                    keep
                });
            }
        }
    }
    o.insert("num_hidden_layers".into(), json!(n as usize - (b - a)));
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
    let adapter = arch::adapter(ir.family.as_deref().unwrap_or("unknown"))?;
    let (full_cfg, mut t) = text_config(&opts.reference)?;
    let pruned = pruned_layers(ir);
    if let Some((a, b)) = pruned {
        prune_config(&mut t, a, b);
    }
    let dims = Dims::from_config(&t, adapter)?;
    if ir.layers.len() != dims.layers {
        return Err(SurgeryError::Incompatible(format!(
            "the GGUF has {} layers, the reference config {}",
            ir.layers.len(),
            dims.layers
        )));
    }
    if ir.hidden_size != Some(dims.hidden as u64) {
        return Err(SurgeryError::Incompatible(format!(
            "hidden size {:?} vs the reference's {}",
            ir.hidden_size, dims.hidden
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
        adapter,
        &dims,
        grouped_out,
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
    adapter: &arch::Adapter,
    dims: &Dims,
    grouped_out: bool,
    full_cfg: &Value,
    t: &Value,
    written: &mut Vec<PathBuf>,
) -> Result<SurgeryReport, SurgeryError> {
    let rotation = ir.weight_rotation.as_ref();
    let mut mapped = Vec::new();
    for t in &ir.raw.tensors {
        let rotated = rotation.is_some_and(|r| r.is_rotated(&t.name));
        let Some(map) = arch::to_hf(adapter, dims, &t.name, &t.shape, grouped_out)? else {
            continue;
        };
        // Small F32 tensors (norms, gates, biases) stay F32.
        let dtype = if t.dtype == DType::F32 {
            DType::F32
        } else {
            opts.dtype
        };
        let m = Mapped {
            src: t,
            map,
            dtype,
            rotated,
        };
        let keep = match (m.map.layer, opts.layers) {
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
    let size = |m: &Mapped| {
        m.map.hf_shape.iter().product::<u64>() * if m.dtype == DType::F32 { 4 } else { 2 }
    };
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
                name: mapped[i].map.hf_name.clone(),
                dtype: mapped[i].dtype,
                shape: mapped[i].map.hf_shape.clone(),
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
            weight_map.insert(mapped[i].map.hf_name.clone(), file.clone().into());
            total += size(&mapped[i]);
        }
    }
    let index = json!({ "metadata": { "total_size": total }, "weight_map": weight_map });
    written.push(out_dir.join("model.safetensors.index.json"));
    write_json(&out_dir.join("model.safetensors.index.json"), &index)?;

    // How each tensor is stored in the source, so training can fake-quantize
    // to exactly that format (and `surgery replace` can re-encode losslessly).
    let mut qt = serde_json::Map::new();
    for m in &mapped {
        qt.insert(
            m.map.hf_name.clone(),
            json!({
                "gguf": m.src.name,
                "dtype": m.src.dtype.to_string(),
                "rotated": m.rotated,
                "rows_reordered": m.map.rows.is_some(),
                "cols_reordered": m.map.cols.is_some(),
            }),
        );
    }
    let rot_json = rotation.map(|r| {
        let signs: serde_json::Map<String, Value> = r
            .sign_widths
            .iter()
            .filter_map(|w| r.signs(*w).map(|s| (w.to_string(), json!(s))))
            .collect();
        json!({ "scheme": r.scheme, "block_size": r.block_size, "sign_mode": r.sign_mode, "signs": signs })
    });
    let quant = json!({
        "format": "modelbuilder.source-quantization",
        "version": 1,
        "source": ir.raw.root.display().to_string(),
        "tensors": qt,
        "rotation": rot_json,
    });
    written.push(out_dir.join("modelbuilder_quantization.json"));
    write_json(&out_dir.join("modelbuilder_quantization.json"), &quant)?;

    // A text-only config for the adapter's causal-LM class, from the reference's (text) config.
    let mut cfg = t.clone();
    let tied = !mapped.iter().any(|m| m.map.hf_name == "lm_head.weight") && opts.globals;
    if let Value::Object(o) = &mut cfg {
        o.insert("architectures".into(), json!([adapter.hf_architecture]));
        o.insert("model_type".into(), json!(adapter.hf_model_type));
        o.insert("tie_word_embeddings".into(), json!(tied));
        o.insert(
            "dtype".into(),
            json!(if opts.dtype == DType::F32 {
                "float32"
            } else {
                "bfloat16"
            }),
        );
        // An MTP block isn't exported; the reference's belongs to the reference.
        if o.contains_key("mtp_num_hidden_layers") {
            o.insert("mtp_num_hidden_layers".into(), json!(0));
        }
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
            name: m.map.hf_name.clone(),
            dtype: m.dtype.to_string(),
            shape: m.map.hf_shape.clone(),
            source: format!("gguf:{}", m.src.name),
            transform: [
                Some(format!("{} → {}", m.src.dtype, m.dtype)),
                m.rotated.then(|| "rotation undone".to_string()),
                match m.map.elem {
                    Elem::Copy => None,
                    Elem::MinusOne => Some("norm − 1".into()),
                    Elem::LogNeg => Some("ln(−a) → A_log".into()),
                },
                (!m.map.note.is_empty()).then(|| m.map.note.to_string()),
                m.map.rows.is_some().then(|| "rows reordered".into()),
                m.map.cols.is_some().then(|| "columns reordered".into()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(", "),
        })
        .collect();
    let mut notes = vec![
        "Primal basis: quantized weights are decoded and any folded rotation is undone.".into(),
        format!("config.json is the reference's (text) config for {}; load it with transformers' AutoModelForCausalLM.", adapter.hf_architecture),
    ];
    if let Some((a, b)) = opts.layers {
        notes.push(format!(
            "Partial export: layers {a}..{b} only (config.json still describes all {}).",
            dims.layers
        ));
    }
    if let Some((a, b)) = pruned_layers(ir) {
        notes.push(format!(
            "The GGUF was pruned (layers {a}..{b} removed); config.json drops them too."
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

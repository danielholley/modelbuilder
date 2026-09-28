//! Write trained tensors back into a GGUF: the inverse of [`crate::hf_export`].
//!
//! Given the original GGUF and an HF-layout directory with some updated
//! tensors (primal basis, HF names; e.g. the output of a QAT stage), writes a
//! new GGUF where each updated tensor is:
//!
//! 1. mapped back through the architecture adapter (names, row/column
//!    reorders, element transforms, shape);
//! 2. folded back into the checkpoint's rotated basis, if it was rotated
//!    (`WeightRotation::from_primal`);
//! 3. re-encoded to the **original tensor's type** with ggml's reference
//!    quantizer. Nothing is silently re-saved at a different precision. A
//!    type without an encoder is an error.
//!
//! All other tensors and all metadata are copied byte for byte. For each
//! replaced tensor the report gives the re-quantization error: close to zero
//! when the training fake-quantized to the same format, larger when it didn't.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use mb_formats::dequant::dequantize;
use mb_formats::quant::{quantize, supports};
use mb_formats::{gguf, LoadedModel, TensorToWrite};
use mb_ir::{MetaValue, Metadata, ModelIr, SourceFormat};
use serde_json::Value;

use crate::arch::{self, Dims, Mapping};
use crate::{SurgeryError, SurgeryReport, WrittenTensor};

#[derive(Clone, Debug)]
pub struct ReplaceOptions {
    /// The HF config describing the architecture (e.g. the updates
    /// directory's own `config.json`, or the reference model's).
    pub config: PathBuf,
    pub overwrite: bool,
    /// Refuse to write if any tensor's relative re-quantization error
    /// (RMS error / RMS value) exceeds this.
    pub max_rel_error: Option<f64>,
}

/// Replaces tensors of `target` (a GGUF) with those found in `updates`.
pub fn replace_tensors(
    target: &LoadedModel,
    target_ir: &ModelIr,
    updates: &LoadedModel,
    out: &Path,
    opts: &ReplaceOptions,
) -> Result<SurgeryReport, SurgeryError> {
    if target_ir.raw.format != SourceFormat::Gguf {
        return Err(SurgeryError::Unsupported(
            "the target must be a GGUF file".into(),
        ));
    }
    let adapter = arch::adapter(target_ir.family.as_deref().unwrap_or("unknown"))?;
    let cfg: Value = std::fs::read_to_string(&opts.config)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .ok_or_else(|| {
            SurgeryError::Incompatible(format!("cannot read {}", opts.config.display()))
        })?;
    let dims = Dims::from_config(&cfg, adapter)?;
    let rotation = target_ir.weight_rotation.as_ref();
    let grouped_out = rotation.is_some_and(|r| r.gdn_v_grouped);
    if out.exists() {
        let same = |p: &Path| std::fs::canonicalize(p).ok() == std::fs::canonicalize(out).ok();
        if target_ir
            .raw
            .files
            .iter()
            .chain(&updates.raw.files)
            .any(|p| same(p))
        {
            return Err(SurgeryError::Output(
                "the output path is one of the input files".into(),
            ));
        }
        if !opts.overwrite {
            return Err(SurgeryError::Output(format!(
                "{} exists (pass overwrite)",
                out.display()
            )));
        }
    }

    let mut used = std::collections::BTreeSet::new();
    let mut tensors = Vec::new();
    let mut written = Vec::new();
    for t in &target_ir.raw.tensors {
        let bytes = target.tensor_bytes(t)?;
        let mapping = arch::to_hf(adapter, &dims, &t.name, &t.shape, grouped_out)?;
        let update = mapping
            .as_ref()
            .and_then(|m| updates.tensor(&m.hf_name).map(|u| (m, u)));
        let Some((m, u)) = update else {
            tensors.push(TensorToWrite {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                data: Cow::Borrowed(bytes),
            });
            continue;
        };
        used.insert(m.hf_name.clone());
        if u.shape != m.hf_shape {
            return Err(SurgeryError::Incompatible(format!(
                "{}: shape {:?}, but {} needs {:?}",
                m.hf_name, u.shape, t.name, m.hf_shape
            )));
        }
        if !supports(t.dtype) {
            return Err(SurgeryError::Unsupported(format!(
                "{} is {}, which has no encoder; re-encoding it at another precision would change the checkpoint",
                t.name, t.dtype
            )));
        }
        let mut hf = Vec::new();
        dequantize(u.dtype, updates.tensor_bytes(u)?, &mut hf).map_err(|source| {
            SurgeryError::Dequant {
                name: u.name.clone(),
                source,
            }
        })?;
        let mut vals = to_gguf_order(m, &hf, &t.shape);
        let rotated = rotation.is_some_and(|r| r.is_rotated(&t.name));
        let width = *t.shape.last().unwrap_or(&1) as usize;
        if rotated {
            let r = rotation.expect("rotated implies a rotation");
            for row in vals.chunks_exact_mut(width) {
                r.from_primal(&t.name, row)
                    .map_err(|e| SurgeryError::Unsupported(format!("{}: {e}", t.name)))?;
            }
        }
        let mut data = Vec::with_capacity(bytes.len());
        quantize(t.dtype, &vals, &mut data)
            .map_err(|e| SurgeryError::Unsupported(format!("{}: {e}", t.name)))?;
        // Re-quantization error, measured in the stored basis (orthogonal, so it equals the primal error).
        let mut back = Vec::with_capacity(vals.len());
        dequantize(t.dtype, &data, &mut back).map_err(|source| SurgeryError::Dequant {
            name: t.name.clone(),
            source,
        })?;
        let (mut err, mut norm) = (0f64, 0f64);
        for (a, b) in vals.iter().zip(&back) {
            err += f64::from(a - b).powi(2);
            norm += f64::from(*a).powi(2);
        }
        let rel = if norm > 0.0 { (err / norm).sqrt() } else { 0.0 };
        if let Some(max) = opts.max_rel_error {
            if rel > max {
                return Err(SurgeryError::Incompatible(format!(
                    "{}: re-quantizing to {} loses {:.2}% (RMS), above the limit of {:.2}%. Train with fake-quantization to {} first",
                    t.name,
                    t.dtype,
                    100.0 * rel,
                    100.0 * max,
                    t.dtype
                )));
            }
        }
        written.push(WrittenTensor {
            name: t.name.clone(),
            dtype: t.dtype.to_string(),
            shape: t.shape.clone(),
            source: format!("updates:{}", m.hf_name),
            transform: format!(
                "{} → {}{}, re-quantization error {:.3}% RMS",
                u.dtype,
                t.dtype,
                if rotated {
                    " (rotation re-applied)"
                } else {
                    ""
                },
                100.0 * rel
            ),
        });
        tensors.push(TensorToWrite {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: Cow::Owned(data),
        });
    }
    let unused: Vec<&str> = updates
        .raw
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| !used.contains(*n))
        .collect();
    if written.is_empty() {
        return Err(SurgeryError::Incompatible(
            "no tensor in the updates matches a tensor of the target".into(),
        ));
    }
    let Metadata::Gguf { kv, .. } = &target_ir.raw.metadata else {
        unreachable!("GGUF targets carry GGUF metadata")
    };
    let mut kv = kv.clone();
    kv.push((
        "modelbuilder.replaced_tensors".into(),
        MetaValue::U32(written.len() as u32),
    ));
    gguf::write(out, &kv, &tensors)?;
    let bytes = std::fs::metadata(out)
        .map_err(|e| SurgeryError::Output(e.to_string()))?
        .len();
    let mut notes = vec![format!(
        "{} tensor(s) replaced and re-encoded to their original types; all other tensors copied byte for byte.",
        written.len()
    )];
    if !unused.is_empty() {
        notes.push(format!(
            "Not in the target (ignored): {}",
            unused.join(", ")
        ));
    }
    Ok(SurgeryReport {
        output: out.to_owned(),
        bytes,
        tensors: written,
        metadata: vec!["modelbuilder.replaced_tensors (added)".into()],
        notes,
    })
}

/// HF values (row-major, HF shape) to GGUF order and shape, undoing the
/// mapping's reorders and element transform.
fn to_gguf_order(m: &Mapping, hf: &[f32], gguf_shape: &[u64]) -> Vec<f32> {
    let width = *gguf_shape.last().unwrap_or(&1) as usize;
    let rows = hf.len() / width.max(1);
    // mapping.rows[hf_row] = gguf_row, so gguf_row takes hf_row = inverse[gguf_row].
    let row_src = m.rows.as_ref().map(|r| Mapping::inverse(r));
    let col_src = m.cols.as_ref().map(|c| Mapping::inverse(c));
    let mut out = Vec::with_capacity(hf.len());
    for g in 0..rows {
        let r = row_src.as_ref().map_or(g, |p| p[g]);
        let src = &hf[r * width..(r + 1) * width];
        match &col_src {
            Some(c) => out.extend(c.iter().map(|&j| m.elem.to_gguf(src[j]))),
            None => out.extend(src.iter().map(|&v| m.elem.to_gguf(v))),
        }
    }
    out
}

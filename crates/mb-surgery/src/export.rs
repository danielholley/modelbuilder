//! Export selected tensors, decoded and in the primal (un-rotated) basis, to a
//! safetensors file. Used to hand a low-bit or rotated checkpoint's frozen
//! pieces (token embedding, output head) to the PyTorch training side.
//!
//! Tensors are decoded a chunk of rows at a time and written as they go,
//! so memory stays bounded by the chunk size, not the tensor size.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use mb_formats::dequant::dequantize;
use mb_formats::safetensors::{write_streaming, TensorHeader};
use mb_formats::LoadedModel;
use mb_ir::{DType, ModelIr};

use crate::{f32_to_bf16, SurgeryError, SurgeryReport, WrittenTensor};

/// Rows are decoded in chunks of about this many elements.
const CHUNK_ELEMS: usize = 1 << 22;

#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Exact tensor names to export.
    pub names: Vec<String>,
    /// Output dtype: BF16 or F32.
    pub dtype: DType,
    pub overwrite: bool,
}

pub fn export_primal(
    model: &LoadedModel,
    ir: &ModelIr,
    out: &Path,
    opts: &ExportOptions,
) -> Result<SurgeryReport, SurgeryError> {
    if !matches!(opts.dtype, DType::Bf16 | DType::F32) {
        return Err(SurgeryError::Unsupported(format!(
            "export dtype {} (use BF16 or F32)",
            opts.dtype
        )));
    }
    if out.exists() {
        let same = |p: &Path| std::fs::canonicalize(p).ok() == std::fs::canonicalize(out).ok();
        if ir.raw.files.iter().any(|p| same(p)) {
            return Err(SurgeryError::Output(
                "the output path is one of the input files".into(),
            ));
        }
        if !opts.overwrite {
            return Err(SurgeryError::Output(format!(
                "{} exists (pass overwrite to replace it)",
                out.display()
            )));
        }
    }
    let tensors = opts
        .names
        .iter()
        .map(|n| {
            model
                .tensor(n)
                .ok_or_else(|| SurgeryError::Incompatible(format!("no tensor named {n}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rotation = ir.weight_rotation.as_ref();

    let headers: Vec<TensorHeader> = tensors
        .iter()
        .map(|t| TensorHeader {
            name: t.name.clone(),
            dtype: opts.dtype,
            shape: t.shape.clone(),
        })
        .collect();
    let mut meta = BTreeMap::new();
    meta.insert("modelbuilder.export".to_string(), "primal".to_string());
    meta.insert(
        "modelbuilder.source".to_string(),
        ir.raw.root.display().to_string(),
    );

    let mut failure: Option<SurgeryError> = None;
    let result = write_streaming(out, &headers, &meta, |i, w: &mut dyn Write| {
        let t = tensors[i];
        let width = t.shape.last().copied().unwrap_or(1).max(1) as usize;
        let rows = (t.n_elements() as usize) / width;
        let bytes = model.tensor_bytes(t).map_err(std::io::Error::other)?;
        let row_bytes = bytes.len() / rows.max(1);
        let rows_per_chunk = (CHUNK_ELEMS / width).max(1);
        let rotated = rotation.filter(|r| r.is_rotated(&t.name));
        let mut buf = Vec::with_capacity(rows_per_chunk * width);
        let mut out_bytes = Vec::with_capacity(rows_per_chunk * width * 4);
        for chunk in bytes.chunks(rows_per_chunk * row_bytes) {
            buf.clear();
            if let Err(source) = dequantize(t.dtype, chunk, &mut buf) {
                failure = Some(SurgeryError::Dequant {
                    name: t.name.clone(),
                    source,
                });
                return Err(std::io::Error::other("decode failed"));
            }
            if let Some(r) = rotated {
                for row in buf.chunks_exact_mut(width) {
                    if let Err(e) = r.to_primal(&t.name, row) {
                        failure = Some(SurgeryError::Unsupported(format!("{}: {e}", t.name)));
                        return Err(std::io::Error::other("rotation failed"));
                    }
                }
            }
            out_bytes.clear();
            match opts.dtype {
                DType::Bf16 => {
                    out_bytes.extend(buf.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()))
                }
                _ => out_bytes.extend(buf.iter().flat_map(|x| x.to_le_bytes())),
            }
            w.write_all(&out_bytes)?;
        }
        Ok(())
    });
    if let Some(f) = failure {
        let _ = std::fs::remove_file(out);
        return Err(f);
    }
    result?;

    let written = tensors
        .iter()
        .map(|t| WrittenTensor {
            name: t.name.clone(),
            dtype: opts.dtype.to_string(),
            shape: t.shape.clone(),
            source: format!("model:{}", t.name),
            transform: match rotation.filter(|r| r.is_rotated(&t.name)) {
                Some(r) => format!(
                    "{} → {} (rotation undone: {})",
                    t.dtype, opts.dtype, r.scheme
                ),
                None => format!("{} → {}", t.dtype, opts.dtype),
            },
        })
        .collect();
    Ok(SurgeryReport {
        output: out.to_owned(),
        bytes: std::fs::metadata(out).map_err(|e| SurgeryError::Output(e.to_string()))?.len(),
        tensors: written,
        metadata: vec!["modelbuilder.export = primal".into()],
        notes: vec!["Rotated tensors are written in the primal basis (w = s ⊙ H·w_stored per row); row order and names are unchanged.".into()],
    })
}

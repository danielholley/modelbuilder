//! Port an MTP head from a Hugging Face reference model into a GGUF sidecar
//! for a `qwen35` target (the Qwen3.8-27B → Ternary-Bonsai-2-27B case).
//!
//! Layout and conversion follow the PrismML-Eng/llama.cpp fork:
//! - `conversion/qwen.py` (`_QwenMtpMixin`): `mtp.layers.0.*` becomes
//!   `blk.{n_layer}.*`; `mtp.fc`, `mtp.pre_fc_norm_embedding`,
//!   `mtp.pre_fc_norm_hidden` and `mtp.norm` become `blk.{n_layer}.nextn.{eh_proj,
//!   enorm, hnorm, shared_head_norm}`; `block_count` grows by one and
//!   `{arch}.nextn_predict_layers` is written.
//! - `Qwen3NextModel.modify_tensors`: every `*norm.weight` (except
//!   `linear_attn.norm`) is stored with 1 added (zero-centered RMSNorm). This
//!   was also checked against Bonsai 2's own F32 norms, which match Qwen3.8's
//!   plus one (cosine 1.0000).
//! - `src/models/qwen35.cpp`: a file with nextn layers but no `blk.0` trunk
//!   loads as an MTP-only model (`mtp_only`), with its own token embedding,
//!   output norm and output head. `common/speculative.cpp` runs it as the draft
//!   for `--spec-type draft-mtp`, fed with the target's hidden states.
//!
//! The ported head keeps the reference's BF16 weights, unrotated (block
//! weights not listed in `prism.hadamard.weight_names` use a plain matmul).
//! It was trained against the reference trunk, so it is an initialization for
//! the `mtp-align` training stage, not a finished drafter.

use std::borrow::Cow;
use std::path::Path;

use mb_formats::dequant::dequantize;
use mb_formats::{gguf, LoadedModel, TensorToWrite};
use mb_ir::{DType, MetaType, MetaValue, Metadata, Mixer, ModelIr, SourceFormat, TensorInfo};

use crate::{f32_to_bf16, SurgeryError, SurgeryReport, WrittenTensor};

/// Target architectures whose MTP layout has been checked against the loader.
const SUPPORTED_ARCHS: &[&str] = &["qwen35"];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Transform {
    /// Weight matrix: copied if BF16, converted to BF16 otherwise.
    Matrix,
    /// RMSNorm weight: +1, stored as F32.
    NormPlusOne,
}

/// `(reference suffix after "mtp.", GGUF suffix after "blk.{n}.", transform)`.
const TENSOR_MAP: &[(&str, &str, Transform)] = &[
    ("fc.weight", "nextn.eh_proj.weight", Transform::Matrix),
    (
        "pre_fc_norm_embedding.weight",
        "nextn.enorm.weight",
        Transform::NormPlusOne,
    ),
    (
        "pre_fc_norm_hidden.weight",
        "nextn.hnorm.weight",
        Transform::NormPlusOne,
    ),
    (
        "norm.weight",
        "nextn.shared_head_norm.weight",
        Transform::NormPlusOne,
    ),
    (
        "layers.0.input_layernorm.weight",
        "attn_norm.weight",
        Transform::NormPlusOne,
    ),
    (
        "layers.0.post_attention_layernorm.weight",
        "post_attention_norm.weight",
        Transform::NormPlusOne,
    ),
    (
        "layers.0.self_attn.q_proj.weight",
        "attn_q.weight",
        Transform::Matrix,
    ),
    (
        "layers.0.self_attn.k_proj.weight",
        "attn_k.weight",
        Transform::Matrix,
    ),
    (
        "layers.0.self_attn.v_proj.weight",
        "attn_v.weight",
        Transform::Matrix,
    ),
    (
        "layers.0.self_attn.o_proj.weight",
        "attn_output.weight",
        Transform::Matrix,
    ),
    (
        "layers.0.self_attn.q_norm.weight",
        "attn_q_norm.weight",
        Transform::NormPlusOne,
    ),
    (
        "layers.0.self_attn.k_norm.weight",
        "attn_k_norm.weight",
        Transform::NormPlusOne,
    ),
    (
        "layers.0.mlp.gate_proj.weight",
        "ffn_gate.weight",
        Transform::Matrix,
    ),
    (
        "layers.0.mlp.up_proj.weight",
        "ffn_up.weight",
        Transform::Matrix,
    ),
    (
        "layers.0.mlp.down_proj.weight",
        "ffn_down.weight",
        Transform::Matrix,
    ),
];

/// Tensors the sidecar copies verbatim from the target (the MTP-only loader
/// needs its own embedding, final norm and output head).
const TARGET_TENSORS: &[&str] = &["token_embd.weight", "output_norm.weight", "output.weight"];

#[derive(Clone, Debug, Default)]
pub struct MtpSidecarOptions {
    /// Replace the output file if it exists.
    pub overwrite: bool,
    /// Recorded in the sidecar's provenance metadata.
    pub reference_label: Option<String>,
    /// The head was already trained against this target (`modelbuilder job
    /// mtp-align`), not only against the reference trunk.
    pub aligned_to_target: bool,
}

fn reference_mtp_name(reference: &ModelIr, suffix: &str) -> Option<String> {
    ["mtp.", "model.mtp."]
        .iter()
        .map(|p| format!("{p}{suffix}"))
        .find(|n| reference.raw.tensors.iter().any(|t| &t.name == n))
}

fn kv_index(kv: &[(String, MetaValue)], key: &str) -> Option<usize> {
    kv.iter().position(|(k, _)| k == key)
}

/// Writes an MTP-only GGUF sidecar for `target`, porting the MTP head from `reference`.
pub fn port_mtp_sidecar(
    target: &LoadedModel,
    target_ir: &ModelIr,
    reference: &LoadedModel,
    reference_ir: &ModelIr,
    out: &Path,
    opts: &MtpSidecarOptions,
) -> Result<SurgeryReport, SurgeryError> {
    // --- Target checks.
    if target_ir.raw.format != SourceFormat::Gguf {
        return Err(SurgeryError::Unsupported(
            "the target must be a GGUF file".into(),
        ));
    }
    let arch = target_ir.family.clone().unwrap_or_default();
    if !SUPPORTED_ARCHS.contains(&arch.as_str()) {
        return Err(SurgeryError::Unsupported(format!(
            "target architecture `{arch}`; the MTP layout is verified for {SUPPORTED_ARCHS:?}"
        )));
    }
    if target_ir.mtp.is_some() {
        return Err(SurgeryError::Incompatible(
            "the target already has MTP tensors".into(),
        ));
    }
    let Metadata::Gguf { kv, .. } = &target_ir.raw.metadata else {
        unreachable!("GGUF targets carry GGUF metadata")
    };
    let block_key = format!("{arch}.block_count");
    let nextn_key = format!("{arch}.nextn_predict_layers");
    if kv
        .iter()
        .any(|(k, v)| k == &nextn_key && v.as_u64().unwrap_or(0) > 0)
    {
        return Err(SurgeryError::Incompatible(format!(
            "the target already declares {nextn_key}"
        )));
    }
    let n_layer = target_ir.layers.len();
    // The MTP block mirrors a full-attention trunk block: its tensors must have the same shapes.
    let template_layer = target_ir
        .layers
        .iter()
        .find(|l| matches!(l.mixer, Mixer::Attention(_)))
        .map(|l| l.index)
        .ok_or_else(|| {
            SurgeryError::Incompatible("the target has no full-attention layer".into())
        })?;
    let target_tensor = |name: &str| target.tensor(name);
    let hidden = target_ir
        .hidden_size
        .ok_or_else(|| SurgeryError::Incompatible("target hidden size unknown".into()))?;

    // --- Reference checks.
    if reference_ir.mtp.is_none() {
        return Err(SurgeryError::Incompatible(
            "the reference model has no MTP head".into(),
        ));
    }
    let depth = reference_ir.mtp.as_ref().map_or(0, |m| m.num_modules);
    if depth != 1 {
        return Err(SurgeryError::Unsupported(format!(
            "reference MTP depth {depth}; only depth 1 is supported"
        )));
    }
    let mapped: Vec<String> = TENSOR_MAP
        .iter()
        .filter_map(|(s, _, _)| reference_mtp_name(reference_ir, s))
        .collect();
    let unknown: Vec<&str> = reference_ir
        .raw
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| {
            (n.starts_with("mtp.") || n.starts_with("model.mtp.")) && !mapped.iter().any(|m| m == n)
        })
        .collect();
    if !unknown.is_empty() {
        return Err(SurgeryError::Unsupported(format!(
            "reference MTP tensors without a known mapping: {unknown:?}"
        )));
    }

    // --- Output checks: never touch an input.
    if out.exists() {
        let same = |p: &Path| std::fs::canonicalize(p).ok() == std::fs::canonicalize(out).ok();
        if target_ir
            .raw
            .files
            .iter()
            .chain(&reference_ir.raw.files)
            .any(|p| same(p))
        {
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

    let mut tensors: Vec<TensorToWrite<'_>> = Vec::new();
    let mut written = Vec::new();

    // Target tensors, copied byte for byte.
    for &name in TARGET_TENSORS {
        let Some(t) = target_tensor(name) else {
            if name == "output.weight" {
                continue; // tied embeddings: the loader reuses token_embd
            }
            return Err(SurgeryError::Incompatible(format!(
                "the target has no {name}"
            )));
        };
        tensors.push(TensorToWrite {
            name: name.into(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: Cow::Borrowed(target.tensor_bytes(t)?),
        });
        written.push(WrittenTensor {
            name: name.into(),
            dtype: t.dtype.to_string(),
            shape: t.shape.clone(),
            source: format!("target:{name}"),
            transform: "copied".into(),
        });
    }

    // Reference MTP tensors, mapped and transformed.
    for &(suffix, gguf_suffix, transform) in TENSOR_MAP {
        let src_name = reference_mtp_name(reference_ir, suffix).ok_or_else(|| {
            SurgeryError::Incompatible(format!("the reference has no mtp.{suffix}"))
        })?;
        let src: &TensorInfo = reference.tensor(&src_name).expect("found above");
        let dst_name = format!("blk.{n_layer}.{gguf_suffix}");

        // Shape checks: block tensors against the target's full-attention layer,
        // nextn tensors against the hidden size.
        let expected: Vec<u64> = if let Some(rest) = gguf_suffix.strip_prefix("nextn.") {
            if rest.starts_with("eh_proj") {
                vec![hidden, 2 * hidden]
            } else {
                vec![hidden]
            }
        } else {
            target_tensor(&format!("blk.{template_layer}.{gguf_suffix}"))
                .map(|t| t.shape.clone())
                .ok_or_else(|| {
                    SurgeryError::Incompatible(format!(
                        "the target has no blk.{template_layer}.{gguf_suffix} to match against"
                    ))
                })?
        };
        if src.shape != expected {
            return Err(SurgeryError::Incompatible(format!(
                "{src_name} has shape {:?}, the target needs {expected:?}",
                src.shape
            )));
        }

        let bytes = reference.tensor_bytes(src)?;
        let (dtype, data, how) = match (transform, src.dtype) {
            (Transform::Matrix, DType::Bf16) => (
                DType::Bf16,
                Cow::Borrowed(bytes),
                "copied (BF16)".to_string(),
            ),
            (Transform::Matrix, other) => {
                let mut v = Vec::new();
                dequantize(other, bytes, &mut v).map_err(|source| SurgeryError::Dequant {
                    name: src_name.clone(),
                    source,
                })?;
                let b: Vec<u8> = v
                    .iter()
                    .flat_map(|x| f32_to_bf16(*x).to_le_bytes())
                    .collect();
                (DType::Bf16, Cow::Owned(b), format!("{other} → BF16"))
            }
            (Transform::NormPlusOne, other) => {
                let mut v = Vec::new();
                dequantize(other, bytes, &mut v).map_err(|source| SurgeryError::Dequant {
                    name: src_name.clone(),
                    source,
                })?;
                let b: Vec<u8> = v.iter().flat_map(|x| (x + 1.0).to_le_bytes()).collect();
                (
                    DType::F32,
                    Cow::Owned(b),
                    format!("{other} + 1 → F32 (zero-centered RMSNorm)"),
                )
            }
        };
        written.push(WrittenTensor {
            name: dst_name.clone(),
            dtype: dtype.to_string(),
            shape: src.shape.clone(),
            source: format!("reference:{src_name}"),
            transform: how,
        });
        tensors.push(TensorToWrite {
            name: dst_name,
            dtype,
            shape: src.shape.clone(),
            data,
        });
    }

    // --- Metadata.
    let mut kv = kv.clone();
    let mut changes = Vec::new();
    let bi = kv_index(&kv, &block_key)
        .ok_or_else(|| SurgeryError::Incompatible(format!("the target has no {block_key}")))?;
    let new_count = n_layer as u64 + 1;
    kv[bi].1 = match kv[bi].1 {
        MetaValue::U32(_) => MetaValue::U32(new_count as u32),
        MetaValue::U64(_) => MetaValue::U64(new_count),
        ref other => {
            return Err(SurgeryError::Unsupported(format!(
                "{block_key} has type {:?}",
                other.meta_type()
            )))
        }
    };
    changes.push(format!("{block_key}: {n_layer} → {new_count}"));
    kv.insert(bi + 1, (nextn_key.clone(), MetaValue::U32(1)));
    changes.push(format!("{nextn_key} = 1 (added)"));

    // Rotation lists may only name tensors present in this file.
    let present: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    for key in [
        "prism.hadamard.weight_names",
        "prism.hadamard.inverse_weight_names",
    ] {
        if let Some(i) = kv_index(&kv, key) {
            let MetaValue::Array { values, .. } = &kv[i].1 else {
                continue;
            };
            let kept: Vec<MetaValue> = values
                .iter()
                .filter(|v| v.as_str().is_some_and(|s| present.contains(&s)))
                .cloned()
                .collect();
            changes.push(format!(
                "{key}: {} → {} (tensors in this file)",
                values.len(),
                kept.len()
            ));
            if kept.is_empty() && key.ends_with(".weight_names") {
                return Err(SurgeryError::Unsupported(
                    "no rotated tensor would remain in the sidecar, and the PrismML loader rejects an empty prism.hadamard.weight_names".into(),
                ));
            }
            kv[i].1 = MetaValue::Array {
                elem: MetaType::String,
                values: kept,
            };
        }
    }
    let s = |v: &str| MetaValue::String(v.into());
    kv.push(("modelbuilder.surgery".into(), s("mtp-port")));
    kv.push(("modelbuilder.version".into(), s(env!("CARGO_PKG_VERSION"))));
    if let Some(label) = &opts.reference_label {
        kv.push(("modelbuilder.mtp.reference".into(), s(label)));
    }
    changes.push("modelbuilder.* provenance keys (added; ignored by llama.cpp)".into());

    gguf::write(out, &kv, &tensors)?;
    let bytes = std::fs::metadata(out)
        .map_err(|e| SurgeryError::Output(e.to_string()))?
        .len();

    let mut notes = vec![
        if opts.aligned_to_target {
            "The head was aligned to this target's hidden states (mtp_align).".into()
        } else {
            "The head was trained against the reference trunk, not this target: expect low draft acceptance until the mtp-align stage retrains it against the target's hidden states.".into()
        },
        format!(
            "Run with the PrismML fork: -m <target> -md {} --spec-type draft-mtp",
            out.display()
        ),
    ];
    if target_ir.weight_rotation.is_some() {
        notes.push("Head weights are stored unrotated and are not listed in prism.hadamard.weight_names, so the runtime multiplies them without the activation transform, as their training expects.".into());
    }
    Ok(SurgeryReport {
        output: out.to_owned(),
        bytes,
        tensors: written,
        metadata: changes,
        notes,
    })
}

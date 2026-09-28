//! GQA/MHA → multi-head latent attention (MLA, DeepSeek-V2, arXiv
//! 2405.04434, §2.1): K and V come from one shared latent of `kv_rank`
//! dimensions per token, plus a small decoupled RoPE key of `rope_dim`
//! dimensions, so the cache holds `kv_rank + rope_dim` values per token and
//! layer.
//!
//! Converting a trained model follows MHA2MLA (Ji et al., 2025, arXiv
//! 2502.14837): keep RoPE on a subset of dimensions (partial RoPE), then
//! initialize the latent projections from a joint SVD of the K/V weights
//! and finetune. TransMLA (Meng et al., 2025, arXiv 2502.07864) shows GQA is
//! a special case of MLA, so the conversion can start loss-free before the
//! rank is reduced. `modelbuilder stats --kv-spectra` measures how much of
//! the K/V spectrum a given rank keeps.

use serde::Deserialize;

use super::{backprop_params_from, global_attention_layers, params_of};
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
use mb_ir::{AttentionKind, Mixer, TensorKind};

pub struct Mla;

const ID: &str = "mla";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    kv_rank: u64,
    rope_dim: u64,
}

impl Default for P {
    fn default() -> Self {
        // DeepSeek-V2/V3: d_c = 512, d_h^R = 64.
        Self {
            kv_rank: 512,
            rope_dim: 64,
        }
    }
}

impl Feature for Mla {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "GQA → MLA conversion"
    }

    fn summary(&self) -> &'static str {
        "Convert full-attention layers to multi-head latent attention (MHA2MLA / TransMLA): SVD-initialized latent K/V, partial RoPE, then retrain."
    }

    fn detect(&self, ctx: &Context) -> Detection {
        let mla = ctx
            .ir
            .layers
            .iter()
            .filter(|l| matches!(&l.mixer, Mixer::Attention(a) if a.kind == AttentionKind::Mla))
            .count();
        if mla > 0 {
            Detection::Present(format!("{mla} layers already use MLA"))
        } else {
            Detection::Absent
        }
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        let attn = global_attention_layers(ctx.ir);
        if attn.is_empty() {
            c.blockers
                .push("no full-attention layers to convert".into());
        }
        if let Some(g) = attn.first() {
            let (kv, hd) = (
                g.spec.num_kv_heads.unwrap_or(0),
                g.spec.head_dim.unwrap_or(0),
            );
            let before = 2 * kv * hd;
            if p.kv_rank + p.rope_dim >= before {
                c.blockers.push(format!(
                    "kv_rank + rope_dim = {} is not smaller than the current {before} cached values per token and layer ({kv} KV heads × {hd} × 2)",
                    p.kv_rank + p.rope_dim
                ));
            }
            if p.rope_dim > hd {
                c.blockers
                    .push(format!("rope_dim {} exceeds the head dim {hd}", p.rope_dim));
            }
        }
        c.warnings.push("Partial RoPE changes positional behavior in every converted layer; recovery needs real retraining, not just a finetune of the new projections.".into());
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let attn = global_attention_layers(ctx.ir);
        let layers: Vec<u32> = attn.iter().map(|g| g.layer).collect();
        let per_layer_before = attn.first().map_or(0, |g| {
            2 * g.spec.num_kv_heads.unwrap_or(0) * g.spec.head_dim.unwrap_or(0)
        });
        let per_layer_after = p.kv_rank + p.rope_dim;
        let n = layers.len() as f64;
        let effects = vec![Effect {
            metric: "KV cache per token (BF16)".into(),
            before: 2.0 * per_layer_before as f64 * n,
            after: 2.0 * per_layer_after as f64 * n,
            unit: "bytes".into(),
        }];
        let attn_params = params_of(
            ctx.ir,
            &layers,
            &[
                TensorKind::AttnQ,
                TensorKind::AttnK,
                TensorKind::AttnV,
                TensorKind::AttnO,
                TensorKind::QkNorm,
            ],
        );
        let backprop = layers
            .first()
            .map_or(0, |&l| backprop_params_from(ctx.ir, l));
        let stages = vec![
            Stage {
                name: "mla-recover".into(),
                what: format!("train the converted attention in {} layers (latent K/V, partial-RoPE Q/K, O) against the original", layers.len()),
                trainable_params: attn_params,
                backprop_params: backprop,
                tokens: Range::new(1e9, 5e9),
                seq_len: 4096,
                loss: "KL to the unmodified model plus next-token cross-entropy".into(),
                data: "a pretraining-style mixture; teacher logits from the unmodified model".into(),
                teacher_forward: true,
                trunk: TrunkUse::Restructure,
                precomputed_features: false,
            },
        ];
        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: RiskLevel::High,
                expected: "Loss rises right after conversion (partial RoPE and rank truncation) and long-context retrieval is the slowest to recover.".into(),
                recovery: "A larger kv_rank, choosing which RoPE dimensions to keep by their contribution (MHA2MLA §3.1), more retraining tokens.".into(),
            },
            assumptions: vec![
                format!("Cache after conversion: {per_layer_after} values per token and layer (kv_rank {} + rope_dim {}).", p.kv_rank, p.rope_dim),
                "Trainable: the attention tensors of converted layers; the MLPs stay frozen.".into(),
                "The token budget is a heuristic; MHA2MLA reports recovery with a small fraction of the original pretraining data on models up to 7B.".into(),
                "Low-bit sources: the new projections are initialized in the primal basis, then rotated (if the source is rotated) and quantized to the source format.".into(),
            ],
            confidence: Confidence::Low,
            references: vec![
                "DeepSeek-V2, arXiv 2405.04434, §2.1 (MLA, decoupled RoPE)".into(),
                "Ji et al., 2025, MHA2MLA, arXiv 2502.14837 (partial RoPE, joint SVD init)".into(),
                "Meng et al., 2025, TransMLA, arXiv 2502.07864 (GQA as a special case of MLA)".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let layers: Vec<u32> = global_attention_layers(ctx.ir)
            .iter()
            .map(|g| g.layer)
            .collect();
        Ok(vec![
            format!("In layers {layers:?}: replace attn_k/attn_v with attn_kv_a_mqa [{} , hidden] (latent + RoPE key), attn_kv_a_norm, attn_kv_b [heads × (nope + v), {}].", p.kv_rank + p.rope_dim, p.kv_rank),
            "Initialize attn_kv_b · attn_kv_a from a truncated SVD of the stacked (non-RoPE) K and V projections, in the primal basis.".into(),
            "Metadata: kv_lora_rank, rope dimension count, key/value head sizes, as the deepseek2 loader expects.".into(),
            "Not implemented as surgery yet.".into(),
        ])
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "llama.cpp and vLLM run MLA only for DeepSeek-style architectures; a converted model must be exported in that layout (with its MLPs, norms and vocabulary mapped over) or needs runtime support.".into(),
        ])
    }
}

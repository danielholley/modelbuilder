//! Cross-layer KV sharing, after CSA2's Full/Reuse modes in
//! DeepSeek-V4.1-Flash (arXiv 2609.19969, §2.3.1): in each group of
//! consecutive global-attention layers, the first computes K/V and the rest
//! reuse them, keeping their own queries. Related: Cross-Layer Attention
//! (Brandon et al., 2024).
//!
//! DeepSeek trains this from scratch. Retrofitting it onto a trained model is
//! an extrapolation, and the estimate says so.

use serde::Deserialize;

use super::{backprop_params_from, global_attention_layers, params_of};
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage,
};
use mb_ir::TensorKind;

pub struct KvShare;

const ID: &str = "kv-share";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    /// Global-attention layers per shared KV (one producer + `group - 1` reusers).
    group: u32,
}

impl Default for P {
    fn default() -> Self {
        // The V4.1 decoder uses groups of four (Full/Reindex + 3 × Reuse).
        Self { group: 4 }
    }
}

struct Groups {
    producers: Vec<u32>,
    reusers: Vec<u32>,
}

fn groups(layers: &[u32], group: u32) -> Groups {
    let (mut producers, mut reusers) = (Vec::new(), Vec::new());
    for chunk in layers.chunks(group.max(1) as usize) {
        producers.push(chunk[0]);
        reusers.extend_from_slice(&chunk[1..]);
    }
    Groups { producers, reusers }
}

impl Feature for KvShare {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "Cross-layer KV sharing"
    }

    fn summary(&self) -> &'static str {
        "Group global-attention layers so one layer's K/V is reused by the next ones (CSA2 Full/Reuse modes, DeepSeek-V4.1-Flash)."
    }

    fn detect(&self, ctx: &Context) -> Detection {
        let attn = global_attention_layers(ctx.ir);
        let missing: Vec<u32> = attn
            .iter()
            .filter(|g| {
                params_of(
                    ctx.ir,
                    &[g.layer],
                    &[TensorKind::AttnK, TensorKind::AttnV, TensorKind::AttnQkv],
                ) == 0
            })
            .map(|g| g.layer)
            .collect();
        if missing.is_empty() || missing.len() == attn.len() {
            Detection::Absent
        } else {
            Detection::Present(format!(
                "attention layers {missing:?} have no K/V projections of their own"
            ))
        }
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        let attn = global_attention_layers(ctx.ir);
        if attn.len() < 2 {
            c.blockers.push(format!(
                "needs at least 2 global-attention layers; the model has {}",
                attn.len()
            ));
        }
        if p.group < 2 {
            c.blockers.push("group must be at least 2".into());
        } else if p.group as usize > attn.len() && attn.len() >= 2 {
            c.blockers.push(format!(
                "group {} exceeds the {} global-attention layers",
                p.group,
                attn.len()
            ));
        }
        if let Some(first) = attn.first() {
            if attn.iter().any(|g| {
                (g.spec.num_kv_heads, g.spec.head_dim)
                    != (first.spec.num_kv_heads, first.spec.head_dim)
            }) {
                c.blockers.push("attention layers differ in KV heads or head dim, so their caches can't be shared".into());
            }
        }
        let gaps: Vec<u32> = attn.windows(2).map(|w| w[1].layer - w[0].layer).collect();
        if gaps.iter().any(|&g| g > 1) {
            c.warnings.push(format!(
                "consecutive global-attention layers are {} layers apart, with other mixers between them; CSA2 groups adjacent layers, so sharing across interleaved linear-attention layers is untested",
                gaps.iter().max().unwrap_or(&1)
            ));
        }
        c.warnings.push("DeepSeek trained cross-layer sharing from scratch (45T tokens); retrofitting it onto a trained model is not demonstrated in the cited sources.".into());
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let attn: Vec<u32> = global_attention_layers(ctx.ir)
            .iter()
            .map(|g| g.layer)
            .collect();
        let g = groups(&attn, p.group);
        let keep = g.producers.len() as f64 / attn.len().max(1) as f64;

        let mut effects = Vec::new();
        if let Some(bf16) = ctx
            .report
            .kv_cache
            .precisions
            .iter()
            .find(|x| x.name == "bf16")
        {
            effects.push(Effect {
                metric: "KV cache per token (BF16)".into(),
                before: bf16.bytes_per_token,
                after: bf16.bytes_per_token * keep,
                unit: "bytes".into(),
            });
        }
        let removed = params_of(ctx.ir, &g.reusers, &[TensorKind::AttnK, TensorKind::AttnV]);
        effects.push(Effect {
            metric: "K/V projection params".into(),
            before: removed as f64,
            after: 0.0,
            unit: "params (reuse layers)".into(),
        });

        let trainable = params_of(
            ctx.ir,
            &g.reusers,
            &[TensorKind::AttnQ, TensorKind::AttnO, TensorKind::QkNorm],
        ) + params_of(
            ctx.ir,
            &g.producers,
            &[TensorKind::AttnK, TensorKind::AttnV, TensorKind::QkNorm],
        );
        let backprop = attn.first().map_or(0, |&l| backprop_params_from(ctx.ir, l));
        let stages = vec![
            Stage {
                name: "reuse-adapt".into(),
                what: format!(
                    "{} reuse layers attend over the K/V of their group's producer; train their Q/O and the producers' K/V",
                    g.reusers.len()
                ),
                trainable_params: trainable,
                backprop_params: backprop,
                tokens: Range::new(1e9, 5e9),
                seq_len: 8192,
                loss: "KL to the unmodified model plus MSE on each attention block's output".into(),
                data: "broad pretraining-style text; teacher logits come from the unmodified model".into(),
                teacher_forward: true,
            },
            Stage {
                name: "long-context".into(),
                what: "same tensors, long sequences, to recover retrieval over the shared cache".into(),
                trainable_params: trainable,
                backprop_params: backprop,
                tokens: Range::new(2e8, 1e9),
                seq_len: 65_536,
                loss: "KL to the unmodified model".into(),
                data: "long documents (books, code repositories, multi-document QA)".into(),
                teacher_forward: true,
            },
        ];

        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: if p.group <= 2 { RiskLevel::Medium } else { RiskLevel::High },
                expected: "Reuse layers lose their own view of the context, so expect the largest drops on long-context retrieval and multi-hop tasks until retrained.".into(),
                recovery: "Smaller groups, keeping the most compressible layers as reusers, or adding CSA2's Reindex mode (a per-layer indexer) instead of plain reuse.".into(),
            },
            assumptions: vec![
                "Groups are formed over consecutive global-attention layers in order; the first layer of each group keeps its K/V.".into(),
                "Token budgets are heuristics with no published retrofit to calibrate against.".into(),
                "Trainable tensors are re-quantized to the source scheme; ternary QAT runs inside the same stages.".into(),
            ],
            confidence: Confidence::Low,
            references: vec![
                "DeepSeek-V4.1-Flash, arXiv 2609.19969, §2.3.1 (CSA2 cross-layer KV and index reuse) and §4.2.1 (group layout)".into(),
                "Brandon et al., 2024, Reducing Transformer Key-Value Cache Size with Cross-Layer Attention".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let attn: Vec<u32> = global_attention_layers(ctx.ir)
            .iter()
            .map(|g| g.layer)
            .collect();
        let g = groups(&attn, p.group);
        Ok(vec![
            format!("KV producers (keep K/V): layers {:?}", g.producers),
            format!("Reuse layers (drop attn_k, attn_v, attn_k_norm): layers {:?}", g.reusers),
            "Record each reuse layer's source layer in new metadata (no existing key; runtime patch required).".into(),
            "Fine-tuned tensors are re-folded into the rotated basis (if any) and re-quantized to the source scheme.".into(),
        ])
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "No runtime supports this layout yet: stock llama.cpp and the PrismML fork expect K/V projections in every attention layer. Running the result needs a graph change (reuse layers read the source layer's cache) and a cache allocator that skips reuse layers.".into(),
            "HF export needs matching custom modeling code (trust_remote_code).".into(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::groups;

    #[test]
    fn grouping() {
        let g = groups(&[3, 7, 11, 15, 19], 2);
        assert_eq!(g.producers, [3, 11, 19]);
        assert_eq!(g.reusers, [7, 15]);
    }
}

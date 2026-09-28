//! Depth pruning: remove a contiguous block of layers, then heal with a short
//! finetune. Follows Gromov et al., 2024, "The Unreasonable Ineffectiveness
//! of the Deeper Layers" (arXiv 2403.17887): the best block to drop is found
//! by the angular distance between the representations entering and leaving
//! it, is usually deep (but not the last layer), and a QLoRA "healing" run
//! on ~164M tokens of C4 recovers much of the loss. ShortGPT (Men et al.,
//! 2024, arXiv 2403.03853) scores single layers by Block Influence
//! (1 − cosine similarity of input and output).
//!
//! Choosing the block well needs activations (the Python probes); without
//! them this plugin proposes the deepest block that keeps the final layer.

use serde::Deserialize;

use super::gib;
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
use mb_ir::{Mixer, ModelIr};

pub struct Prune;

const ID: &str = "prune-layers";

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    /// How many layers to remove (default: a quarter).
    count: Option<u32>,
    /// First layer of the removed block (default: the deepest block before the last layer).
    start: Option<u32>,
    /// Finetune (LoRA) after pruning.
    #[serde(default = "yes")]
    heal: bool,
}

fn yes() -> bool {
    true
}

/// The mixer kind of each layer, as a small code for pattern checks.
fn pattern(ir: &ModelIr) -> Vec<u8> {
    ir.layers
        .iter()
        .map(|l| match &l.mixer {
            Mixer::Attention(a) if a.sliding_window.is_some() => 1,
            Mixer::Attention(_) => 0,
            Mixer::LinearAttention(_) => 2,
            Mixer::Unknown => 3,
        })
        .collect()
}

/// Smallest period of the layer pattern (the whole length if aperiodic).
pub(crate) fn period(p: &[u8]) -> usize {
    (1..=p.len())
        .find(|&k| (k..p.len()).all(|i| p[i] == p[i - k]))
        .unwrap_or(p.len().max(1))
}

/// The block to remove: `[start, start + count)`.
fn block(ir: &ModelIr, p: &P) -> (u32, u32) {
    let n = ir.layers.len() as u32;
    let per = period(&pattern(ir)) as u32;
    let mut count = p.count.unwrap_or((n / 4).max(1));
    if p.count.is_none() && per > 1 {
        // Keep a periodic hybrid pattern intact.
        count = (count / per).max(1) * per;
    }
    // Removing a whole number of periods keeps a periodic pattern at any start.
    let start = p
        .start
        .unwrap_or_else(|| n.saturating_sub(1).saturating_sub(count));
    (start, count)
}

impl Feature for Prune {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "Layer pruning"
    }

    fn summary(&self) -> &'static str {
        "Remove a contiguous block of layers (Gromov et al., 2024), then heal with a short LoRA finetune."
    }

    fn detect(&self, _ctx: &Context) -> Detection {
        Detection::Absent
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        let n = ctx.ir.layers.len() as u32;
        let (start, count) = block(ctx.ir, &p);
        if n < 2 {
            c.blockers.push("needs at least 2 layers".into());
        } else if count == 0 || count >= n {
            c.blockers
                .push(format!("count must be between 1 and {}", n - 1));
        } else if start + count > n {
            c.blockers.push(format!(
                "block {start}..{} runs past the {n} layers",
                start + count
            ));
        }
        let pat = pattern(ctx.ir);
        let per = period(&pat);
        if per > 1 && per < pat.len() && count as usize % per != 0 {
            c.blockers.push(format!(
                "the layers follow a period-{per} pattern of mixer types; removing {count} (not a multiple of {per}) breaks it, and runtimes that derive layer types from an interval would misread the result"
            ));
        }
        if count as f64 > 0.5 * n as f64 {
            c.warnings.push("Gromov et al. saw knowledge benchmarks collapse beyond roughly half the layers, and earlier on smaller models".into());
        }
        if p.start.is_none() {
            c.warnings.push("the block is a default guess; measure angular distances between layers on real text (behavioral probes) to choose it".into());
        }
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let (start, count) = block(ctx.ir, &p);
        let removed: Vec<_> = ctx
            .ir
            .layers
            .iter()
            .filter(|l| (start..start + count).contains(&l.index))
            .collect();
        let removed_params: u64 = removed.iter().map(|l| l.params).sum();
        let n = ctx.cost_inputs().forward_params;
        let removed_attn = removed
            .iter()
            .filter(|l| matches!(&l.mixer, Mixer::Attention(a) if a.sliding_window.is_none()))
            .count();
        let mut effects = vec![
            Effect {
                metric: "layers".into(),
                before: ctx.ir.layers.len() as f64,
                after: (ctx.ir.layers.len() - removed.len()) as f64,
                unit: "layers".into(),
            },
            Effect {
                metric: "forward params".into(),
                before: n as f64,
                after: n.saturating_sub(removed_params) as f64,
                unit: "params".into(),
            },
        ];
        let kv = &ctx.report.kv_cache;
        if let Some(bf16) = kv.precisions.iter().find(|x| x.name == "bf16") {
            let keep = 1.0 - removed_attn as f64 / kv.global_layers.max(1) as f64;
            effects.push(Effect {
                metric: "KV cache per token (BF16)".into(),
                before: bf16.bytes_per_token,
                after: bf16.bytes_per_token * keep,
                unit: "bytes".into(),
            });
        }
        let after = n.saturating_sub(removed_params);
        let stages = if p.heal {
            vec![Stage {
                name: "heal".into(),
                what: "LoRA on every remaining linear layer, next-token loss".into(),
                trainable_params: after / 100,
                backprop_params: after,
                tokens: Range::new(1e8, 5e8),
                seq_len: 2048,
                loss: "next-token cross-entropy (optionally KL to the unpruned model)".into(),
                data: "general web text (the paper used C4)".into(),
                teacher_forward: false,
                trunk: TrunkUse::Restructure,
                precomputed_features: false,
            }]
        } else {
            vec![]
        };
        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: if count as f64 <= 0.25 * ctx.ir.layers.len() as f64 { RiskLevel::Medium } else { RiskLevel::High },
                expected: "Knowledge QA holds up best; reasoning, math and long-context tasks drop first, and much more on small models.".into(),
                recovery: "Prune fewer layers, choose the block by measured angular distance, heal longer or with distillation from the unpruned model.".into(),
            },
            assumptions: vec![
                format!("Removes layers {start}..{} ({} params, {:.2} GiB at BF16).", start + count, removed_params, gib(removed_params as f64 * 2.0)),
                "LoRA trainable params taken as ~1% of the remaining model (a heuristic for rank 64 on every linear layer).".into(),
                "Healing budget brackets the paper's 164M tokens; it is not calibrated for other model sizes.".into(),
            ],
            confidence: Confidence::Low,
            references: vec![
                "Gromov et al., 2024, The Unreasonable Ineffectiveness of the Deeper Layers, arXiv 2403.17887 (§3 method, §4 results)".into(),
                "Men et al., 2024, ShortGPT, arXiv 2403.03853 (Block Influence)".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let (start, count) = block(ctx.ir, &p);
        Ok(vec![
            format!("Drop every tensor of layers {start}..{}; renumber the layers above down by {count} (`modelbuilder surgery prune --layers {start}..{}`).", start + count, start + count),
            "Metadata: block_count decreases; per-layer arrays (e.g. KV heads per layer) lose the removed entries.".into(),
            "Remaining tensors are copied byte for byte, in their source format and basis.".into(),
        ])
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "A pruned model is the same architecture with fewer layers, so every runtime runs it unchanged.".into(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::period;

    #[test]
    fn periods() {
        assert_eq!(period(&[2, 2, 2, 0, 2, 2, 2, 0]), 4);
        assert_eq!(period(&[0, 0, 0]), 1);
        assert_eq!(period(&[0, 2, 2]), 3);
    }
}

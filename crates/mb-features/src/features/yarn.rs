//! Context extension with YaRN RoPE scaling (Peng et al., 2023, "YaRN:
//! Efficient Context Window Extension of Large Language Models", arXiv
//! 2309.00071): per-frequency interpolation ("NTK-by-parts") plus an attention
//! temperature, optionally followed by a short long-context finetune.
//!
//! The paper's Llama 2 runs finetuned for 400 steps of 64 sequences at 64k
//! tokens (about 1.7B tokens) for a 16× extension (§4). Several model releases
//! instead enable static YaRN with no finetune for a moderate factor.

use serde::Deserialize;

use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
use mb_ir::Mixer;

pub struct Yarn;

const ID: &str = "yarn";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    /// Context multiplier.
    factor: f64,
    /// Context the model was trained at (default: the checkpoint's).
    original_context: Option<u64>,
    /// Run a long-context finetune after enabling the scaling.
    finetune: bool,
}

impl Default for P {
    fn default() -> Self {
        Self {
            factor: 4.0,
            original_context: None,
            finetune: true,
        }
    }
}

fn original(ctx: &Context, p: &P) -> Option<u64> {
    p.original_context
        .or(ctx.report.provenance.original_context)
        .or(ctx.ir.max_positions)
}

fn has_rope(ctx: &Context) -> bool {
    ctx.ir.rope.theta.is_some()
        && ctx
            .ir
            .layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::Attention(_)))
}

impl Feature for Yarn {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "Context extension (YaRN)"
    }

    fn summary(&self) -> &'static str {
        "Extend the context window with YaRN RoPE scaling, optionally followed by a short long-context finetune."
    }

    fn detect(&self, ctx: &Context) -> Detection {
        match &ctx.ir.rope.scaling {
            Some(s) => {
                let kind = s
                    .get("rope_type")
                    .or_else(|| s.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                if kind == "yarn" {
                    Detection::Present(format!("RoPE scaling is already YaRN: {s}"))
                } else {
                    Detection::Partial(format!("RoPE scaling `{kind}` is configured: {s}"))
                }
            }
            None => Detection::Absent,
        }
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        if !has_rope(ctx) {
            c.blockers
                .push("no RoPE attention layers (no rope theta in the config)".into());
        }
        if p.factor <= 1.0 || !p.factor.is_finite() {
            c.blockers.push("factor must be greater than 1".into());
        }
        if original(ctx, &p).is_none() {
            c.blockers
                .push("the trained context length is unknown; pass original_context".into());
        }
        if ctx.ir.rope.scaling.is_some() {
            c.warnings.push(
                "the model already scales RoPE; YaRN replaces that scaling, and original_context should be the pre-scaling length".into(),
            );
        }
        if ctx
            .ir
            .layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::LinearAttention(_)))
        {
            c.warnings.push("linear-attention layers have no RoPE and are unaffected; their long-context behavior is untested by the scaling".into());
        }
        if !p.finetune && p.factor > 4.0 {
            c.warnings.push(format!(
                "a {}× extension without finetuning is beyond what releases enable statically (up to ~4×)",
                p.factor
            ));
        }
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let orig = original(ctx, &p).unwrap_or(0);
        let new = (orig as f64 * p.factor).round();
        let mut effects = vec![Effect {
            metric: "context length".into(),
            before: orig as f64,
            after: new,
            unit: "tokens".into(),
        }];
        if let Some(bf16) = ctx
            .report
            .kv_cache
            .precisions
            .iter()
            .find(|x| x.name == "bf16")
        {
            effects.push(Effect {
                metric: "KV cache for one full-length sequence (BF16)".into(),
                before: bf16.bytes_per_token * orig as f64,
                after: bf16.bytes_per_token * new,
                unit: "bytes".into(),
            });
        }
        let n = ctx.cost_inputs().forward_params;
        let stages = if p.finetune {
            vec![Stage {
                name: "long-context".into(),
                what: format!("finetune the whole model at up to {new} tokens with YaRN enabled"),
                trainable_params: n,
                backprop_params: n,
                tokens: Range::new(4e8, 2e9),
                seq_len: (new as u64).min(65_536),
                loss: "next-token cross-entropy on long documents".into(),
                data: "long documents (books, code repositories, long-form web), packed to the target length; mix in short data to protect short-context quality".into(),
                teacher_forward: false,
                trunk: TrunkUse::Adapt,
                precomputed_features: false,
            }]
        } else {
            vec![]
        };
        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: if p.finetune || p.factor <= 4.0 { RiskLevel::Low } else { RiskLevel::Medium },
                expected: "Slight perplexity increase on short contexts; retrieval beyond the trained length degrades without the finetune.".into(),
                recovery: "Finetune longer, lower the factor, or use dynamic scaling (factor grows with the sequence) at inference.".into(),
            },
            assumptions: vec![
                format!("Extends {orig} → {new} tokens (factor {}).", p.factor),
                "Token budget follows the paper's Llama 2 runs (~1.7B tokens for 16×); smaller factors likely need less. The range is a heuristic around that point.".into(),
                "Seq len is capped at 64k for pricing; ring or context-parallel attention is needed beyond that on most hardware.".into(),
                "Low-bit sources: the finetune runs as QAT at the source format, and export re-quantizes.".into(),
            ],
            confidence: Confidence::Medium,
            references: vec![
                "Peng et al., 2023, YaRN, arXiv 2309.00071 (§3.2 NTK-by-parts, §3.4 attention temperature, §4 training)".into(),
                "llama.cpp: rope.scaling.{type,factor,original_context_length} GGUF keys (llama-model.cpp)".into(),
                "transformers: rope_scaling {rope_type: yarn} (modeling_rope_utils.py)".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let orig = original(ctx, &p).unwrap_or(0);
        let new = (orig as f64 * p.factor).round() as u64;
        Ok(vec![
            format!("GGUF metadata: {{arch}}.rope.scaling.type = yarn, .factor = {}, .original_context_length = {orig}, {{arch}}.context_length = {new} (`modelbuilder surgery yarn`).", p.factor),
            format!("HF config: rope_scaling = {{rope_type: yarn, factor: {}, original_max_position_embeddings: {orig}}}, max_position_embeddings = {new}.", p.factor),
            "No tensors change until the finetune; finetuned tensors are written back in the source format (`surgery replace`).".into(),
        ])
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "llama.cpp, vLLM and transformers all implement YaRN from the metadata; no code changes.".into(),
            "Serving at the full length needs the KV-cache memory shown above; combine with a quantized KV cache (fp4-kv) if it doesn't fit.".into(),
        ])
    }
}

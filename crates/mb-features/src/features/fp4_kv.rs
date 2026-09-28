//! FP4 main KV cache, following DeepSeek-V4.1-Flash (arXiv 2609.19969,
//! §2.4.4): E2M1 values with one E4M3 scale per 16 channels (4.5 bits per
//! element), quantized after RoPE, introduced by QAT during post-training.
//! The paper keeps its sliding-window KV in FP8.

use serde::Deserialize;

use super::{backprop_params_from, global_attention_layers, params_of};
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
use mb_ir::TensorKind;

pub struct Fp4Kv;

const ID: &str = "fp4-kv";

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    /// Quantize the cache at inference only.
    Ptq,
    /// Quantization-aware fine-tune of the K/V projections (the paper's approach).
    #[default]
    Qat,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    mode: Mode,
}

fn kv_bytes(ctx: &Context, name: &str) -> Option<(f64, f64)> {
    ctx.report
        .kv_cache
        .precisions
        .iter()
        .find(|p| p.name == name)
        .map(|p| (p.bytes_per_token, p.bytes_at_context))
}

impl Feature for Fp4Kv {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "FP4 KV cache"
    }

    fn summary(&self) -> &'static str {
        "Store the attention KV cache in 4-bit floating point (E2M1 + E4M3 scale per 16 channels), as in DeepSeek-V4.1-Flash."
    }

    fn detect(&self, _ctx: &Context) -> Detection {
        // KV cache precision is a runtime choice; checkpoints don't record it.
        Detection::Absent
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let _: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        let kv = &ctx.report.kv_cache;
        if kv.global_layers + kv.windowed_layers == 0 {
            c.blockers
                .push("the model has no attention layers with a per-token KV cache".into());
        }
        for g in global_attention_layers(ctx.ir) {
            if g.spec.head_dim.is_some_and(|d| d % 16 != 0) {
                c.warnings.push(format!(
                    "layer {}: head dim {:?} is not a multiple of the 16-channel scale group",
                    g.layer, g.spec.head_dim
                ));
            }
        }
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut effects = Vec::new();
        if let (Some(bf16), Some(fp4)) = (kv_bytes(ctx, "bf16"), kv_bytes(ctx, "fp4_e2m1_g16")) {
            effects.push(Effect {
                metric: "KV cache per token".into(),
                before: bf16.0,
                after: fp4.0,
                unit: "bytes".into(),
            });
            if let Some(c) = ctx.report.kv_cache.context {
                effects.push(Effect {
                    metric: format!("KV cache at {c} tokens"),
                    before: bf16.1,
                    after: fp4.1,
                    unit: "bytes".into(),
                });
            }
        }

        let attn: Vec<u32> = global_attention_layers(ctx.ir)
            .iter()
            .map(|g| g.layer)
            .collect();
        let stages = match p.mode {
            Mode::Ptq => vec![],
            Mode::Qat => {
                let trainable = params_of(
                    ctx.ir,
                    &attn,
                    &[TensorKind::AttnK, TensorKind::AttnV, TensorKind::QkNorm],
                );
                vec![Stage {
                    name: "kv-fp4-qat".into(),
                    what: format!(
                        "fine-tune K/V projections and QK-norms of {} attention layers with the cache fake-quantized to FP4 after RoPE",
                        attn.len()
                    ),
                    trainable_params: trainable,
                    backprop_params: attn.first().map_or(0, |&l| backprop_params_from(ctx.ir, l)),
                    tokens: Range::new(1e8, 5e8),
                    seq_len: 16_384,
                    loss: "KL divergence to the same model with a BF16 KV cache (self-distillation)".into(),
                    data: "general text with a share of long documents; the model's own generations work, since the teacher is the unmodified model".into(),
                    teacher_forward: true,
                trunk: TrunkUse::Adapt,
                precomputed_features: false,
                }]
            }
        };

        let (risk, confidence) = match p.mode {
            Mode::Ptq => (
                QualityRisk {
                    level: RiskLevel::Medium,
                    expected: "Small losses concentrated in long-context retrieval; not measured for this model. Measure them first with `modelbuilder_train probe kv-cache` (perplexity and retrieval per cache type).".into(),
                    recovery: "Run the long-context probes; if retrieval drops, switch to mode = \"qat\".".into(),
                },
                Confidence::Medium,
            ),
            Mode::Qat => (
                QualityRisk {
                    level: RiskLevel::Low,
                    expected: "DeepSeek adopted FP4 main KV via QAT with no reported regression; the fine-tune only adapts K/V to the cache format.".into(),
                    recovery: "Increase QAT tokens or include the Q projections.".into(),
                },
                Confidence::Low,
            ),
        };

        Ok(Estimate {
            effects,
            stages,
            risk,
            assumptions: vec![
                "4.5 bits per cached element (4-bit value + 8-bit scale per 16 channels); llama.cpp's q4_0 cache type is also 4.5 bits (FP16 scale per 32).".into(),
                "QAT token budget (0.1–0.5B) is a heuristic: the paper states QAT was done in post-training but not for how many tokens.".into(),
                "Only the main (per-token) KV is converted; the paper keeps sliding-window KV in FP8.".into(),
            ],
            confidence,
            references: vec![
                "DeepSeek-V4.1-Flash, arXiv 2609.19969, §2.4.4 (FP4 main KV cache)".into(),
            ],
        })
    }

    fn surgery_outline(
        &self,
        _ctx: &Context,
        params: &Params,
    ) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        Ok(match p.mode {
            Mode::Ptq => vec!["No checkpoint changes; the cache format is a runtime setting.".into()],
            Mode::Qat => vec![
                "No shape changes. Updated K/V projections and QK-norms replace the originals.".into(),
                "Updated weights are re-quantized to the source scheme (and re-folded into the rotated basis, if any).".into(),
            ],
        })
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "llama.cpp: `--cache-type-k q4_0 --cache-type-v q4_0` is the nearest available cache format (same 4.5 bits); the paper's E2M1+E4M3/16 layout is not a llama.cpp cache type, so QAT should target the format the runtime will actually use.".into(),
            "PrismML fork known issues: CUDA builds may need -DGGML_CUDA_FA_ALL_QUANTS=ON for quantized caches with flash attention; crashes near 200K tokens with a quantized cache are reported and under investigation.".into(),
        ])
    }
}

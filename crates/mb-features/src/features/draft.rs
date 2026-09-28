//! Speculative-decoding draft heads trained against a frozen target:
//!
//! - **Medusa** (Cai et al., 2024, arXiv 2401.10774): K extra heads on the
//!   last hidden state, head k predicts token t+k+1. Each head is one
//!   residual SiLU block plus its own LM head, initialized from the target's
//!   (Medusa-1 trains the heads with the backbone frozen).
//! - **EAGLE** (Li et al., 2024, arXiv 2401.15077): an autoregressive draft
//!   over features: an FC layer fuses the target's last hidden state with the
//!   next token's embedding (2h → h), one decoder layer follows, and the
//!   target's frozen LM head produces the draft. EAGLE-3 (arXiv 2503.01840)
//!   fuses low, middle and high-layer features instead.
//! - **DSpark** (DeepSeek-V4.1-Flash, arXiv 2609.19969): a 3-block drafter
//!   with a 128-token sliding window that drafts 5 positions per pass with a
//!   Markov head and a confidence head, trained with the backbone frozen.
//!
//! All three learn from the trunk's features, so the features can be
//! precomputed once in the serving runtime, as for `mtp`.

use serde::Deserialize;

use super::gib;
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
use mb_ir::Mixer;

pub struct DraftHead;

const ID: &str = "draft-head";

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Medusa,
    #[default]
    Eagle,
    Dspark,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    kind: Kind,
    /// Medusa: number of heads (default 4). EAGLE/DSpark: decoder blocks (default 1 / 3).
    size: Option<u32>,
}

/// Parameters of one attention decoder block of the trunk (the drafter copies its shape).
fn block_params(ctx: &Context) -> u64 {
    let attn: Vec<u64> = ctx
        .ir
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, Mixer::Attention(_)))
        .map(|l| l.params)
        .collect();
    if attn.is_empty() {
        let n = ctx.ir.layers.len().max(1) as u64;
        ctx.ir.layers.iter().map(|l| l.params).sum::<u64>() / n
    } else {
        attn.iter().sum::<u64>() / attn.len() as u64
    }
}

struct Shape {
    trainable: u64,
    /// Whether the loss runs through the target's frozen LM head.
    shared_head: bool,
    what: String,
}

fn shape(ctx: &Context, p: &P) -> Shape {
    let h = ctx.ir.hidden_size.unwrap_or(0);
    let v = ctx.ir.vocab_size.unwrap_or(0);
    match p.kind {
        Kind::Medusa => {
            let k = u64::from(p.size.unwrap_or(4));
            Shape {
                trainable: k * (h * h + h + v * h),
                shared_head: false,
                what: format!("{k} Medusa heads (residual block + own LM head each)"),
            }
        }
        Kind::Eagle => {
            let n = u64::from(p.size.unwrap_or(1));
            Shape {
                trainable: 2 * h * h + n * block_params(ctx),
                shared_head: true,
                what: format!("EAGLE drafter: feature/embedding fusion + {n} decoder block(s)"),
            }
        }
        Kind::Dspark => {
            let n = u64::from(p.size.unwrap_or(3));
            Shape {
                // The confidence and Markov heads are small next to the blocks.
                trainable: 2 * h * h + n * block_params(ctx) + h,
                shared_head: true,
                what: format!(
                    "DSpark drafter: {n} sliding-window blocks, 5 draft positions per pass"
                ),
            }
        }
    }
}

impl Feature for DraftHead {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "Draft heads (Medusa / EAGLE / DSpark)"
    }

    fn summary(&self) -> &'static str {
        "Train a speculative-decoding drafter on the frozen target's features: Medusa heads, an EAGLE layer, or a DSpark block drafter."
    }

    fn detect(&self, ctx: &Context) -> Detection {
        let named = ctx.ir.raw.tensors.iter().any(|t| {
            ["medusa", "eagle", "draft"]
                .iter()
                .any(|k| t.name.contains(k))
        });
        if named {
            Detection::Present("the checkpoint has draft-head tensors".into())
        } else if ctx.ir.mtp.is_some() {
            Detection::Partial(
                "the model has an MTP head, which can serve as a drafter (see `mtp`)".into(),
            )
        } else {
            Detection::Absent
        }
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        if ctx.ir.hidden_size.is_none() || ctx.ir.vocab_size.is_none() {
            c.blockers
                .push("hidden size or vocabulary size unknown".into());
        }
        if p.size == Some(0) {
            c.blockers.push("size must be at least 1".into());
        }
        if p.kind == Kind::Dspark {
            c.warnings.push("DSpark's modules are described in the paper but no reference code or runtime is public; the drafter would need custom inference support".into());
        }
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let s = shape(ctx, &p);
        let h = ctx.ir.hidden_size.unwrap_or(0);
        let v = ctx.ir.vocab_size.unwrap_or(0);
        let bits = ctx.report.quantization.trunk_bits_per_param;
        let effects = vec![Effect {
            metric: "drafter params".into(),
            before: 0.0,
            after: s.trainable as f64,
            unit: "params".into(),
        }];
        let stages = vec![Stage {
            name: format!("{:?}-train", p.kind).to_lowercase(),
            what: format!("{} with the trunk frozen", s.what),
            trainable_params: s.trainable,
            backprop_params: s.trainable + if s.shared_head { v * h } else { 0 },
            tokens: Range::new(5e7, 3e8),
            seq_len: 2048,
            loss: match p.kind {
                Kind::Medusa => "cross-entropy of head k on token t+k+1".into(),
                _ => "feature regression (smooth L1) plus cross-entropy through the frozen LM head".into(),
            },
            data: "the target model's own responses to chat, code and reasoning prompts (self-distillation), so drafts match what verification accepts".into(),
            teacher_forward: false,
            trunk: TrunkUse::Frozen,
            precomputed_features: true,
        }];
        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: RiskLevel::Low,
                expected: "Outputs are unchanged when verification is exact; only the speedup is at stake.".into(),
                recovery: "More data from the served traffic, a larger drafter, or tree drafting at inference.".into(),
            },
            assumptions: vec![
                format!("Drafter size {} ({:.2} GiB at BF16, {:.2} GiB at the trunk's {bits:.2} bits/weight).", s.trainable, gib(s.trainable as f64 * 2.0), gib(s.trainable as f64 * bits / 8.0)),
                "Token budget brackets the published training sets (tens of thousands of chat dialogues); it is a heuristic, not a calibration.".into(),
                "Speedup depends on acceptance and the runtime's draft overhead; measure it (`modelbuilder_train bench-draft`).".into(),
            ],
            confidence: Confidence::Medium,
            references: vec![
                "Cai et al., 2024, Medusa, arXiv 2401.10774 (§2.1 heads, §2.2 Medusa-1 training)".into(),
                "Li et al., 2024, EAGLE, arXiv 2401.15077 (§3 architecture and training); EAGLE-3, arXiv 2503.01840".into(),
                "DeepSeek-V4.1-Flash, arXiv 2609.19969 (DSpark drafter)".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let s = shape(ctx, &p);
        Ok(vec![
            format!(
                "Write the trained {} as a separate draft checkpoint; the target is untouched.",
                s.what
            ),
            if s.shared_head {
                "The drafter reuses the target's embedding and LM head, so they are not duplicated."
                    .into()
            } else {
                "Each head carries its own LM head (initialized from the target's).".into()
            },
        ])
    }

    fn export_notes(&self, _ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        Ok(vec![match p.kind {
            Kind::Medusa | Kind::Eagle => "vLLM and SGLang load Medusa and EAGLE drafters; check your llama.cpp build for the draft types it supports.".into(),
            Kind::Dspark => "No public runtime runs DSpark drafters; this needs custom inference code.".into(),
        }])
    }
}

//! Multi-token prediction (MTP) head for self-speculative decoding.
//!
//! Two routes:
//! - **Port** the head from a reference model that has one (typically the base
//!   model a quantized or fine-tuned target was derived from, e.g. one with
//!   `mtp_num_hidden_layers = 1`), then realign it to the target's trunk with
//!   the trunk frozen.
//! - **Train** a fresh head, initialized from the last full-attention block.
//!
//! GGUF layout follows llama.cpp's nextn convention (see the architecture
//! adapters in `mb-surgery`): the head is an extra full-attention decoder
//! block `blk.{n_layer}` plus `blk.{n_layer}.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`.

use serde::Deserialize;

use super::{gib, global_attention_layers};
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage,
};

pub struct Mtp;

const ID: &str = "mtp";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    /// Path of a reference model to port the head from. The caller loads it
    /// into [`Context::reference`].
    from: Option<String>,
    depth: u32,
}

impl Default for P {
    fn default() -> Self {
        Self {
            from: None,
            depth: 1,
        }
    }
}

/// Parameters of the head: measured from the reference if available, otherwise
/// one full-attention block plus `eh_proj` (2h × h) and three norms.
fn head_params(ctx: &Context) -> (u64, bool) {
    if let Some(m) = ctx.reference.and_then(|r| r.mtp.as_ref()) {
        return (m.params, true);
    }
    let h = ctx.ir.hidden_size.unwrap_or(0);
    let block = global_attention_layers(ctx.ir)
        .last()
        .and_then(|g| ctx.ir.layers.iter().find(|l| l.index == g.layer))
        .map_or(0, |l| l.params);
    (block + 2 * h * h + 3 * h, false)
}

impl Feature for Mtp {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "MTP head (self-speculative decoding)"
    }

    fn summary(&self) -> &'static str {
        "Add a multi-token-prediction block that drafts the next-next token for speculative decoding, trained with the trunk frozen."
    }

    fn detect(&self, ctx: &Context) -> Detection {
        match &ctx.ir.mtp {
            Some(m) => {
                Detection::Present(format!("{} module(s), {} params", m.num_modules, m.params))
            }
            None => Detection::Absent,
        }
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        if ctx.ir.mtp.is_some() {
            c.warnings
                .push("the model already has an MTP head; this would retrain or replace it".into());
        }
        if p.depth != 1 {
            c.warnings.push(format!("depth {}: only depth-1 heads can be ported or run by llama.cpp today; deeper heads must be trained fresh", p.depth));
        }
        if global_attention_layers(ctx.ir).is_empty() {
            c.blockers
                .push("no full-attention layer to model the MTP block on".into());
        }
        match (p.from.as_deref(), ctx.reference) {
            (Some(path), None) => c.blockers.push(format!("reference model `{path}` was not loaded")),
            (_, Some(r)) => {
                if r.mtp.is_none() {
                    c.blockers.push("the reference model has no MTP head to port".into());
                }
                if r.hidden_size != ctx.ir.hidden_size || r.vocab_size != ctx.ir.vocab_size {
                    c.blockers.push(format!(
                        "hidden/vocab differ: reference {:?}/{:?} vs model {:?}/{:?}",
                        r.hidden_size, r.vocab_size, ctx.ir.hidden_size, ctx.ir.vocab_size
                    ));
                }
                let dims = |ir| {
                    global_attention_layers(ir)
                        .first()
                        .map(|g| (g.spec.num_heads, g.spec.num_kv_heads, g.spec.head_dim, g.spec.output_gate))
                };
                if dims(r) != dims(ctx.ir) {
                    c.blockers.push(format!("attention shapes differ: reference {:?} vs model {:?}", dims(r), dims(ctx.ir)));
                }
            }
            (None, None) => c.warnings.push(
                "no reference head given (`from`): a fresh head needs roughly 10× more training than a ported one".into(),
            ),
        }
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let (n_head, ported) = head_params(ctx);
        let n_head = n_head * u64::from(p.depth.max(1));
        let bits = ctx.report.quantization.trunk_bits_per_param;
        let effects = vec![
            Effect {
                metric: "MTP head params".into(),
                before: 0.0,
                after: n_head as f64,
                unit: "params".into(),
            },
            Effect {
                metric: format!("head memory at the trunk's {bits:.2} bits/weight"),
                before: 0.0,
                after: n_head as f64 * bits / 8.0,
                unit: "bytes".into(),
            },
        ];
        let tokens = if ported {
            Range::new(5e7, 3e8)
        } else {
            Range::new(5e8, 2e9)
        };
        let stages = vec![Stage {
            name: "mtp-align".into(),
            what: format!(
                "{} MTP block with the trunk frozen, as QAT at the trunk's weight format",
                if ported { "realign the ported" } else { "train a fresh" }
            ),
            trainable_params: n_head,
            // Gradients stop at the head: the trunk only runs forward.
            backprop_params: n_head,
            tokens,
            seq_len: 4096,
            loss: "cross-entropy on token t+2 from the trunk's hidden state at t and the embedding of t+1".into(),
            data: "the target model's own generations (self-distillation), so drafts match what verification accepts; prompts from chat, code and reasoning sets".into(),
            teacher_forward: false,
        }];
        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: RiskLevel::Low,
                expected: "No change to outputs: speculative verification is exact. The only risk is a smaller speedup than hoped.".into(),
                recovery: "More alignment tokens, data closer to the served traffic, or a larger multi-block drafter.".into(),
            },
            assumptions: vec![
                format!(
                    "Head size {} ({:.2} GiB at BF16).",
                    if ported { "measured from the reference model's MTP tensors" } else { "estimated as one full-attention block + eh_proj + norms" },
                    gib(n_head as f64 * 2.0)
                ),
                "Token budgets are heuristics; EAGLE-style draft heads are typically trained on well under 1B tokens.".into(),
                "Speedup depends on acceptance and on the runtime's draft overhead; measure it with `modelbuilder_train bench-draft` on the deployment hardware rather than trusting this estimate.".into(),
                "Frozen-trunk stages can precompute trunk features once (the trunk forward then runs at inference speed in the serving runtime, not in training).".into(),
            ],
            confidence: if ported { Confidence::Medium } else { Confidence::Low },
            references: vec![
                "llama.cpp: nextn tensors in the model loaders, common/speculative.cpp (--spec-type draft-mtp)".into(),
                "EAGLE (Li et al., 2024): draft heads on the target's features, trained with the target frozen".into(),
                "DeepSeek-V3 technical report (MTP modules)".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let _: P = parse_params(ID, params)?;
        let n = ctx.ir.layers.len();
        let mut out = vec![
            format!("Add block blk.{n}: attn_norm, attn_q (gated, 2× rows), attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm, post_attention_norm, ffn_gate, ffn_up, ffn_down."),
            format!("Add blk.{n}.nextn.eh_proj [h, 2h], nextn.enorm, nextn.hnorm, nextn.shared_head_norm (HF: mtp.fc, mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden, mtp.norm)."),
            format!("Metadata: block_count {n} → {}, nextn_predict_layers = 1.", n + 1),
        ];
        if let Some(r) = &ctx.ir.weight_rotation {
            out.push(format!(
                "Rotation: a BF16 port stays unrotated (not listed in {p}.weight_names, so it gets a plain matmul). A ternary head after training must be folded into the {} basis with the existing sign vectors and listed in {p}.weight_names; nextn.eh_proj (input width 2h) is not a foldable kind and has no sign vector, so it stays unrotated either way.",
                r.scheme,
                p = r.metadata_prefix
            ));
        }
        out.push("Implemented for qwen35 GGUF targets as an MTP-only sidecar: `modelbuilder surgery mtp <target.gguf> --from <reference> -o <name>-mtp.gguf`.".into());
        Ok(out)
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "PrismML fork: run with `--spec-type draft-mtp`; MTP support for these files landed in PR #205 (merged 2026-09-21), so use a build from after that date.".into(),
            "The head can ship as a separate MTP-only GGUF (the qwen35 loader accepts a file without trunk layers); name it with \"mtp\" in the filename to be auto-discovered next to the model. It also carries the embedding, final norm and LM head.".into(),
        ])
    }
}

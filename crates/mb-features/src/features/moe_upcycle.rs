//! Dense → MoE "sparse upcycling" (Komatsuzaki et al., 2022, "Sparse
//! Upcycling: Training Mixture-of-Experts from Dense Checkpoints", arXiv
//! 2212.05055): every expert starts as a copy of the layer's dense MLP, a new
//! router is initialized, and the whole model keeps training. With top-k
//! weights renormalized to sum to one (Mixtral-style routing), the upcycled
//! model computes exactly the dense model's function before training.
//!
//! Upcycling adds capacity; it needs continued pretraining at a scale of
//! tens of billions of tokens to pay off, so this is the most expensive
//! feature in the catalog.

use serde::Deserialize;

use super::params_of;
use crate::{
    parse_params, Compat, Confidence, Context, Detection, Effect, Estimate, Feature, FeatureError,
    Params, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
use mb_ir::{FfnSpec, TensorKind};

pub struct MoeUpcycle;

const ID: &str = "moe-upcycle";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct P {
    experts: u32,
    top_k: u32,
    /// Upcycle every `every`-th dense layer (1 = all; the paper also tried every other layer).
    every: u32,
}

impl Default for P {
    fn default() -> Self {
        Self {
            experts: 8,
            top_k: 2,
            every: 1,
        }
    }
}

fn targets(ctx: &Context, p: &P) -> Vec<u32> {
    ctx.ir
        .layers
        .iter()
        .filter(|l| matches!(l.ffn, FfnSpec::Dense { .. }))
        .map(|l| l.index)
        .enumerate()
        .filter(|(i, _)| *i as u32 % p.every.max(1) == 0)
        .map(|(_, l)| l)
        .collect()
}

const MLP: &[TensorKind] = &[
    TensorKind::FfnGate,
    TensorKind::FfnUp,
    TensorKind::FfnDown,
    TensorKind::FfnGateUp,
];

impl Feature for MoeUpcycle {
    fn id(&self) -> &'static str {
        ID
    }

    fn title(&self) -> &'static str {
        "Dense → MoE upcycling"
    }

    fn summary(&self) -> &'static str {
        "Replace dense MLPs with routed experts initialized as copies (sparse upcycling), then continue pretraining."
    }

    fn detect(&self, ctx: &Context) -> Detection {
        let moe = ctx
            .ir
            .layers
            .iter()
            .filter(|l| matches!(l.ffn, FfnSpec::Moe(_)))
            .count();
        match moe {
            0 => Detection::Absent,
            n if n == ctx.ir.layers.len() => Detection::Present(format!("all {n} layers are MoE")),
            n => Detection::Partial(format!("{n} of {} layers are MoE", ctx.ir.layers.len())),
        }
    }

    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let mut c = Compat::default();
        if targets(ctx, &p).is_empty() {
            c.blockers.push("no dense MLP layers to upcycle".into());
        }
        if p.experts < 2 {
            c.blockers.push("experts must be at least 2".into());
        }
        if p.top_k == 0 || p.top_k > p.experts {
            c.blockers
                .push("top_k must be between 1 and experts".into());
        }
        if ctx.report.quantization.trunk_bits_per_param < 4.0 {
            c.warnings.push("low-bit source: expert copies stay exact, but continued pretraining at this scale means QAT at the source format for every expert".into());
        }
        c.warnings.push("Upcycling pays off only with substantial continued pretraining; a short finetune leaves the model no better than the dense one.".into());
        Ok(c)
    }

    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let layers = targets(ctx, &p);
        let mlp = params_of(ctx.ir, &layers, MLP);
        let hidden = ctx.ir.hidden_size.unwrap_or(0);
        let router = layers.len() as u64 * hidden * u64::from(p.experts);
        let n = ctx.cost_inputs().forward_params;
        let total = n + mlp * u64::from(p.experts - 1) + router;
        let active = n + mlp * u64::from(p.top_k.saturating_sub(1)) + router;
        let effects = vec![
            Effect {
                metric: "total params".into(),
                before: n as f64,
                after: total as f64,
                unit: "params".into(),
            },
            Effect {
                metric: "active params per token".into(),
                before: n as f64,
                after: active as f64,
                unit: "params".into(),
            },
        ];
        let stages = vec![Stage {
            name: "upcycle".into(),
            what: format!(
                "continued pretraining of the whole model with {} experts (top-{}) in {} layers",
                p.experts,
                p.top_k,
                layers.len()
            ),
            trainable_params: total,
            backprop_params: active,
            tokens: Range::new(1e10, 1e11),
            seq_len: 4096,
            loss: "next-token cross-entropy plus a load-balancing loss on the routers".into(),
            data: "a pretraining mixture close to the original model's".into(),
            teacher_forward: false,
            trunk: TrunkUse::Restructure,
            precomputed_features: false,
        }];
        Ok(Estimate {
            effects,
            stages,
            risk: QualityRisk {
                level: RiskLevel::Medium,
                expected: "Early in training the experts are identical and the model behaves like the dense one; an unstable router or too few tokens can leave it worse.".into(),
                recovery: "Add noise to expert copies to break symmetry, lower the learning rate for copied weights, upcycle fewer layers.".into(),
            },
            assumptions: vec![
                "Compute is priced on active params per token (routed MLPs × top_k); memory holds every expert.".into(),
                "The token budget is a heuristic: the paper's gains are reported relative to the dense model's original budget, which is often unknown for released models.".into(),
            ],
            confidence: Confidence::Low,
            references: vec![
                "Komatsuzaki et al., 2022, Sparse Upcycling, arXiv 2212.05055 (§3 method)".into(),
                "Jiang et al., 2024, Mixtral of Experts, arXiv 2401.04088 (top-2 routing with renormalized weights)".into(),
            ],
        })
    }

    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError> {
        let p: P = parse_params(ID, params)?;
        let layers = targets(ctx, &p);
        Ok(vec![
            format!("In layers {layers:?}: ffn_{{gate,up,down}} → ffn_{{gate,up,down}}_exps with {} copies stacked along a new leading axis. Quantized blocks are copied, not re-encoded, so the copies are exact.", p.experts),
            format!("Add ffn_gate_inp [{}, hidden] (router, F32), initialized small and random.", p.experts),
            format!("Metadata: expert_count = {}, expert_used_count = {}.", p.experts, p.top_k),
            "Not implemented as surgery yet.".into(),
        ])
    }

    fn export_notes(&self, _ctx: &Context, _params: &Params) -> Result<Vec<String>, FeatureError> {
        Ok(vec![
            "llama.cpp runs MoE only in architectures whose loader expects expert tensors (e.g. llama with expert_count, qwen2moe, qwen3moe); upcycling a dense architecture may need an architecture change in the metadata.".into(),
        ])
    }
}

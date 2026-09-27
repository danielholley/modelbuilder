use mb_ir::{Component, FfnSpec, ModelIr, TensorKind};
use serde::Serialize;

/// Parameter counts by role. Counts are logical elements, so block-quantized
/// GGUF tensors count their true weights; packed safetensors (e.g. int4 in U8)
/// count storage elements; see `QuantSummary` for bytes.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ParamBreakdown {
    pub total: u64,
    pub embedding: u64,
    pub lm_head: u64,
    pub attention: u64,
    pub linear_attention: u64,
    pub ffn_dense: u64,
    pub moe_routed_experts: u64,
    pub moe_shared_experts: u64,
    pub router: u64,
    pub norms: u64,
    pub mtp: u64,
    pub multimodal: u64,
    pub other: u64,
    /// Trunk parameters used per token (routed experts scaled by top-k / experts).
    pub active_per_token: u64,
}

impl ParamBreakdown {
    pub fn of(ir: &ModelIr) -> Self {
        let mut p = Self::default();
        for (t, r) in ir.tensors() {
            let n = t.n_elements();
            p.total += n;
            let slot = match r.component {
                Component::Embedding => &mut p.embedding,
                Component::LmHead => &mut p.lm_head,
                Component::FinalNorm => &mut p.norms,
                Component::Mtp => &mut p.mtp,
                Component::Multimodal => &mut p.multimodal,
                Component::Other => &mut p.other,
                Component::Layer => match r.kind {
                    k if k.is_attention() => &mut p.attention,
                    TensorKind::LinearAttn => &mut p.linear_attention,
                    TensorKind::FfnGate
                    | TensorKind::FfnUp
                    | TensorKind::FfnDown
                    | TensorKind::FfnGateUp => &mut p.ffn_dense,
                    TensorKind::Expert => &mut p.moe_routed_experts,
                    TensorKind::SharedExpert => &mut p.moe_shared_experts,
                    TensorKind::Router => &mut p.router,
                    TensorKind::Norm => &mut p.norms,
                    _ => &mut p.other,
                },
            };
            *slot += n;
        }

        // Scale routed experts by the fraction active per token.
        let fraction = ir
            .layers
            .iter()
            .find_map(|l| match &l.ffn {
                FfnSpec::Moe(m) => Some((m.experts_per_token?, m.num_experts?)),
                _ => None,
            })
            .map_or(1.0, |(k, n)| k as f64 / n.max(1) as f64);
        let tied_head = if p.lm_head == 0 {
            p.embedding
        } else {
            p.lm_head
        };
        // Embedding lookups are nearly free; count the output head's matmul instead.
        let trunk = p.total - p.mtp - p.multimodal - p.embedding - p.lm_head - p.moe_routed_experts;
        p.active_per_token =
            trunk + tied_head + (p.moe_routed_experts as f64 * fraction).round() as u64;
        p
    }
}

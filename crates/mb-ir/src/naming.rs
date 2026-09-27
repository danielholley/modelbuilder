//! Tensor-name conventions for HF and GGUF checkpoints.
//!
//! The tensor names are the ground truth for what a checkpoint contains, so
//! the IR is built from them first and uses config values only for sizes.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Embedding,
    LmHead,
    FinalNorm,
    /// A decoder layer of the main trunk.
    Layer,
    /// Multi-token-prediction / next-n prediction modules.
    Mtp,
    /// Vision/audio towers and multimodal projectors.
    Multimodal,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    AttnQ,
    AttnK,
    AttnV,
    AttnQkv,
    AttnO,
    /// Output gate on attention (e.g. Qwen3-Next gated attention).
    AttnGate,
    /// Per-head Q/K normalization.
    QkNorm,
    MlaQA,
    MlaQB,
    MlaKvA,
    MlaKvB,
    /// Any tensor of a linear-attention / SSM mixer (DeltaNet, Mamba, RWKV).
    LinearAttn,
    Norm,
    FfnGate,
    FfnUp,
    FfnDown,
    FfnGateUp,
    Router,
    Expert,
    SharedExpert,
    Embedding,
    LmHead,
    Other,
}

impl TensorKind {
    /// Softmax-attention projections and norms (including MLA's).
    pub fn is_attention(self) -> bool {
        use TensorKind::*;
        matches!(
            self,
            AttnQ
                | AttnK
                | AttnV
                | AttnQkv
                | AttnO
                | AttnGate
                | QkNorm
                | MlaQA
                | MlaQB
                | MlaKvA
                | MlaKvB
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct TensorRole {
    pub component: Component,
    /// Layer index for [`Component::Layer`] and layered MTP modules.
    pub layer: Option<u32>,
    pub kind: TensorKind,
}

const LAYER_CONTAINERS: &[&str] = &["layers", "layer", "blk", "h", "blocks"];

fn layer_index(name: &str) -> Option<u32> {
    let mut parts = name.split('.').peekable();
    while let Some(p) = parts.next() {
        if LAYER_CONTAINERS.contains(&p) {
            if let Some(n) = parts.peek().and_then(|n| n.parse().ok()) {
                return Some(n);
            }
        }
    }
    None
}

fn is_multimodal(n: &str) -> bool {
    n.starts_with("v.")
        || n.starts_with("mm.")
        || n.starts_with("a.")
        || [
            "visual.",
            "vision",
            "multi_modal_projector",
            "mm_projector",
            "audio_tower",
            "image_newline",
        ]
        .iter()
        .any(|p| n.contains(p))
}

fn kind_of(n: &str) -> TensorKind {
    use TensorKind::*;
    let has = |pats: &[&str]| pats.iter().any(|p| n.contains(p));

    if has(&[
        "linear_attn",
        "ssm_",
        ".mixer.",
        "mamba",
        "time_mix",
        "deltanet",
    ]) {
        return LinearAttn;
    }
    if has(&["q_norm", "k_norm", "attn_q_norm", "attn_k_norm"]) {
        return QkNorm;
    }
    if has(&["norm", "ln_", "ln1", "ln2"]) {
        return Norm;
    }
    if has(&["kv_a_proj", "attn_kv_a"]) {
        return MlaKvA;
    }
    if has(&["kv_b_proj", "attn_kv_b", "attn_k_b", "attn_v_b"]) {
        return MlaKvB;
    }
    if has(&["q_a_proj", "attn_q_a"]) {
        return MlaQA;
    }
    if has(&["q_b_proj", "attn_q_b"]) {
        return MlaQB;
    }
    if has(&[
        "ffn_gate_inp",
        "router",
        "e_score_correction",
        "exp_probs_b",
    ]) || n.ends_with("mlp.gate.weight")
        || n.ends_with("block_sparse_moe.gate.weight")
    {
        return Router;
    }
    if has(&["shared_expert", "_shexp", "shared_mlp"]) {
        return SharedExpert;
    }
    if has(&[".experts.", "_exps"]) {
        return Expert;
    }
    if has(&[
        "qkv_proj",
        "query_key_value",
        "attn_qkv",
        "c_attn",
        "w_pack",
    ]) {
        return AttnQkv;
    }
    if has(&["q_proj", "attn_q."]) {
        return AttnQ;
    }
    if has(&["k_proj", "attn_k."]) {
        return AttnK;
    }
    if has(&["v_proj", "attn_v."]) {
        return AttnV;
    }
    if has(&["o_proj", "attn_output", "attention.dense", "attn.c_proj"]) {
        return AttnO;
    }
    if has(&["attn_gate", "g_proj"]) {
        return AttnGate;
    }
    if has(&["gate_up_proj"]) {
        return FfnGateUp;
    }
    if has(&["gate_proj", "ffn_gate.", ".w1."]) {
        return FfnGate;
    }
    if has(&["up_proj", "ffn_up.", ".w3.", "c_fc"]) {
        return FfnUp;
    }
    if has(&["down_proj", "ffn_down.", ".w2.", "mlp.c_proj"]) {
        return FfnDown;
    }
    Other
}

/// Classifies a tensor by name. `num_layers` (from config) lets DeepSeek-style
/// checkpoints, which store MTP modules as extra trailing layers, be recognized.
pub fn classify(name: &str, num_layers: Option<u64>) -> TensorRole {
    let n = name.to_ascii_lowercase();
    let layer = layer_index(&n);

    if is_multimodal(&n) {
        return TensorRole {
            component: Component::Multimodal,
            layer: None,
            kind: TensorKind::Other,
        };
    }

    let beyond_trunk = matches!((layer, num_layers), (Some(l), Some(nl)) if u64::from(l) >= nl);
    if n.starts_with("mtp.") || n.contains(".mtp.") || n.contains("nextn") || beyond_trunk {
        return TensorRole {
            component: Component::Mtp,
            layer,
            kind: kind_of(&n),
        };
    }

    if let Some(l) = layer {
        return TensorRole {
            component: Component::Layer,
            layer: Some(l),
            kind: kind_of(&n),
        };
    }

    let (component, kind) = if [
        "embed_tokens",
        "token_embd",
        "wte",
        "word_embeddings",
        "tok_embeddings",
    ]
    .iter()
    .any(|p| n.contains(p))
    {
        (Component::Embedding, TensorKind::Embedding)
    } else if n.starts_with("lm_head") || n == "output.weight" || n.starts_with("embed_out") {
        (Component::LmHead, TensorKind::LmHead)
    } else if n.contains("norm") || n.starts_with("ln_f") {
        (Component::FinalNorm, TensorKind::Norm)
    } else {
        (Component::Other, TensorKind::Other)
    };
    TensorRole {
        component,
        layer: None,
        kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Component as C;
    use TensorKind as K;

    fn check(name: &str, component: Component, layer: Option<u32>, kind: TensorKind) {
        assert_eq!(
            classify(name, Some(4)),
            TensorRole {
                component,
                layer,
                kind
            },
            "{name}"
        );
    }

    #[test]
    fn hf_names() {
        check(
            "model.embed_tokens.weight",
            C::Embedding,
            None,
            K::Embedding,
        );
        check("lm_head.weight", C::LmHead, None, K::LmHead);
        check("model.norm.weight", C::FinalNorm, None, K::Norm);
        check(
            "model.layers.0.self_attn.q_proj.weight",
            C::Layer,
            Some(0),
            K::AttnQ,
        );
        check(
            "model.layers.0.self_attn.k_norm.weight",
            C::Layer,
            Some(0),
            K::QkNorm,
        );
        check(
            "model.layers.1.linear_attn.in_proj_qkvz.weight",
            C::Layer,
            Some(1),
            K::LinearAttn,
        );
        check(
            "model.layers.2.mlp.gate.weight",
            C::Layer,
            Some(2),
            K::Router,
        );
        check(
            "model.layers.2.mlp.gate_proj.weight",
            C::Layer,
            Some(2),
            K::FfnGate,
        );
        check(
            "model.layers.2.mlp.experts.7.down_proj.weight",
            C::Layer,
            Some(2),
            K::Expert,
        );
        check(
            "model.layers.2.mlp.shared_experts.up_proj.weight",
            C::Layer,
            Some(2),
            K::SharedExpert,
        );
        check(
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            C::Layer,
            Some(3),
            K::MlaKvA,
        );
        check(
            "model.layers.3.self_attn.kv_a_layernorm.weight",
            C::Layer,
            Some(3),
            K::Norm,
        );
        check(
            "model.language_model.layers.3.mlp.down_proj.weight",
            C::Layer,
            Some(3),
            K::FfnDown,
        );
    }

    #[test]
    fn mtp_and_multimodal() {
        check(
            "mtp.layers.0.self_attn.q_proj.weight",
            C::Mtp,
            Some(0),
            K::AttnQ,
        );
        check("model.layers.4.eh_proj.weight", C::Mtp, Some(4), K::Other);
        check(
            "model.visual.blocks.0.attn.qkv.weight",
            C::Multimodal,
            None,
            K::Other,
        );
        check("v.blk.0.attn_q.weight", C::Multimodal, None, K::Other);
    }

    #[test]
    fn gguf_names() {
        check("token_embd.weight", C::Embedding, None, K::Embedding);
        check("output.weight", C::LmHead, None, K::LmHead);
        check("output_norm.weight", C::FinalNorm, None, K::Norm);
        check("blk.0.attn_q.weight", C::Layer, Some(0), K::AttnQ);
        check("blk.0.attn_q_norm.weight", C::Layer, Some(0), K::QkNorm);
        check("blk.1.ssm_conv1d.weight", C::Layer, Some(1), K::LinearAttn);
        check("blk.2.ffn_gate_inp.weight", C::Layer, Some(2), K::Router);
        check("blk.2.ffn_gate_exps.weight", C::Layer, Some(2), K::Expert);
        check(
            "blk.2.ffn_down_shexp.weight",
            C::Layer,
            Some(2),
            K::SharedExpert,
        );
        check("blk.2.ffn_gate.weight", C::Layer, Some(2), K::FfnGate);
        check("blk.3.attn_kv_a_mqa.weight", C::Layer, Some(3), K::MlaKvA);
        check("blk.3.nextn.eh_proj.weight", C::Mtp, Some(3), K::Other);
    }
}

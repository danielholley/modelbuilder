mod draft;
mod fp4_kv;
mod kv_share;
mod mla;
mod moe_upcycle;
mod mtp;
mod prune;
mod yarn;

use mb_ir::{AttentionKind, AttentionSpec, Component, Mixer, ModelIr, TensorKind};

use crate::Feature;

static CATALOG: &[&dyn Feature] = &[
    &fp4_kv::Fp4Kv,
    &kv_share::KvShare,
    &mla::Mla,
    &mtp::Mtp,
    &draft::DraftHead,
    &yarn::Yarn,
    &prune::Prune,
    &moe_upcycle::MoeUpcycle,
];

/// Every built-in feature.
pub fn catalog() -> &'static [&'static dyn Feature] {
    CATALOG
}

/// A full-attention layer with a per-token (non-windowed, non-MLA) KV cache.
pub(crate) struct GlobalAttn<'a> {
    pub layer: u32,
    pub spec: &'a AttentionSpec,
}

pub(crate) fn global_attention_layers(ir: &ModelIr) -> Vec<GlobalAttn<'_>> {
    ir.layers
        .iter()
        .filter_map(|l| match &l.mixer {
            Mixer::Attention(a) if a.sliding_window.is_none() && a.kind != AttentionKind::Mla => {
                Some(GlobalAttn {
                    layer: l.index,
                    spec: a,
                })
            }
            _ => None,
        })
        .collect()
}

/// Parameters of the trunk layer tensors matching `kinds` in `layers`.
pub(crate) fn params_of(ir: &ModelIr, layers: &[u32], kinds: &[TensorKind]) -> u64 {
    ir.tensors()
        .filter(|(_, r)| {
            r.component == Component::Layer
                && r.layer.is_some_and(|l| layers.contains(&l))
                && kinds.contains(&r.kind)
        })
        .map(|(t, _)| t.n_elements())
        .sum()
}

/// Parameters activation gradients must flow through when the lowest
/// trainable layer is `from`: every trunk layer at or above it plus the
/// output head and final norm.
pub(crate) fn backprop_params_from(ir: &ModelIr, from: u32) -> u64 {
    ir.tensors()
        .filter(|(_, r)| match r.component {
            Component::Layer => r.layer.is_some_and(|l| l >= from),
            Component::LmHead | Component::FinalNorm => true,
            _ => false,
        })
        .map(|(t, _)| t.n_elements())
        .sum()
}

pub(crate) fn gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    classify, Component, ConfigView, Key, RawModel, SourceFormat, TensorInfo, TensorKind,
    TensorRole, WeightRotation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    /// Multi-head attention: one KV head per query head.
    Mha,
    /// Grouped-query attention.
    Gqa,
    /// Multi-query attention: a single KV head.
    Mqa,
    /// Multi-head latent attention (DeepSeek-V2/V3 style).
    Mla,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct MlaSpec {
    pub kv_lora_rank: Option<u64>,
    pub q_lora_rank: Option<u64>,
    pub qk_rope_head_dim: Option<u64>,
    pub qk_nope_head_dim: Option<u64>,
    pub v_head_dim: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct AttentionSpec {
    pub kind: AttentionKind,
    pub num_heads: Option<u64>,
    pub num_kv_heads: Option<u64>,
    pub head_dim: Option<u64>,
    /// Sigmoid output gate on attention (Qwen3-Next style).
    pub output_gate: bool,
    pub qk_norm: bool,
    /// Sliding-window size, if this layer attends locally.
    pub sliding_window: Option<u64>,
    pub mla: Option<MlaSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct LinearAttentionSpec {
    /// e.g. `gated_deltanet`, `mamba`, `rwkv`, or `unknown`.
    pub variant: String,
    pub num_key_heads: Option<u64>,
    pub num_value_heads: Option<u64>,
    pub key_head_dim: Option<u64>,
    pub value_head_dim: Option<u64>,
    pub conv_kernel: Option<u64>,
}

/// The token-mixing block of a layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Mixer {
    Attention(AttentionSpec),
    LinearAttention(LinearAttentionSpec),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct MoeSpec {
    pub num_experts: Option<u64>,
    pub experts_per_token: Option<u64>,
    pub num_shared_experts: u64,
    pub expert_intermediate_size: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FfnSpec {
    Dense { intermediate_size: Option<u64> },
    Moe(MoeSpec),
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Layer {
    pub index: u32,
    pub mixer: Mixer,
    pub ffn: FfnSpec,
    pub params: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct RopeInfo {
    pub theta: Option<f64>,
    pub scaling: Option<Value>,
}

/// True unless the RoPE config is just the default (no actual scaling).
fn is_real_rope_scaling(v: &Value) -> bool {
    let kind = v
        .get("rope_type")
        .or_else(|| v.get("type"))
        .and_then(Value::as_str);
    kind.is_some_and(|k| k != "default") || v.get("factor").is_some()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct MtpInfo {
    pub num_modules: u64,
    pub tensor_count: usize,
    pub params: u64,
}

/// Normalized, format-independent view of a model checkpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct ModelIr {
    pub family: Option<String>,
    pub architectures: Vec<String>,
    pub hidden_size: Option<u64>,
    pub vocab_size: Option<u64>,
    pub max_positions: Option<u64>,
    pub tie_word_embeddings: Option<bool>,
    pub rope: RopeInfo,
    pub layers: Vec<Layer>,
    pub mtp: Option<MtpInfo>,
    pub has_multimodal: bool,
    pub weight_rotation: Option<WeightRotation>,
    /// HF `quantization_config`, if the checkpoint declares one.
    pub quantization_config: Option<Value>,
    /// Classification of each tensor, parallel to `raw.tensors`.
    pub roles: Vec<TensorRole>,
    /// Inconsistencies found while normalizing (config vs. tensors, etc.).
    pub warnings: Vec<String>,
    pub raw: RawModel,
}

impl ModelIr {
    pub fn from_raw(raw: RawModel) -> Self {
        let cfg = ConfigView::new(&raw.metadata);
        let mut warnings = Vec::new();

        let mtp_layers_cfg = cfg.u64(Key::MtpLayers).filter(|&n| n > 0);
        let trunk_layers_cfg =
            cfg.u64(Key::NumLayers)
                .map(|n| match (raw.format, mtp_layers_cfg) {
                    // llama.cpp counts next-n layers in block_count.
                    (SourceFormat::Gguf, Some(m)) if m < n => n - m,
                    _ => n,
                });

        let mut roles: Vec<TensorRole> = raw
            .tensors
            .iter()
            .map(|t| classify(&t.name, trunk_layers_cfg))
            .collect();

        let mut by_layer: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, r) in roles.iter().enumerate() {
            if let (Component::Layer, Some(l)) = (r.component, r.layer) {
                by_layer.entry(l).or_default().push(i);
            }
        }
        let num_layers = trunk_layers_cfg
            .map(|n| n as u32)
            .or_else(|| by_layer.keys().last().map(|l| l + 1))
            .unwrap_or(0);
        if let Some(n) = trunk_layers_cfg {
            let found = by_layer.len() as u64;
            if found != n {
                warnings.push(format!(
                    "config declares {n} layers but tensors cover {found}"
                ));
            }
        }

        let family = cfg.family();
        let hidden_size = cfg.u64(Key::HiddenSize);
        let layer_types = cfg.layer_types();
        let empty = Vec::new();
        let layers: Vec<Layer> = (0..num_layers)
            .map(|l| {
                let idx = by_layer.get(&l).unwrap_or(&empty);
                let tensors: Vec<(&TensorInfo, &TensorRole)> =
                    idx.iter().map(|&i| (&raw.tensors[i], &roles[i])).collect();
                let lt = layer_types
                    .as_ref()
                    .and_then(|v| v.get(l as usize))
                    .map(String::as_str);
                Layer {
                    index: l,
                    mixer: detect_mixer(
                        &cfg,
                        family.as_deref(),
                        hidden_size,
                        &tensors,
                        lt,
                        &mut warnings,
                    ),
                    ffn: detect_ffn(&cfg, &tensors),
                    params: tensors.iter().map(|(t, _)| t.n_elements()).sum(),
                }
            })
            .collect();

        // Linear-attention layers reuse attention-like names in GGUF (`attn_qkv`,
        // `attn_gate`); they belong to the linear mixer, not to softmax attention.
        for layer in &layers {
            if matches!(layer.mixer, Mixer::LinearAttention(_)) {
                for &i in by_layer.get(&layer.index).unwrap_or(&empty) {
                    if roles[i].kind.is_attention() {
                        roles[i].kind = TensorKind::LinearAttn;
                    }
                }
            }
        }

        let mtp_tensors: Vec<usize> = (0..roles.len())
            .filter(|&i| roles[i].component == Component::Mtp)
            .collect();
        let mtp = (!mtp_tensors.is_empty()).then(|| {
            let distinct: BTreeSet<Option<u32>> =
                mtp_tensors.iter().map(|&i| roles[i].layer).collect();
            MtpInfo {
                num_modules: mtp_layers_cfg.unwrap_or(distinct.len() as u64),
                tensor_count: mtp_tensors.len(),
                params: mtp_tensors
                    .iter()
                    .map(|&i| raw.tensors[i].n_elements())
                    .sum(),
            }
        });
        if mtp.is_none() && mtp_layers_cfg.is_some() {
            warnings
                .push("config declares MTP layers but the checkpoint has no MTP tensors".into());
        }

        let embed_rows = raw
            .tensors
            .iter()
            .zip(&roles)
            .find(|(_, r)| r.component == Component::Embedding)
            .and_then(|(t, _)| t.shape.first().copied());
        let has_lm_head = roles.iter().any(|r| r.component == Component::LmHead);

        Self {
            architectures: cfg.architectures(),
            hidden_size,
            vocab_size: cfg.u64(Key::VocabSize).or(embed_rows),
            max_positions: cfg.u64(Key::MaxPositions),
            tie_word_embeddings: cfg.bool(Key::TieWordEmbeddings).or(Some(!has_lm_head)),
            rope: RopeInfo {
                theta: cfg.f64(Key::RopeTheta),
                scaling: cfg.rope_scaling().filter(is_real_rope_scaling),
            },
            weight_rotation: WeightRotation::detect(&cfg),
            layers,
            mtp,
            has_multimodal: roles.iter().any(|r| r.component == Component::Multimodal),
            quantization_config: cfg.quantization_config().cloned(),
            family,
            roles,
            warnings,
            raw,
        }
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&TensorInfo, &TensorRole)> {
        self.raw.tensors.iter().zip(&self.roles)
    }

    pub fn total_params(&self) -> u64 {
        self.raw.tensors.iter().map(TensorInfo::n_elements).sum()
    }

    pub fn attention_layers(&self) -> impl Iterator<Item = (&Layer, &AttentionSpec)> {
        self.layers.iter().filter_map(|l| match &l.mixer {
            Mixer::Attention(a) => Some((l, a)),
            _ => None,
        })
    }
}

fn has(tensors: &[(&TensorInfo, &TensorRole)], kinds: &[TensorKind]) -> bool {
    tensors.iter().any(|(_, r)| kinds.contains(&r.kind))
}

fn find<'a>(tensors: &[(&'a TensorInfo, &TensorRole)], kind: TensorKind) -> Option<&'a TensorInfo> {
    tensors
        .iter()
        .find(|(t, r)| r.kind == kind && t.name.ends_with("weight"))
        .map(|(t, _)| *t)
}

fn detect_mixer(
    cfg: &ConfigView,
    family: Option<&str>,
    hidden: Option<u64>,
    tensors: &[(&TensorInfo, &TensorRole)],
    layer_type: Option<&str>,
    warnings: &mut Vec<String>,
) -> Mixer {
    use TensorKind::*;
    let is_linear = has(tensors, &[LinearAttn])
        || (tensors.is_empty() && layer_type.is_some_and(|t| t.contains("linear")));
    if is_linear {
        return Mixer::LinearAttention(linear_spec(cfg, family, tensors));
    }

    let is_mla = has(tensors, &[MlaKvA, MlaKvB]);
    let is_attn = is_mla
        || has(tensors, &[AttnQ, AttnK, AttnV, AttnQkv])
        || (tensors.is_empty() && layer_type.is_some_and(|t| t.contains("attention")));
    if !is_attn {
        return Mixer::Unknown;
    }

    let num_heads = cfg.u64(Key::NumHeads);
    let head_dim = cfg
        .u64(Key::HeadDim)
        .or_else(|| Some(hidden? / num_heads.filter(|&h| h > 0)?));
    let mut num_kv_heads = cfg.u64(Key::NumKvHeads).or(num_heads);

    // Cross-check KV heads against the K projection's output rows.
    if let (Some(k), Some(hd), false) = (find(tensors, AttnK), head_dim, is_mla) {
        if k.shape.len() == 2 {
            let rows = k.shape[0];
            if hd > 0 && rows % hd == 0 && Some(rows / hd) != num_kv_heads {
                warnings.push(format!(
                    "{}: {} KV heads implied by shape, config says {:?}",
                    k.name,
                    rows / hd,
                    num_kv_heads
                ));
                num_kv_heads = Some(rows / hd);
            }
        }
    }

    let q_rows = find(tensors, AttnQ).map(|q| q.shape[0]);
    let output_gate = has(tensors, &[AttnGate])
        || matches!((q_rows, num_heads, head_dim), (Some(r), Some(h), Some(d)) if r == 2 * h * d);

    let sliding_window = match layer_type {
        Some(t) => t
            .contains("sliding")
            .then(|| cfg.u64(Key::SlidingWindow))
            .flatten(),
        None => match cfg.hf_value("use_sliding_window").and_then(Value::as_bool) {
            Some(false) => None,
            _ => cfg.u64(Key::SlidingWindow),
        },
    };

    let mla = is_mla.then(|| MlaSpec {
        kv_lora_rank: cfg.u64(Key::KvLoraRank),
        q_lora_rank: cfg.u64(Key::QLoraRank),
        qk_rope_head_dim: cfg.u64(Key::QkRopeHeadDim),
        qk_nope_head_dim: cfg.u64(Key::QkNopeHeadDim),
        v_head_dim: cfg.u64(Key::VHeadDim),
    });

    let kind = match (is_mla, num_heads, num_kv_heads) {
        (true, _, _) => AttentionKind::Mla,
        (_, Some(h), Some(kv)) if kv == h => AttentionKind::Mha,
        (_, Some(h), Some(1)) if h > 1 => AttentionKind::Mqa,
        (_, Some(_), Some(_)) => AttentionKind::Gqa,
        _ => AttentionKind::Mha,
    };

    Mixer::Attention(AttentionSpec {
        kind,
        num_heads,
        num_kv_heads,
        head_dim,
        output_gate,
        qk_norm: has(tensors, &[QkNorm]),
        sliding_window,
        mla,
    })
}

fn linear_spec(
    cfg: &ConfigView,
    family: Option<&str>,
    tensors: &[(&TensorInfo, &TensorRole)],
) -> LinearAttentionSpec {
    let names = |pat: &str| tensors.iter().any(|(t, _)| t.name.contains(pat));
    let fam = family.unwrap_or_default();
    // Qwen3-Next fuses projections (`in_proj_qkvz`); Qwen3.5+ splits them
    // (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`); llama.cpp names
    // them `ssm_alpha`/`ssm_beta`.
    let deltanet_family = ["qwen3_next", "qwen3next", "qwen3_5", "qwen35"]
        .iter()
        .any(|f| fam.contains(f));
    let variant = if deltanet_family
        || [
            "in_proj_qkvz",
            "in_proj_ba",
            "in_proj_z",
            "ssm_alpha",
            "ssm_beta",
        ]
        .iter()
        .any(|n| names(n))
        || (names("linear_attn") && cfg.u64(Key::LinearNumValueHeads).is_some())
    {
        "gated_deltanet"
    } else if names("mamba") || names("mixer.") || fam.contains("mamba") {
        "mamba"
    } else if names("time_mix") || fam.contains("rwkv") {
        "rwkv"
    } else {
        "unknown"
    };
    LinearAttentionSpec {
        variant: variant.into(),
        num_key_heads: cfg.u64(Key::LinearNumKeyHeads),
        num_value_heads: cfg.u64(Key::LinearNumValueHeads),
        key_head_dim: cfg.u64(Key::LinearKeyHeadDim),
        // GGUF has no value-dim key; derive it from `ssm.inner_size / v_heads`.
        value_head_dim: cfg.u64(Key::LinearValueHeadDim).or_else(|| {
            let inner = cfg.gguf_arch_value("ssm.inner_size")?.as_u64()?;
            Some(inner / cfg.u64(Key::LinearNumValueHeads).filter(|&v| v > 0)?)
        }),
        conv_kernel: cfg.u64(Key::LinearConvKernel),
    }
}

fn detect_ffn(cfg: &ConfigView, tensors: &[(&TensorInfo, &TensorRole)]) -> FfnSpec {
    use TensorKind::*;
    if has(tensors, &[Expert]) {
        let inferred_experts = {
            // HF stores experts as `...experts.{j}...`; GGUF packs them as a leading dim.
            let hf: BTreeSet<u64> = tensors
                .iter()
                .filter(|(_, r)| r.kind == Expert)
                .filter_map(|(t, _)| {
                    let mut parts = t.name.split('.');
                    parts.find(|p| *p == "experts")?;
                    parts.next()?.parse().ok()
                })
                .collect();
            if hf.is_empty() {
                find(tensors, Expert)
                    .filter(|t| t.shape.len() == 3)
                    .map(|t| t.shape[0])
            } else {
                Some(hf.len() as u64)
            }
        };
        return FfnSpec::Moe(MoeSpec {
            num_experts: cfg.u64(Key::NumExperts).or(inferred_experts),
            experts_per_token: cfg.u64(Key::ExpertsPerToken),
            num_shared_experts: if has(tensors, &[SharedExpert]) {
                cfg.u64(Key::NumSharedExperts).unwrap_or(1)
            } else {
                0
            },
            expert_intermediate_size: cfg.u64(Key::MoeIntermediateSize),
        });
    }
    if has(tensors, &[FfnGate, FfnUp, FfnDown, FfnGateUp]) {
        let from_shape = find(tensors, FfnDown).and_then(|t| t.shape.get(1).copied());
        return FfnSpec::Dense {
            intermediate_size: cfg.u64(Key::IntermediateSize).or(from_shape),
        };
    }
    FfnSpec::None
}

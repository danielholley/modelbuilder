use serde_json::Value;

#[cfg(test)]
use crate::MetaType;
use crate::{MetaValue, Metadata};

/// Canonical config keys, mapped onto both HF `config.json` names and GGUF keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    NumLayers,
    HiddenSize,
    IntermediateSize,
    NumHeads,
    NumKvHeads,
    HeadDim,
    VocabSize,
    MaxPositions,
    RopeTheta,
    SlidingWindow,
    TieWordEmbeddings,
    NumExperts,
    ExpertsPerToken,
    NumSharedExperts,
    MoeIntermediateSize,
    SharedExpertIntermediateSize,
    KvLoraRank,
    QLoraRank,
    QkRopeHeadDim,
    QkNopeHeadDim,
    VHeadDim,
    MtpLayers,
    FullAttentionInterval,
    LinearNumKeyHeads,
    LinearNumValueHeads,
    LinearKeyHeadDim,
    LinearValueHeadDim,
    LinearConvKernel,
}

impl Key {
    /// Candidate names in HF `config.json` (searched in `text_config` first, then the root).
    fn hf(self) -> &'static [&'static str] {
        match self {
            Key::NumLayers => &["num_hidden_layers", "n_layer", "num_layers", "n_layers"],
            Key::HiddenSize => &["hidden_size", "d_model", "n_embd"],
            Key::IntermediateSize => &["intermediate_size", "ffn_hidden_size", "n_inner"],
            Key::NumHeads => &["num_attention_heads", "n_head", "num_heads"],
            Key::NumKvHeads => &[
                "num_key_value_heads",
                "num_kv_heads",
                "multi_query_group_num",
            ],
            Key::HeadDim => &["head_dim", "kv_channels"],
            Key::VocabSize => &["vocab_size", "padded_vocab_size"],
            Key::MaxPositions => &["max_position_embeddings", "n_positions", "seq_length"],
            Key::RopeTheta => &["rope_theta", "rotary_emb_base"],
            Key::SlidingWindow => &["sliding_window"],
            Key::TieWordEmbeddings => &["tie_word_embeddings"],
            Key::NumExperts => &[
                "num_experts",
                "n_routed_experts",
                "num_local_experts",
                "moe_num_experts",
            ],
            Key::ExpertsPerToken => &["num_experts_per_tok", "moe_top_k", "num_experts_per_token"],
            Key::NumSharedExperts => &["n_shared_experts", "num_shared_experts"],
            Key::MoeIntermediateSize => &["moe_intermediate_size", "expert_intermediate_size"],
            Key::SharedExpertIntermediateSize => &["shared_expert_intermediate_size"],
            Key::KvLoraRank => &["kv_lora_rank"],
            Key::QLoraRank => &["q_lora_rank"],
            Key::QkRopeHeadDim => &["qk_rope_head_dim"],
            Key::QkNopeHeadDim => &["qk_nope_head_dim"],
            Key::VHeadDim => &["v_head_dim"],
            Key::MtpLayers => &[
                "num_nextn_predict_layers",
                "mtp_num_hidden_layers",
                "num_mtp_layers",
            ],
            Key::FullAttentionInterval => &["full_attention_interval"],
            Key::LinearNumKeyHeads => &["linear_num_key_heads"],
            Key::LinearNumValueHeads => &["linear_num_value_heads"],
            Key::LinearKeyHeadDim => &["linear_key_head_dim"],
            Key::LinearValueHeadDim => &["linear_value_head_dim"],
            Key::LinearConvKernel => &["linear_conv_kernel_dim"],
        }
    }

    /// Candidate GGUF key suffixes, appended to `"{general.architecture}."`.
    fn gguf(self) -> &'static [&'static str] {
        match self {
            Key::NumLayers => &["block_count"],
            Key::HiddenSize => &["embedding_length"],
            Key::IntermediateSize => &["feed_forward_length"],
            Key::NumHeads => &["attention.head_count"],
            Key::NumKvHeads => &["attention.head_count_kv"],
            Key::HeadDim => &["attention.key_length"],
            Key::VocabSize => &["vocab_size"],
            Key::MaxPositions => &["context_length"],
            Key::RopeTheta => &["rope.freq_base"],
            Key::SlidingWindow => &["attention.sliding_window"],
            Key::TieWordEmbeddings => &[],
            Key::NumExperts => &["expert_count"],
            Key::ExpertsPerToken => &["expert_used_count"],
            Key::NumSharedExperts => &["expert_shared_count"],
            Key::MoeIntermediateSize => &["expert_feed_forward_length"],
            Key::SharedExpertIntermediateSize => &["expert_shared_feed_forward_length"],
            Key::KvLoraRank => &["attention.kv_lora_rank"],
            Key::QLoraRank => &["attention.q_lora_rank"],
            Key::QkRopeHeadDim => &["rope.dimension_count"],
            Key::QkNopeHeadDim => &[],
            Key::VHeadDim => &["attention.value_length"],
            Key::MtpLayers => &["nextn_predict_layers"],
            Key::FullAttentionInterval => &["full_attention_interval"],
            Key::LinearNumKeyHeads => &["ssm.group_count"],
            Key::LinearNumValueHeads => &["ssm.time_step_rank"],
            Key::LinearKeyHeadDim => &["ssm.state_size"],
            Key::LinearValueHeadDim => &[],
            Key::LinearConvKernel => &["ssm.conv_kernel"],
        }
    }
}

/// Uniform read access to model config, whichever format it came from.
pub struct ConfigView<'a> {
    meta: &'a Metadata,
    gguf_arch: Option<String>,
}

impl<'a> ConfigView<'a> {
    pub fn new(meta: &'a Metadata) -> Self {
        let mut view = Self {
            meta,
            gguf_arch: None,
        };
        view.gguf_arch = view
            .gguf_raw("general.architecture")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        view
    }

    /// Architecture family: HF `model_type` (text model's, if nested) or GGUF `general.architecture`.
    pub fn family(&self) -> Option<String> {
        match self.meta {
            Metadata::Hf { .. } => self
                .hf_value("model_type")
                .and_then(Value::as_str)
                .map(str::to_owned),
            Metadata::Gguf { .. } => self.gguf_arch.clone(),
        }
    }

    /// HF `architectures` class names, or the GGUF architecture.
    pub fn architectures(&self) -> Vec<String> {
        match self.meta {
            Metadata::Hf { config, .. } => config
                .get("architectures")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default(),
            Metadata::Gguf { .. } => self.gguf_arch.iter().cloned().collect(),
        }
    }

    /// Looks up `name` in `text_config` first, then at the root of `config.json`.
    pub fn hf_value(&self, name: &str) -> Option<&'a Value> {
        let Metadata::Hf { config, .. } = self.meta else {
            return None;
        };
        for scope in ["text_config", "language_config", "llm_config"] {
            if let Some(v) = config.get(scope).and_then(|t| t.get(name)) {
                return Some(v);
            }
        }
        config.get(name)
    }

    /// A GGUF key by its full name.
    pub fn gguf_raw(&self, key: &str) -> Option<&'a MetaValue> {
        let Metadata::Gguf { kv, .. } = self.meta else {
            return None;
        };
        kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A GGUF key relative to the architecture prefix (`"{arch}.{suffix}"`).
    pub fn gguf_arch_value(&self, suffix: &str) -> Option<&'a MetaValue> {
        let arch = self.gguf_arch.as_deref()?;
        self.gguf_raw(&format!("{arch}.{suffix}"))
    }

    fn hf_first(&self, key: Key) -> Option<&'a Value> {
        let found = key.hf().iter().find_map(|n| self.hf_value(n));
        match (found, key) {
            (None, Key::RopeTheta) => ["rope_parameters", "rope_scaling"]
                .iter()
                .find_map(|n| self.hf_value(n)?.get("rope_theta")),
            (v, _) => v,
        }
    }

    fn gguf_first(&self, key: Key) -> Option<&'a MetaValue> {
        key.gguf().iter().find_map(|s| self.gguf_arch_value(s))
    }

    pub fn u64(&self, key: Key) -> Option<u64> {
        match self.meta {
            Metadata::Hf { .. } => self.hf_first(key)?.as_u64(),
            Metadata::Gguf { .. } => {
                let v = self.gguf_first(key)?;
                // Some GGUF keys are per-layer arrays; use the first value.
                v.as_u64().or_else(|| v.as_array()?.first()?.as_u64())
            }
        }
    }

    pub fn f64(&self, key: Key) -> Option<f64> {
        match self.meta {
            Metadata::Hf { .. } => self.hf_first(key)?.as_f64(),
            Metadata::Gguf { .. } => self.gguf_first(key)?.as_f64(),
        }
    }

    pub fn bool(&self, key: Key) -> Option<bool> {
        match self.meta {
            Metadata::Hf { .. } => self.hf_first(key)?.as_bool(),
            Metadata::Gguf { .. } => self.gguf_first(key)?.as_bool(),
        }
    }

    /// Per-layer type labels (`layer_types` in newer HF configs), if present.
    pub fn layer_types(&self) -> Option<Vec<String>> {
        let arr = self.hf_value("layer_types")?.as_array()?;
        arr.iter().map(|v| v.as_str().map(str::to_owned)).collect()
    }

    /// Raw RoPE scaling config, if any.
    pub fn rope_scaling(&self) -> Option<Value> {
        match self.meta {
            Metadata::Hf { .. } => ["rope_scaling", "rope_parameters"]
                .iter()
                .find_map(|n| self.hf_value(n))
                .filter(|v| !v.is_null())
                .cloned(),
            Metadata::Gguf { .. } => {
                let kind = self.gguf_arch_value("rope.scaling.type")?.as_str()?;
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), kind.into());
                if let Some(f) = self
                    .gguf_arch_value("rope.scaling.factor")
                    .and_then(MetaValue::as_f64)
                {
                    obj.insert("factor".into(), f.into());
                }
                if let Some(n) = self
                    .gguf_arch_value("rope.scaling.original_context_length")
                    .and_then(MetaValue::as_u64)
                {
                    obj.insert("original_max_position_embeddings".into(), n.into());
                }
                Some(Value::Object(obj))
            }
        }
    }

    /// `quantization_config` from HF configs.
    pub fn quantization_config(&self) -> Option<&'a Value> {
        self.hf_value("quantization_config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hf_prefers_text_config() {
        let meta = Metadata::Hf {
            config: json!({
                "model_type": "wrapper_vl",
                "hidden_size": 1,
                "text_config": {"model_type": "inner", "hidden_size": 64, "num_attention_heads": 4}
            }),
            safetensors_metadata: Default::default(),
        };
        let v = ConfigView::new(&meta);
        assert_eq!(v.family().as_deref(), Some("inner"));
        assert_eq!(v.u64(Key::HiddenSize), Some(64));
        assert_eq!(v.u64(Key::NumHeads), Some(4));
    }

    #[test]
    fn rope_theta_falls_back_to_rope_parameters() {
        let meta = Metadata::Hf {
            config: json!({"rope_parameters": {"rope_theta": 1.0e7, "rope_type": "default"}}),
            safetensors_metadata: Default::default(),
        };
        assert_eq!(ConfigView::new(&meta).f64(Key::RopeTheta), Some(1.0e7));
    }

    #[test]
    fn gguf_keys_use_arch_prefix() {
        let meta = Metadata::Gguf {
            version: 3,
            kv: vec![
                (
                    "general.architecture".into(),
                    MetaValue::String("llama".into()),
                ),
                ("llama.block_count".into(), MetaValue::U32(2)),
                (
                    "llama.attention.head_count_kv".into(),
                    MetaValue::Array {
                        elem: MetaType::U32,
                        values: vec![MetaValue::U32(2), MetaValue::U32(2)],
                    },
                ),
            ],
        };
        let v = ConfigView::new(&meta);
        assert_eq!(v.family().as_deref(), Some("llama"));
        assert_eq!(v.u64(Key::NumLayers), Some(2));
        assert_eq!(v.u64(Key::NumKvHeads), Some(2));
    }
}

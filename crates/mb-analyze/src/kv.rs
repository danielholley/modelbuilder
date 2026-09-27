use mb_ir::{AttentionKind, Mixer, ModelIr};
use serde::Serialize;

/// KV cache storage precisions to report. Bits per element include scales.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct KvPrecision {
    pub name: &'static str,
    pub bits_per_element: f64,
    /// KV bytes per token across all unbounded (global) attention layers.
    pub bytes_per_token: f64,
    /// Total KV bytes for one sequence at `context` tokens, including sliding-window layers.
    pub bytes_at_context: f64,
}

const PRECISIONS: &[(&str, f64)] = &[
    ("bf16", 16.0),
    ("fp8", 8.0),
    // E2M1 with one 8-bit scale per 16 channels (DeepSeek-V4.1-Flash style).
    ("fp4_e2m1_g16", 4.0 + 8.0 / 16.0),
];

/// Estimated per-sequence inference memory for attention state.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct KvCacheEstimate {
    /// Attention layers whose cache grows with the sequence.
    pub global_layers: usize,
    /// Attention layers bounded by a sliding window.
    pub windowed_layers: usize,
    /// Linear-attention / SSM layers (fixed-size state, no per-token cache).
    pub linear_layers: usize,
    /// Cached elements per token across global layers.
    pub elements_per_token: u64,
    pub context: Option<u64>,
    pub precisions: Vec<KvPrecision>,
    /// Fixed recurrent state of linear layers per sequence, in bytes (fp32 state, bf16 conv).
    pub linear_state_bytes: Option<u64>,
    /// Layers skipped because their dimensions are unknown.
    pub unknown_layers: Vec<u32>,
    pub assumptions: Vec<String>,
}

impl KvCacheEstimate {
    pub fn of(ir: &ModelIr, context: Option<u64>) -> Self {
        let context = context.or(ir.max_positions);
        let mut est = Self {
            global_layers: 0,
            windowed_layers: 0,
            linear_layers: 0,
            elements_per_token: 0,
            context,
            precisions: Vec::new(),
            linear_state_bytes: None,
            unknown_layers: Vec::new(),
            assumptions: vec![
                "Only the main trunk is counted: MTP modules and vision towers are excluded."
                    .into(),
                "Per sequence, excluding paging overhead and activation memory.".into(),
            ],
        };
        // Elements held by windowed layers at the given context.
        let mut windowed_elements = 0u64;
        let mut linear_state = Some(0u64);

        for layer in &ir.layers {
            match &layer.mixer {
                Mixer::Attention(a) => {
                    let per_token = if a.kind == AttentionKind::Mla {
                        let m = a.mla.as_ref();
                        m.and_then(|m| Some(m.kv_lora_rank? + m.qk_rope_head_dim?))
                    } else {
                        a.num_kv_heads.zip(a.head_dim).map(|(h, d)| 2 * h * d)
                    };
                    let Some(per_token) = per_token else {
                        est.unknown_layers.push(layer.index);
                        continue;
                    };
                    match a.sliding_window {
                        Some(w) => {
                            est.windowed_layers += 1;
                            windowed_elements += per_token * context.map_or(w, |c| c.min(w));
                        }
                        None => {
                            est.global_layers += 1;
                            est.elements_per_token += per_token;
                        }
                    }
                }
                Mixer::LinearAttention(l) => {
                    est.linear_layers += 1;
                    let state = (|| {
                        let (kh, vh, kd, vd) = (
                            l.num_key_heads?,
                            l.num_value_heads?,
                            l.key_head_dim?,
                            l.value_head_dim?,
                        );
                        let recurrent = vh * kd * vd * 4;
                        let conv = l
                            .conv_kernel
                            .map_or(0, |k| (k - 1) * (2 * kh * kd + vh * vd) * 2);
                        Some(recurrent + conv)
                    })();
                    linear_state = linear_state.zip(state).map(|(a, b)| a + b);
                }
                Mixer::Unknown => est.unknown_layers.push(layer.index),
            }
        }
        if est.linear_layers > 0 {
            est.linear_state_bytes = linear_state;
            est.assumptions
                .push("Linear-attention state assumes gated-DeltaNet layout: fp32 [v_heads, k_dim, v_dim] + bf16 conv buffer.".into());
        }
        if ir
            .layers
            .iter()
            .any(|l| matches!(&l.mixer, Mixer::Attention(a) if a.kind == AttentionKind::Mla))
        {
            est.assumptions.push(
                "MLA caches the compressed latent (kv_lora_rank + rope dim) once per layer.".into(),
            );
        }

        est.precisions = PRECISIONS
            .iter()
            .map(|&(name, bits)| {
                let bytes_per_token = est.elements_per_token as f64 * bits / 8.0;
                let windowed = windowed_elements as f64 * bits / 8.0;
                KvPrecision {
                    name,
                    bits_per_element: bits,
                    bytes_per_token,
                    bytes_at_context: context.map_or(0.0, |c| bytes_per_token * c as f64)
                        + windowed,
                }
            })
            .collect();
        est
    }
}

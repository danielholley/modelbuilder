//! Tiny synthetic checkpoints (a few KB each) that mimic the layouts of real
//! model families. Tests generate these instead of downloading real models.
//!
//! Weight values are deterministic pseudo-random numbers; they carry no
//! meaning beyond having realistic dtypes and shapes.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mb_formats::{gguf, safetensors, TensorToWrite};
use mb_ir::{DType, GgmlType, MetaType, MetaValue};
use serde_json::{json, Value};

/// GGML type id used to stand in for an unknown vendor quantization type
/// (like PrismML's PQ2_0).
pub const VENDOR_GGML_TYPE: u32 = 200;
/// Bits per weight of the stand-in vendor type (ternary + a 16-bit scale per 128).
pub const VENDOR_BITS: f64 = 2.125;

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.04
    }
}

fn seed(name: &str) -> u64 {
    name.bytes().fold(0x9E37_79B9_7F4A_7C15, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3)
    }) | 1
}

/// Synthesizes data for a tensor of the given dtype and shape.
fn data(name: &str, dtype: DType, shape: &[u64]) -> Vec<u8> {
    let n: u64 = shape.iter().product();
    let mut rng = Rng(seed(name));
    match dtype {
        DType::F32 => (0..n).flat_map(|_| rng.next_f32().to_le_bytes()).collect(),
        DType::Bf16 => (0..n)
            .flat_map(|_| ((rng.next_f32().to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        DType::F16 => (0..n)
            .flat_map(|_| f32_to_f16(rng.next_f32()).to_le_bytes())
            .collect(),
        other => {
            let bytes = other
                .storage_bytes(n)
                .unwrap_or_else(|| (n as f64 * VENDOR_BITS / 8.0).ceil() as u64);
            (0..bytes)
                .map(|_| (rng.next_f32().to_bits() >> 8) as u8)
                .collect()
        }
    }
}

/// Minimal f32→f16 conversion for small normal values (enough for fixtures).
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp.min(30) as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

struct Spec {
    name: String,
    dtype: DType,
    shape: Vec<u64>,
}

#[derive(Default)]
struct Builder {
    specs: Vec<Spec>,
}

impl Builder {
    fn add(&mut self, name: impl Into<String>, dtype: DType, shape: &[u64]) -> &mut Self {
        self.specs.push(Spec {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
        });
        self
    }

    fn materialize(&self) -> Vec<TensorToWrite<'static>> {
        self.specs
            .iter()
            .map(|s| TensorToWrite {
                name: s.name.clone(),
                dtype: s.dtype,
                shape: s.shape.clone(),
                data: Cow::Owned(data(&s.name, s.dtype, &s.shape)),
            })
            .collect()
    }

    /// Writes an HF directory, split into `shards` safetensors files (with an
    /// index when `shards > 1`).
    fn write_hf(&self, dir: &Path, config: &Value, shards: usize) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(config).unwrap(),
        )
        .unwrap();
        let tensors = self.materialize();
        let meta = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        if shards <= 1 {
            safetensors::write(&dir.join("model.safetensors"), &tensors, &meta).unwrap();
            return dir.to_owned();
        }
        let per = tensors.len().div_ceil(shards);
        let mut weight_map = serde_json::Map::new();
        for (i, chunk) in tensors.chunks(per).enumerate() {
            let file = format!("model-{:05}-of-{:05}.safetensors", i + 1, shards);
            safetensors::write(&dir.join(&file), chunk, &meta).unwrap();
            for t in chunk {
                weight_map.insert(t.name.clone(), file.clone().into());
            }
        }
        let index = json!({ "metadata": {}, "weight_map": weight_map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&index).unwrap(),
        )
        .unwrap();
        dir.to_owned()
    }
}

const BF16: DType = DType::Bf16;
const F32: DType = DType::F32;

fn dense_mlp(b: &mut Builder, p: &str, hidden: u64, inter: u64) {
    b.add(format!("{p}.mlp.gate_proj.weight"), BF16, &[inter, hidden])
        .add(format!("{p}.mlp.up_proj.weight"), BF16, &[inter, hidden])
        .add(format!("{p}.mlp.down_proj.weight"), BF16, &[hidden, inter]);
}

fn norms(b: &mut Builder, p: &str, hidden: u64) {
    b.add(format!("{p}.input_layernorm.weight"), BF16, &[hidden])
        .add(
            format!("{p}.post_attention_layernorm.weight"),
            BF16,
            &[hidden],
        );
}

/// Llama-style dense GQA model: 2 layers, 4 query heads, 2 KV heads, tied embeddings.
/// Includes a model card and tokenizer config with a chat template.
pub fn llama_gqa(dir: &Path) -> PathBuf {
    let (hidden, heads, kv, hd, inter, vocab) = (64, 4, 2, 16, 128, 256);
    let config = json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": 2,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "vocab_size": vocab,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 8192},
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16"
    });
    let mut b = Builder::default();
    b.add("model.embed_tokens.weight", BF16, &[vocab, hidden]);
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        b.add(
            format!("{p}.self_attn.q_proj.weight"),
            BF16,
            &[heads * hd, hidden],
        )
        .add(
            format!("{p}.self_attn.k_proj.weight"),
            BF16,
            &[kv * hd, hidden],
        )
        .add(
            format!("{p}.self_attn.v_proj.weight"),
            BF16,
            &[kv * hd, hidden],
        )
        .add(
            format!("{p}.self_attn.o_proj.weight"),
            BF16,
            &[hidden, heads * hd],
        );
        dense_mlp(&mut b, &p, hidden, inter);
        norms(&mut b, &p, hidden);
    }
    b.add("model.norm.weight", BF16, &[hidden]);
    let out = b.write_hf(dir, &config, 1);
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&json!({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("README.md"),
        "---\nlicense: apache-2.0\nbase_model: example/tiny-llama-base\ntags:\n- text-generation\n---\n# Tiny Llama\n",
    )
    .unwrap();
    out
}

/// Qwen3.8-like hybrid model (HF `qwen3_5` layout and tensor names), scaled down: 8 layers in the pattern
/// 3 × Gated DeltaNet + 1 × gated full attention (GQA), an MTP module, and a
/// vision tower, nested under `text_config`, written as 2 shards.
pub fn qwen_hybrid(dir: &Path) -> PathBuf {
    let (hidden, heads, kv, hd, inter, vocab) = (64, 6, 2, 16, 160, 320);
    let (lk_heads, lv_heads, lk_dim, lv_dim, conv) = (2, 6, 8, 8, 4);
    let layer_types: Vec<&str> = (0..8)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let config = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text",
            "attn_output_gate": true,
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_hidden_layers": 8,
            "num_attention_heads": heads,
            "num_key_value_heads": kv,
            "head_dim": hd,
            "vocab_size": vocab,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "layer_types": layer_types,
            "linear_num_key_heads": lk_heads,
            "linear_num_value_heads": lv_heads,
            "linear_key_head_dim": lk_dim,
            "linear_value_head_dim": lv_dim,
            "linear_conv_kernel_dim": conv,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0}
        },
        "vision_config": {"depth": 1, "hidden_size": 32}
    });
    let mut b = Builder::default();
    let lm = "model.language_model";
    b.add(format!("{lm}.embed_tokens.weight"), BF16, &[vocab, hidden]);
    for l in 0..8u64 {
        let p = format!("{lm}.layers.{l}");
        if l % 4 == 3 {
            let a = format!("{p}.self_attn");
            // Q projection carries the output gate, hence 2×.
            b.add(
                format!("{a}.q_proj.weight"),
                BF16,
                &[2 * heads * hd, hidden],
            )
            .add(format!("{a}.k_proj.weight"), BF16, &[kv * hd, hidden])
            .add(format!("{a}.v_proj.weight"), BF16, &[kv * hd, hidden])
            .add(format!("{a}.o_proj.weight"), BF16, &[hidden, heads * hd])
            .add(format!("{a}.q_norm.weight"), BF16, &[hd])
            .add(format!("{a}.k_norm.weight"), BF16, &[hd]);
        } else {
            let a = format!("{p}.linear_attn");
            // Split projections, as in the released Qwen3.8-27B checkpoint.
            let conv_dim = 2 * lk_heads * lk_dim + lv_heads * lv_dim;
            b.add(format!("{a}.in_proj_qkv.weight"), BF16, &[conv_dim, hidden])
                .add(
                    format!("{a}.in_proj_z.weight"),
                    BF16,
                    &[lv_heads * lv_dim, hidden],
                )
                .add(format!("{a}.in_proj_a.weight"), BF16, &[lv_heads, hidden])
                .add(format!("{a}.in_proj_b.weight"), BF16, &[lv_heads, hidden])
                .add(format!("{a}.conv1d.weight"), BF16, &[conv_dim, 1, conv])
                .add(format!("{a}.A_log"), F32, &[lv_heads])
                .add(format!("{a}.dt_bias"), BF16, &[lv_heads])
                .add(format!("{a}.norm.weight"), F32, &[lv_dim])
                .add(
                    format!("{a}.out_proj.weight"),
                    BF16,
                    &[hidden, lv_heads * lv_dim],
                );
        }
        dense_mlp(&mut b, &p, hidden, inter);
        norms(&mut b, &p, hidden);
    }
    b.add(format!("{lm}.norm.weight"), BF16, &[hidden])
        .add("lm_head.weight", BF16, &[vocab, hidden])
        .add("mtp.fc.weight", BF16, &[hidden, 2 * hidden])
        .add("mtp.pre_fc_norm_embedding.weight", BF16, &[hidden])
        .add("mtp.pre_fc_norm_hidden.weight", BF16, &[hidden])
        .add(
            "mtp.layers.0.self_attn.q_proj.weight",
            BF16,
            &[2 * heads * hd, hidden],
        )
        .add(
            "mtp.layers.0.self_attn.k_proj.weight",
            BF16,
            &[kv * hd, hidden],
        )
        .add(
            "mtp.layers.0.self_attn.v_proj.weight",
            BF16,
            &[kv * hd, hidden],
        )
        .add(
            "mtp.layers.0.self_attn.o_proj.weight",
            BF16,
            &[hidden, heads * hd],
        )
        .add("mtp.layers.0.mlp.down_proj.weight", BF16, &[hidden, inter])
        .add("mtp.norm.weight", BF16, &[hidden])
        .add(
            "model.visual.patch_embed.proj.weight",
            BF16,
            &[32, 3, 2, 16, 16],
        )
        .add("model.visual.blocks.0.attn.qkv.weight", BF16, &[96, 32])
        .add(
            "model.visual.merger.linear_fc2.weight",
            BF16,
            &[hidden, 128],
        );
    b.write_hf(dir, &config, 2)
}

/// DeepSeek-V3-style model: MLA attention, 1 dense layer then MoE layers
/// (4 routed + 1 shared expert), and a next-n MTP module stored as layer 3.
pub fn deepseek_mla_moe(dir: &Path) -> PathBuf {
    let (hidden, heads, vocab) = (64, 4, 256);
    let (q_rank, kv_rank, rope_d, nope_d, v_d) = (48, 32, 8, 16, 16);
    let (inter, moe_inter, experts) = (128, 32, 4);
    let config = json!({
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "num_hidden_layers": 3,
        "first_k_dense_replace": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "n_routed_experts": experts,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "q_lora_rank": q_rank,
        "kv_lora_rank": kv_rank,
        "qk_rope_head_dim": rope_d,
        "qk_nope_head_dim": nope_d,
        "v_head_dim": v_d,
        "num_nextn_predict_layers": 1,
        "vocab_size": vocab,
        "max_position_embeddings": 163840,
        "rope_theta": 10000.0,
        "rope_scaling": {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096},
        "tie_word_embeddings": false,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    });
    let mut b = Builder::default();
    b.add("model.embed_tokens.weight", BF16, &[vocab, hidden]);
    for l in 0..4u64 {
        let p = format!("model.layers.{l}");
        let a = format!("{p}.self_attn");
        b.add(format!("{a}.q_a_proj.weight"), BF16, &[q_rank, hidden])
            .add(format!("{a}.q_a_layernorm.weight"), BF16, &[q_rank])
            .add(
                format!("{a}.q_b_proj.weight"),
                BF16,
                &[heads * (nope_d + rope_d), q_rank],
            )
            .add(
                format!("{a}.kv_a_proj_with_mqa.weight"),
                BF16,
                &[kv_rank + rope_d, hidden],
            )
            .add(format!("{a}.kv_a_layernorm.weight"), BF16, &[kv_rank])
            .add(
                format!("{a}.kv_b_proj.weight"),
                BF16,
                &[heads * (nope_d + v_d), kv_rank],
            )
            .add(format!("{a}.o_proj.weight"), BF16, &[hidden, heads * v_d]);
        if l == 0 {
            dense_mlp(&mut b, &p, hidden, inter);
        } else {
            b.add(format!("{p}.mlp.gate.weight"), BF16, &[experts, hidden])
                .add(
                    format!("{p}.mlp.gate.e_score_correction_bias"),
                    F32,
                    &[experts],
                );
            for e in 0..experts {
                let x = format!("{p}.mlp.experts.{e}");
                b.add(format!("{x}.gate_proj.weight"), BF16, &[moe_inter, hidden])
                    .add(format!("{x}.up_proj.weight"), BF16, &[moe_inter, hidden])
                    .add(format!("{x}.down_proj.weight"), BF16, &[hidden, moe_inter]);
            }
            b.add(
                format!("{p}.mlp.shared_experts.gate_proj.weight"),
                BF16,
                &[moe_inter, hidden],
            )
            .add(
                format!("{p}.mlp.shared_experts.up_proj.weight"),
                BF16,
                &[moe_inter, hidden],
            )
            .add(
                format!("{p}.mlp.shared_experts.down_proj.weight"),
                BF16,
                &[hidden, moe_inter],
            );
        }
        norms(&mut b, &p, hidden);
        if l == 3 {
            b.add(format!("{p}.eh_proj.weight"), BF16, &[hidden, 2 * hidden])
                .add(format!("{p}.enorm.weight"), BF16, &[hidden])
                .add(format!("{p}.hnorm.weight"), BF16, &[hidden])
                .add(format!("{p}.shared_head.norm.weight"), BF16, &[hidden]);
        }
    }
    b.add("model.norm.weight", BF16, &[hidden])
        .add("lm_head.weight", BF16, &[vocab, hidden]);
    b.write_hf(dir, &config, 1)
}

/// Llama-architecture GGUF with mixed quantization: F16 embeddings, Q8_0
/// attention, and FFN weights in an unknown vendor type ([`VENDOR_GGML_TYPE`]).
pub fn gguf_mixed_quant(dir: &Path) -> PathBuf {
    let (hidden, heads, kv, hd, inter, vocab) = (64u64, 4u64, 2u64, 16u64, 128u64, 256u64);
    let q8 = DType::Ggml(GgmlType::from_name("Q8_0").unwrap());
    let vendor = DType::Ggml(GgmlType(VENDOR_GGML_TYPE));
    let mut b = Builder::default();
    b.add("token_embd.weight", DType::F16, &[vocab, hidden]);
    for l in 0..2 {
        let p = format!("blk.{l}");
        b.add(format!("{p}.attn_norm.weight"), F32, &[hidden])
            .add(format!("{p}.attn_q.weight"), q8, &[heads * hd, hidden])
            .add(format!("{p}.attn_k.weight"), q8, &[kv * hd, hidden])
            .add(format!("{p}.attn_v.weight"), q8, &[kv * hd, hidden])
            .add(format!("{p}.attn_output.weight"), q8, &[hidden, heads * hd])
            .add(format!("{p}.ffn_norm.weight"), F32, &[hidden])
            .add(format!("{p}.ffn_gate.weight"), vendor, &[inter, hidden])
            .add(format!("{p}.ffn_up.weight"), vendor, &[inter, hidden])
            .add(format!("{p}.ffn_down.weight"), vendor, &[hidden, inter]);
    }
    b.add("output_norm.weight", F32, &[hidden])
        .add("output.weight", q8, &[vocab, hidden]);

    let u32v = |v: u64| MetaValue::U32(v as u32);
    let kv_pairs = vec![
        (
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        ),
        (
            "general.name".to_string(),
            MetaValue::String("Tiny Mixed Quant".into()),
        ),
        (
            "general.license".to_string(),
            MetaValue::String("apache-2.0".into()),
        ),
        ("llama.block_count".to_string(), u32v(2)),
        ("llama.context_length".to_string(), u32v(8192)),
        ("llama.embedding_length".to_string(), u32v(hidden)),
        ("llama.feed_forward_length".to_string(), u32v(inter)),
        ("llama.attention.head_count".to_string(), u32v(heads)),
        ("llama.attention.head_count_kv".to_string(), u32v(kv)),
        ("llama.rope.freq_base".to_string(), MetaValue::F32(500000.0)),
        (
            "tokenizer.ggml.model".to_string(),
            MetaValue::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array {
                elem: MetaType::String,
                values: (0..vocab)
                    .map(|i| MetaValue::String(format!("t{i}")))
                    .collect(),
            },
        ),
        (
            "tokenizer.chat_template".to_string(),
            MetaValue::String("{{ messages }}".into()),
        ),
    ];
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join("tiny-mixed.gguf");
    gguf::write(&path, &kv_pairs, &b.materialize()).unwrap();
    path
}

/// Bonsai-2-like GGUF: llama.cpp `qwen35` layout (3 × gated DeltaNet + 1 ×
/// gated GQA, repeated), ternary `PQ2_0` weights, bf16/f32 recurrent-state
/// tensors, and PrismML Hadamard-rotation metadata. No MTP tensors, matching
/// the released Ternary-Bonsai-2-27B files.
pub fn gguf_bonsai_like(dir: &Path) -> PathBuf {
    // PQ2_0 blocks are 128 weights along the input dimension, so every input
    // width here is a multiple of 128.
    let (hidden, heads, kv, hd, inter, vocab) = (128u64, 4u64, 2u64, 32u64, 256u64, 256u64);
    let (k_heads, v_heads, state) = (2u64, 4u64, 32u64);
    let pq2 = DType::Ggml(GgmlType::from_name("PQ2_0").unwrap());
    let conv_dim = 2 * k_heads * state + v_heads * state;
    let mut b = Builder::default();
    let mut rotated = vec!["output.weight".to_string()];
    b.add("token_embd.weight", pq2, &[vocab, hidden]);
    for l in 0..8u64 {
        let p = format!("blk.{l}");
        b.add(format!("{p}.attn_norm.weight"), F32, &[hidden]);
        let mut add_rot = |b: &mut Builder, name: String, shape: &[u64]| {
            rotated.push(name.clone());
            b.add(name, pq2, shape);
        };
        if l % 4 == 3 {
            add_rot(
                &mut b,
                format!("{p}.attn_q.weight"),
                &[2 * heads * hd, hidden],
            );
            add_rot(&mut b, format!("{p}.attn_k.weight"), &[kv * hd, hidden]);
            add_rot(&mut b, format!("{p}.attn_v.weight"), &[kv * hd, hidden]);
            add_rot(
                &mut b,
                format!("{p}.attn_output.weight"),
                &[hidden, heads * hd],
            );
            b.add(format!("{p}.attn_q_norm.weight"), F32, &[hd]).add(
                format!("{p}.attn_k_norm.weight"),
                F32,
                &[hd],
            );
        } else {
            add_rot(&mut b, format!("{p}.attn_qkv.weight"), &[conv_dim, hidden]);
            add_rot(
                &mut b,
                format!("{p}.attn_gate.weight"),
                &[v_heads * state, hidden],
            );
            add_rot(
                &mut b,
                format!("{p}.ssm_out.weight"),
                &[hidden, v_heads * state],
            );
            b.add(format!("{p}.ssm_a"), F32, &[v_heads])
                .add(format!("{p}.ssm_alpha.weight"), BF16, &[v_heads, hidden])
                .add(format!("{p}.ssm_beta.weight"), BF16, &[v_heads, hidden])
                .add(format!("{p}.ssm_conv1d.weight"), F32, &[conv_dim, 4])
                .add(format!("{p}.ssm_dt.bias"), F32, &[v_heads])
                .add(format!("{p}.ssm_norm.weight"), F32, &[state]);
        }
        add_rot(&mut b, format!("{p}.ffn_gate.weight"), &[inter, hidden]);
        add_rot(&mut b, format!("{p}.ffn_up.weight"), &[inter, hidden]);
        add_rot(&mut b, format!("{p}.ffn_down.weight"), &[hidden, inter]);
        b.add(format!("{p}.post_attention_norm.weight"), F32, &[hidden]);
    }
    b.add("output_norm.weight", F32, &[hidden])
        .add("output.weight", pq2, &[vocab, hidden]);

    let u32v = |v: u64| MetaValue::U32(v as u32);
    let s = |v: &str| MetaValue::String(v.into());
    let strings = |v: Vec<String>| MetaValue::Array {
        elem: MetaType::String,
        values: v.into_iter().map(MetaValue::String).collect(),
    };
    let kv_pairs = vec![
        ("general.architecture".to_string(), s("qwen35")),
        ("general.name".to_string(), s("Tiny Bonsai-like")),
        ("qwen35.block_count".to_string(), u32v(8)),
        ("qwen35.context_length".to_string(), u32v(262144)),
        ("qwen35.embedding_length".to_string(), u32v(hidden)),
        ("qwen35.feed_forward_length".to_string(), u32v(inter)),
        ("qwen35.attention.head_count".to_string(), u32v(heads)),
        ("qwen35.attention.head_count_kv".to_string(), u32v(kv)),
        ("qwen35.attention.key_length".to_string(), u32v(hd)),
        ("qwen35.attention.value_length".to_string(), u32v(hd)),
        (
            "qwen35.rope.freq_base".to_string(),
            MetaValue::F32(10_000_000.0),
        ),
        ("qwen35.ssm.conv_kernel".to_string(), u32v(4)),
        ("qwen35.ssm.state_size".to_string(), u32v(state)),
        ("qwen35.ssm.group_count".to_string(), u32v(k_heads)),
        ("qwen35.ssm.time_step_rank".to_string(), u32v(v_heads)),
        ("qwen35.ssm.inner_size".to_string(), u32v(v_heads * state)),
        ("qwen35.full_attention_interval".to_string(), u32v(4)),
        ("prism.hadamard.version".to_string(), u32v(1)),
        ("prism.hadamard.block_size".to_string(), u32v(128)),
        (
            "prism.hadamard.transform".to_string(),
            s("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.weight_names".to_string(), strings(rotated)),
        (
            "prism.hadamard.inverse_weight_names".to_string(),
            strings(vec!["token_embd.weight".into()]),
        ),
        ("general.file_type".to_string(), u32v(141)),
    ];
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join("tiny-bonsai-like.gguf");
    gguf::write(&path, &kv_pairs, &b.materialize()).unwrap();
    path
}

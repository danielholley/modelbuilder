use mb_analyze::analyze;
use mb_ir::{AttentionKind, FfnSpec, Metadata, Mixer, ModelIr, RawModel, SourceFormat};
use serde_json::json;

fn load(path: &std::path::Path) -> ModelIr {
    ModelIr::from_raw(mb_formats::open(path).unwrap().raw)
}

fn bf16_bytes_per_token(r: &mb_analyze::Report) -> f64 {
    r.kv_cache
        .precisions
        .iter()
        .find(|p| p.name == "bf16")
        .unwrap()
        .bytes_per_token
}

#[test]
fn llama_gqa() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::llama_gqa(dir.path()));
    assert!(ir.warnings.is_empty(), "{:?}", ir.warnings);
    assert_eq!(ir.layers.len(), 2);
    let Mixer::Attention(a) = &ir.layers[0].mixer else {
        panic!()
    };
    assert_eq!(
        (a.kind, a.num_heads, a.num_kv_heads, a.head_dim),
        (AttentionKind::Gqa, Some(4), Some(2), Some(16))
    );
    assert!(!a.output_gate);

    let r = analyze(&ir, None);
    assert_eq!(
        r.architecture.layer_pattern.to_string(),
        "2 × gqa(4q/2kv,d16)+dense"
    );
    // 2 layers × 2 (K,V) × 2 heads × 16 dims × 2 bytes
    assert_eq!(bf16_bytes_per_token(&r), 256.0);
    assert_eq!(r.params.lm_head, 0);
    assert_eq!(r.architecture.tie_word_embeddings, Some(true));
    assert_eq!(r.provenance.base_models, vec!["example/tiny-llama-base"]);
    assert_eq!(r.provenance.license.as_deref(), Some("apache-2.0"));
    assert!(r.provenance.has_chat_template);
    assert_eq!(r.provenance.original_context, Some(8192));
    assert!((r.quantization.bits_per_param - 16.0).abs() < 1e-9);
}

#[test]
fn qwen_hybrid() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::qwen_hybrid(dir.path()));
    assert!(ir.warnings.is_empty(), "{:?}", ir.warnings);
    assert_eq!(ir.raw.files.len(), 2);
    assert_eq!(ir.family.as_deref(), Some("qwen3_5_text"));

    let r = analyze(&ir, None);
    assert_eq!(
        r.architecture.layer_pattern.to_string(),
        "2 × [3 × gated_deltanet+dense, 1 × gqa(6q/2kv,d16,gated)+dense]"
    );
    assert_eq!(r.architecture.mtp_modules, 1);
    assert!(r.architecture.multimodal);
    assert_eq!((r.kv_cache.global_layers, r.kv_cache.linear_layers), (2, 6));
    // 2 full layers × 2 × 2 kv heads × 16 dims × 2 bytes
    assert_eq!(bf16_bytes_per_token(&r), 256.0);
    assert!(r.kv_cache.linear_state_bytes.unwrap() > 0);
    assert!(r.params.mtp > 0 && r.params.multimodal > 0 && r.params.linear_attention > 0);
    assert_eq!(r.architecture.rope_theta, Some(1.0e7));
}

#[test]
fn deepseek_mla_moe() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::deepseek_mla_moe(dir.path()));
    assert!(ir.warnings.is_empty(), "{:?}", ir.warnings);
    assert_eq!(
        ir.layers.len(),
        3,
        "the next-n layer must not count as trunk"
    );
    assert_eq!(ir.mtp.as_ref().unwrap().num_modules, 1);
    let Mixer::Attention(a) = &ir.layers[1].mixer else {
        panic!()
    };
    assert_eq!(a.kind, AttentionKind::Mla);
    let FfnSpec::Moe(m) = &ir.layers[1].ffn else {
        panic!()
    };
    assert_eq!(
        (m.num_experts, m.experts_per_token, m.num_shared_experts),
        (Some(4), Some(2), 1)
    );
    assert!(matches!(ir.layers[0].ffn, FfnSpec::Dense { .. }));

    let r = analyze(&ir, Some(1000));
    // 3 layers × (kv_lora_rank 32 + rope 8) × 2 bytes
    assert_eq!(bf16_bytes_per_token(&r), 240.0);
    assert_eq!(r.kv_cache.precisions[0].bytes_at_context, 240_000.0);
    assert!(r.params.active_per_token < r.params.total - r.params.mtp);
    assert_eq!(r.provenance.original_context, Some(4096));
    assert!(r.quantization.declared.is_some());
}

#[test]
fn gguf_mixed_quant() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::gguf_mixed_quant(dir.path()));
    assert!(ir.warnings.is_empty(), "{:?}", ir.warnings);
    assert_eq!(ir.family.as_deref(), Some("llama"));
    assert_eq!(ir.vocab_size, Some(256));
    let r = analyze(&ir, None);
    assert_eq!(
        r.architecture.layer_pattern.to_string(),
        "2 × gqa(4q/2kv,d16)+dense"
    );
    let vendor = r
        .quantization
        .by_dtype
        .iter()
        .find(|s| s.dtype == "ggml_type_200")
        .unwrap();
    assert!(!vendor.bytes_exact);
    // Inferred sizes can include alignment padding, so allow a little slack.
    assert!(
        (vendor.bits_per_param - mb_fixtures::VENDOR_BITS).abs() < 0.1,
        "{}",
        vendor.bits_per_param
    );
    assert_eq!(r.quantization.notes.len(), 1);
    assert_eq!(r.provenance.name.as_deref(), Some("Tiny Mixed Quant"));
    assert!(r.provenance.has_chat_template);
}

/// The ternary GGUF path: rotated PQ2_0 weights in the qwen35 layout.
#[test]
fn gguf_hybrid_ternary() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::gguf_hybrid_ternary(dir.path()));
    assert!(ir.warnings.is_empty(), "{:?}", ir.warnings);
    let r = analyze(&ir, None);
    assert_eq!(
        r.architecture.layer_pattern.to_string(),
        "2 × [3 × gated_deltanet+dense, 1 × gqa(4q/2kv,d32,gated)+dense]"
    );
    assert_eq!(r.architecture.tie_word_embeddings, Some(false));
    assert_eq!(r.architecture.mtp_modules, 0);

    // `attn_qkv`/`attn_gate` in DeltaNet layers count as linear attention, not attention.
    let full_attn_per_layer = (2 * 4 * 32 + 2 * 2 * 32 + 128) * 128 + 2 * 32;
    assert_eq!(r.params.attention, 2 * full_attn_per_layer);
    let linear_roles = ir
        .tensors()
        .filter(|(t, r)| t.name.starts_with("blk.0.") && r.kind == mb_ir::TensorKind::LinearAttn)
        .count();
    assert_eq!(linear_roles, 9);

    // Known vendor layout: exact sizes, 2.125 bits per weight.
    let pq2 = r
        .quantization
        .by_dtype
        .iter()
        .find(|s| s.dtype == "PQ2_0")
        .unwrap();
    assert!(pq2.bytes_exact);
    assert_eq!(pq2.bits_per_param, 2.125);
    let rot = r.quantization.rotation.as_ref().unwrap();
    assert_eq!((rot.block_size, rot.inverse_tensors), (Some(128), 1));
    assert_eq!(rot.rotated_tensors, 1 + 2 * 7 + 6 * 6);
    assert_eq!(r.quantization.notes.len(), 2);

    // 2 full layers × 2 × 2 kv heads × 32 dims × 2 bytes
    assert_eq!(bf16_bytes_per_token(&r), 512.0);
    // 6 layers × (fp32 [4 v heads, 32, 32] + bf16 conv (4-1) × 256)
    let expected_state = 6 * (4 * 32 * 32 * 4 + 3 * 256 * 2);
    assert_eq!(r.kv_cache.linear_state_bytes, Some(expected_state));
}

/// A full-size hybrid config (64 layers, 16 GQA layers of 4 KV heads × 256, 262K
/// context), from config alone: 64 KiB/token and 16 GiB at full context.
#[test]
fn full_size_hybrid_kv_from_config() {
    let layer_types: Vec<&str> = (0..64)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let config = json!({
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "max_position_embeddings": 262144, "layer_types": layer_types,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4
        }
    });
    let ir = ModelIr::from_raw(RawModel {
        format: SourceFormat::HfSafetensors,
        root: "config-only".into(),
        files: vec![],
        metadata: Metadata::Hf {
            config,
            safetensors_metadata: Default::default(),
        },
        aux: Default::default(),
        tensors: vec![],
    });
    let r = analyze(&ir, None);
    assert_eq!(
        (r.kv_cache.global_layers, r.kv_cache.linear_layers),
        (16, 48)
    );
    assert_eq!(bf16_bytes_per_token(&r), 65536.0);
    let at_ctx = r.kv_cache.precisions[0].bytes_at_context;
    assert_eq!(at_ctx, 65536.0 * 262144.0); // 16 GiB
}

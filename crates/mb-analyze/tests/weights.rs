use mb_analyze::weights::{weight_stats, WeightStatsOptions};
use mb_ir::ModelIr;

fn open(path: &std::path::Path) -> (mb_formats::LoadedModel, ModelIr) {
    let model = mb_formats::open(path).unwrap();
    let ir = ModelIr::from_raw(model.raw.clone());
    (model, ir)
}

fn all() -> WeightStatsOptions {
    WeightStatsOptions {
        only: vec![],
        kv_spectra: true,
    }
}

#[test]
fn bonsai_like_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let (model, ir) = open(&mb_fixtures::gguf_bonsai_like(dir.path()));
    let r = weight_stats(&model, &ir, &all()).unwrap();
    assert!(r.skipped.is_empty(), "{:?}", r.skipped);
    assert_eq!(r.tensors.len(), ir.raw.tensors.len());

    let q = r
        .tensors
        .iter()
        .find(|t| t.name == "blk.3.attn_q.weight")
        .unwrap();
    assert!(q.unrotated, "rotated tensors get primal statistics");
    assert_eq!(q.ternary_group_fraction, Some(1.0));
    assert!(
        q.zero_fraction > 0.2 && q.zero_fraction < 0.45,
        "{}",
        q.zero_fraction
    );
    // Undoing the rotation spreads each ternary block over many values, so
    // the primal max is no longer a single scale level.
    assert!(q.max_abs > 0.0 && q.channel_outlier_ratio.unwrap() >= 1.0);

    let norm = r
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .unwrap();
    assert!(!norm.unrotated && norm.channel_outlier_ratio.is_none());

    // Two full-attention layers: K and V are 2 heads × 32 = 64 rows over 128 inputs.
    assert_eq!(r.kv_spectra.len(), 2);
    let s = &r.kv_spectra[0];
    assert_eq!((s.layer, s.k_rows, s.v_rows, s.cols), (3, 64, 64, 128));
    assert_eq!((s.k.full_rank, s.kv.full_rank), (64, 128));
    assert!(s.kv.energy_rank_99 <= 128 && s.kv.energy_rank_90 < s.kv.energy_rank_99);
    assert!(r.notes.iter().any(|n| n.contains("rotated basis")));
}

#[test]
fn filters_and_unsupported_types() {
    let dir = tempfile::tempdir().unwrap();
    let (model, ir) = open(&mb_fixtures::gguf_mixed_quant(dir.path()));
    let opts = WeightStatsOptions {
        only: vec!["blk.0.".into()],
        kv_spectra: false,
    };
    let r = weight_stats(&model, &ir, &opts).unwrap();
    assert!(r.tensors.iter().all(|t| t.name.starts_with("blk.0.")));
    // The vendor stand-in type can't be decoded: skipped, not an error.
    let skipped: Vec<&str> = r.skipped.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        skipped,
        [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight"
        ]
    );
    let q = r
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .unwrap();
    assert_eq!(
        q.ternary_group_fraction, None,
        "64-wide rows have no 128-groups"
    );
    assert!(q.rms > 0.0 && q.rms.is_finite());
}

#[test]
fn bf16_safetensors_spectra() {
    let dir = tempfile::tempdir().unwrap();
    let (model, ir) = open(&mb_fixtures::llama_gqa(dir.path()));
    let r = weight_stats(&model, &ir, &all()).unwrap();
    assert_eq!(r.kv_spectra.len(), 2);
    // Independent pseudo-random K and V: no shared low-rank structure. (A random
    // square matrix still reaches 90% energy at about half its rank.)
    let s = &r.kv_spectra[0];
    assert_eq!(s.kv.full_rank, 64);
    assert!(
        s.kv.energy_rank_99 > 40 && s.kv.effective_rank > 40.0,
        "{:?}",
        s.kv
    );
    assert!(
        s.kv.energy_rank_90 > s.k.energy_rank_90,
        "stacking must add rank"
    );
    let k = r
        .tensors
        .iter()
        .find(|t| t.name.ends_with("k_proj.weight"))
        .unwrap();
    assert!(k.kurtosis.abs() < 1.5, "uniform-ish noise: {}", k.kurtosis);
    assert_eq!(k.ternary_group_fraction, None);
}

#[test]
fn deepseek_mla_is_skipped_for_spectra() {
    let dir = tempfile::tempdir().unwrap();
    let (model, ir) = open(&mb_fixtures::deepseek_mla_moe(dir.path()));
    let opts = WeightStatsOptions {
        only: vec!["self_attn".into()],
        kv_spectra: true,
    };
    let r = weight_stats(&model, &ir, &opts).unwrap();
    assert!(r.kv_spectra.is_empty());
    assert_eq!(
        r.skipped
            .iter()
            .filter(|(n, _)| n.contains("K/V spectrum"))
            .count(),
        3
    );
}

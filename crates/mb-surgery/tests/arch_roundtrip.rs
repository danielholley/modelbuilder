//! GGUF → `export_hf` (f32) → `replace_tensors` (every tensor) must give back
//! the original GGUF: quantized tensors byte for byte, floats to float precision.
//! This exercises every adapter transform in both directions.

use std::path::Path;

use mb_formats::dequant::dequantize;
use mb_ir::{DType, ModelIr};
use mb_surgery::hf_export::{export_hf, HfExportOptions};
use mb_surgery::replace::{replace_tensors, ReplaceOptions};

fn open(p: &Path) -> (mb_formats::LoadedModel, ModelIr) {
    let m = mb_formats::open(p).unwrap();
    let ir = ModelIr::from_raw(m.raw.clone());
    (m, ir)
}

fn roundtrip(gguf: &Path, reference: &Path, dir: &Path) {
    let (m, ir) = open(gguf);
    let hf = dir.join("hf");
    let opts = HfExportOptions {
        reference: reference.to_owned(),
        dtype: DType::F32,
        shard_bytes: 1 << 30,
        layers: None,
        globals: true,
        overwrite: false,
    };
    let report = export_hf(&m, &ir, &hf, &opts).unwrap();
    assert_eq!(
        report.tensors.len(),
        ir.raw.tensors.len(),
        "every tensor is exported"
    );

    let (u, _) = open(&hf);
    let out = dir.join("back.gguf");
    let rep = replace_tensors(
        &m,
        &ir,
        &u,
        &out,
        &ReplaceOptions {
            config: hf.join("config.json"),
            overwrite: false,
            max_rel_error: Some(1e-4),
        },
    )
    .unwrap();
    assert_eq!(
        rep.tensors.len(),
        ir.raw.tensors.len(),
        "every tensor is replaced"
    );

    let (b, bir) = open(&out);
    for t in &ir.raw.tensors {
        let bt = bir.raw.tensors.iter().find(|x| x.name == t.name).unwrap();
        assert_eq!((bt.dtype, &bt.shape), (t.dtype, &t.shape), "{}", t.name);
        let (orig, back) = (m.tensor_bytes(t).unwrap(), b.tensor_bytes(bt).unwrap());
        if matches!(t.dtype, DType::F32 | DType::F16 | DType::Bf16) {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            dequantize(t.dtype, orig, &mut x).unwrap();
            dequantize(t.dtype, back, &mut y).unwrap();
            for (a, c) in x.iter().zip(&y) {
                assert!(
                    (a - c).abs() <= 1e-5 * a.abs().max(1.0),
                    "{}: {a} vs {c}",
                    t.name
                );
            }
        } else {
            assert!(
                orig == back,
                "{} ({}) is not byte-identical after the round trip",
                t.name,
                t.dtype
            );
        }
    }
}

#[test]
fn llama_round_trips_through_hf() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = mb_fixtures::gguf_llama(&dir.path().join("src"));
    roundtrip(&gguf, &dir.path().join("src"), dir.path());
}

#[test]
fn hybrid_ternary_round_trips_through_hf() {
    // qwen35: V-head reorders, A_log, conv1d, zero-centered norms, Hadamard rotation, PQ2_0.
    let dir = tempfile::tempdir().unwrap();
    let gguf = mb_fixtures::gguf_hybrid_ternary(&dir.path().join("src"));
    let reference = mb_fixtures::hybrid_mtp_reference(&dir.path().join("ref"));
    roundtrip(&gguf, &reference, dir.path());
}

#[test]
fn llama_export_undoes_the_rope_permutation() {
    // Q rows in the export must be the GGUF rows gathered by the inverse permutation.
    let dir = tempfile::tempdir().unwrap();
    let gguf = mb_fixtures::gguf_llama(&dir.path().join("src"));
    let (m, ir) = open(&gguf);
    let hf = dir.path().join("hf");
    let opts = HfExportOptions {
        reference: dir.path().join("src"),
        dtype: DType::F32,
        shard_bytes: 1 << 30,
        layers: None,
        globals: true,
        overwrite: false,
    };
    export_hf(&m, &ir, &hf, &opts).unwrap();
    let (u, _) = open(&hf);
    let (q_gguf, q_hf) = (
        m.tensor("blk.0.attn_q.weight").unwrap(),
        u.tensor("model.layers.0.self_attn.q_proj.weight").unwrap(),
    );
    let (mut g, mut h) = (Vec::new(), Vec::new());
    dequantize(q_gguf.dtype, m.tensor_bytes(q_gguf).unwrap(), &mut g).unwrap();
    dequantize(q_hf.dtype, u.tensor_bytes(q_hf).unwrap(), &mut h).unwrap();
    let width = 64;
    let perm = mb_surgery::arch::hf_from_llama_rope(4, 16);
    assert_ne!(perm, (0..64).collect::<Vec<_>>());
    for (r, &src) in perm.iter().enumerate() {
        assert_eq!(
            &h[r * width..(r + 1) * width],
            &g[src * width..(src + 1) * width]
        );
    }
}

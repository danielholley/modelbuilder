//! Dense → MoE upcycling on the fixtures.

use std::borrow::Cow;

use mb_formats::{gguf, TensorToWrite};
use mb_ir::{FfnSpec, MetaValue, Metadata, ModelIr};
use mb_surgery::upcycle::{upcycle, UpcycleOptions};

fn open(p: &std::path::Path) -> (mb_formats::LoadedModel, ModelIr) {
    let m = mb_formats::open(p).unwrap();
    let ir = ModelIr::from_raw(m.raw.clone());
    (m, ir)
}

fn meta<'a>(ir: &'a ModelIr, key: &str) -> Option<&'a MetaValue> {
    let Metadata::Gguf { kv, .. } = &ir.raw.metadata else {
        panic!("not GGUF")
    };
    kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

#[test]
fn llama_experts_are_exact_copies() {
    let dir = tempfile::tempdir().unwrap();
    let (m, ir) = open(&mb_fixtures::gguf_llama(dir.path()));
    let out = dir.path().join("moe.gguf");
    let opts = UpcycleOptions {
        experts: 4,
        top_k: 2,
        ..Default::default()
    };
    let report = upcycle(&m, &ir, &opts, &out).unwrap();
    assert!(report.notes.iter().any(|n| n.contains("exactly")));
    let (u, uir) = open(&out);
    assert_eq!(
        meta(&uir, "general.architecture").unwrap().as_str(),
        Some("llama")
    );
    assert_eq!(meta(&uir, "llama.expert_count").unwrap().as_u64(), Some(4));
    assert_eq!(
        meta(&uir, "llama.expert_used_count").unwrap().as_u64(),
        Some(2)
    );
    for l in &uir.layers {
        assert!(
            matches!(l.ffn, FfnSpec::Moe(_)),
            "layer {} {:?}",
            l.index,
            l.ffn
        );
    }
    for l in 0..2 {
        for (d, e) in [
            ("ffn_gate", "ffn_gate_exps"),
            ("ffn_up", "ffn_up_exps"),
            ("ffn_down", "ffn_down_exps"),
        ] {
            assert!(u.tensor(&format!("blk.{l}.{d}.weight")).is_none());
            let src = m.tensor(&format!("blk.{l}.{d}.weight")).unwrap();
            let dst = u.tensor(&format!("blk.{l}.{e}.weight")).unwrap();
            assert_eq!(dst.shape[0], 4);
            assert_eq!(&dst.shape[1..], &src.shape[..]);
            assert_eq!(dst.dtype, src.dtype);
            let (s, d) = (m.tensor_bytes(src).unwrap(), u.tensor_bytes(dst).unwrap());
            for c in d.chunks(s.len()) {
                assert_eq!(c, s);
            }
        }
        let r = u.tensor(&format!("blk.{l}.ffn_gate_inp.weight")).unwrap();
        assert_eq!(r.shape, vec![4, 64]);
    }
    // Everything else is unchanged.
    let a = m.tensor("blk.1.attn_k.weight").unwrap();
    let b = u.tensor("blk.1.attn_k.weight").unwrap();
    assert_eq!(m.tensor_bytes(a).unwrap(), u.tensor_bytes(b).unwrap());

    // Deterministic for a seed; different seeds give different routers.
    let again = dir.path().join("again.gguf");
    upcycle(&m, &ir, &opts, &again).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&again).unwrap());
    let other = dir.path().join("other.gguf");
    upcycle(
        &m,
        &ir,
        &UpcycleOptions {
            seed: 1,
            ..opts.clone()
        },
        &other,
    )
    .unwrap();
    assert_ne!(std::fs::read(&out).unwrap(), std::fs::read(&other).unwrap());

    // Refuses its own output, bad parameters, and models that already have experts.
    assert!(upcycle(&m, &ir, &opts, &out).is_err());
    let bad = UpcycleOptions {
        top_k: 5,
        overwrite: true,
        ..opts.clone()
    };
    assert!(upcycle(&m, &ir, &bad, &other).is_err());
    let e = upcycle(
        &u,
        &uir,
        &UpcycleOptions {
            overwrite: true,
            ..opts
        },
        &other,
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("experts") || e.to_string().contains("ffn_gate"),
        "{e}"
    );
}

/// The llama fixture's tensors under `qwen3` metadata.
fn as_qwen3(dir: &std::path::Path) -> std::path::PathBuf {
    let (m, ir) = open(&mb_fixtures::gguf_llama(dir));
    let Metadata::Gguf { kv, .. } = &ir.raw.metadata else {
        unreachable!()
    };
    let kv: Vec<(String, MetaValue)> = kv
        .iter()
        .map(|(k, v)| match (k.as_str(), k.strip_prefix("llama.")) {
            ("general.architecture", _) => (k.clone(), MetaValue::String("qwen3".into())),
            (_, Some(rest)) => (format!("qwen3.{rest}"), v.clone()),
            _ => (k.clone(), v.clone()),
        })
        .collect();
    let tensors: Vec<TensorToWrite> = ir
        .raw
        .tensors
        .iter()
        .map(|t| TensorToWrite {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: Cow::Borrowed(m.tensor_bytes(t).unwrap()),
        })
        .collect();
    let p = dir.join("tiny-qwen3.gguf");
    gguf::write(&p, &kv, &tensors).unwrap();
    p
}

#[test]
fn qwen3_becomes_qwen3moe() {
    let dir = tempfile::tempdir().unwrap();
    let (m, ir) = open(&as_qwen3(dir.path()));
    let out = dir.path().join("moe.gguf");
    upcycle(&m, &ir, &UpcycleOptions::default(), &out).unwrap();
    let (_, uir) = open(&out);
    assert_eq!(
        meta(&uir, "general.architecture").unwrap().as_str(),
        Some("qwen3moe")
    );
    assert_eq!(
        meta(&uir, "qwen3moe.block_count").unwrap().as_u64(),
        Some(2)
    );
    assert_eq!(
        meta(&uir, "qwen3moe.expert_count").unwrap().as_u64(),
        Some(8)
    );
    assert_eq!(
        meta(&uir, "qwen3moe.expert_feed_forward_length")
            .unwrap()
            .as_u64(),
        Some(128)
    );
    assert!(meta(&uir, "qwen3.block_count").is_none());
    assert!(uir.layers.iter().all(|l| matches!(l.ffn, FfnSpec::Moe(_))));
}

#[test]
fn hybrid_is_refused_with_a_reason() {
    let dir = tempfile::tempdir().unwrap();
    let (m, ir) = open(&mb_fixtures::gguf_hybrid_ternary(dir.path()));
    let e = upcycle(
        &m,
        &ir,
        &UpcycleOptions::default(),
        &dir.path().join("x.gguf"),
    )
    .unwrap_err();
    assert!(e.to_string().contains("qwen35moe"), "{e}");
}

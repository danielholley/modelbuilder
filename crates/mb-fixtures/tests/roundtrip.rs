//! Format round trips: read a fixture, write it back out, and check that the
//! tensor index, data and metadata survive unchanged.

use std::borrow::Cow;
use std::collections::BTreeMap;

use mb_formats::{gguf, safetensors, TensorToWrite};
use mb_ir::Metadata;

fn rewrite(model: &mb_formats::LoadedModel) -> Vec<TensorToWrite<'_>> {
    model
        .raw
        .tensors
        .iter()
        .map(|t| TensorToWrite {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: Cow::Borrowed(model.tensor_bytes(t).unwrap()),
        })
        .collect()
}

fn assert_same_tensors(a: &mb_formats::LoadedModel, b: &mb_formats::LoadedModel) {
    let mut ta: Vec<_> = a.raw.tensors.iter().collect();
    let mut tb: Vec<_> = b.raw.tensors.iter().collect();
    ta.sort_by(|x, y| x.name.cmp(&y.name));
    tb.sort_by(|x, y| x.name.cmp(&y.name));
    assert_eq!(ta.len(), tb.len());
    for (x, y) in ta.iter().zip(&tb) {
        assert_eq!((&x.name, x.dtype, &x.shape), (&y.name, y.dtype, &y.shape));
        assert_eq!(
            a.tensor_bytes(x).unwrap(),
            b.tensor_bytes(y).unwrap(),
            "{}",
            x.name
        );
    }
}

#[test]
fn safetensors_sharded_to_single_file() {
    let dir = tempfile::tempdir().unwrap();
    let src = mb_formats::open(&mb_fixtures::qwen_hybrid(&dir.path().join("src"))).unwrap();
    let out = dir.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::copy(src.raw.root.join("config.json"), out.join("config.json")).unwrap();
    safetensors::write(
        &out.join("model.safetensors"),
        &rewrite(&src),
        &BTreeMap::new(),
    )
    .unwrap();

    let dst = mb_formats::open(&out).unwrap();
    assert_same_tensors(&src, &dst);
    let (Metadata::Hf { config: a, .. }, Metadata::Hf { config: b, .. }) =
        (&src.raw.metadata, &dst.raw.metadata)
    else {
        panic!()
    };
    assert_eq!(a, b);
}

#[test]
fn gguf_is_byte_identical_after_rewrite() {
    let dir = tempfile::tempdir().unwrap();
    assert_gguf_round_trip(&mb_fixtures::gguf_mixed_quant(dir.path()));
    assert_gguf_round_trip(&mb_fixtures::gguf_bonsai_like(dir.path()));
}

fn assert_gguf_round_trip(path: &std::path::Path) {
    let dir = tempfile::tempdir().unwrap();
    let src = mb_formats::open(path).unwrap();
    let Metadata::Gguf { kv, .. } = &src.raw.metadata else {
        panic!()
    };
    let out = dir.path().join("rewritten.gguf");
    gguf::write(&out, kv, &rewrite(&src)).unwrap();
    assert_eq!(std::fs::read(path).unwrap(), std::fs::read(&out).unwrap());
}

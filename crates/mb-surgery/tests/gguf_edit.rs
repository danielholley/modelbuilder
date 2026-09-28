//! Depth pruning and YaRN metadata surgery on the fixtures.

use mb_ir::{MetaValue, Metadata, ModelIr};
use mb_surgery::gguf_edit::{prune_layers, set_yarn};

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
fn prune_keeps_the_hybrid_pattern_and_the_rotation_list() {
    let dir = tempfile::tempdir().unwrap();
    let (m, ir) = open(&mb_fixtures::gguf_hybrid_ternary(dir.path()));
    let out = dir.path().join("p.gguf");
    // Period-4 pattern (3 DeltaNet + 1 attention): removing 2 layers would break it.
    let e = prune_layers(&m, &ir, 2, 2, &out, false).unwrap_err();
    assert!(e.to_string().contains("pattern"), "{e}");

    prune_layers(&m, &ir, 3, 4, &out, false).unwrap();
    let (p, pir) = open(&out);
    assert_eq!(pir.layers.len(), 4);
    let kinds = |ir: &ModelIr| {
        ir.layers
            .iter()
            .map(|l| std::mem::discriminant(&l.mixer))
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds(&pir), kinds(&ir)[..4]);
    // Old blk.7 is the new blk.3, bytes unchanged.
    let old = m.tensor("blk.7.attn_k.weight").unwrap();
    let new = p.tensor("blk.3.attn_k.weight").unwrap();
    assert_eq!(m.tensor_bytes(old).unwrap(), p.tensor_bytes(new).unwrap());
    // Every rotated-weight name still points at a tensor.
    let names = meta(&pir, "prism.hadamard.weight_names")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(!names.is_empty());
    for n in names {
        assert!(p.tensor(n.as_str().unwrap()).is_some(), "{n:?}");
    }
    assert_eq!(pir.weight_rotation.is_some(), ir.weight_rotation.is_some());

    // export-hf drops the pruned layers from the reference config too.
    let reference = mb_fixtures::hybrid_mtp_reference(&dir.path().join("ref"));
    let hf = dir.path().join("hf");
    let opts = mb_surgery::hf_export::HfExportOptions {
        reference,
        dtype: mb_ir::DType::F32,
        shard_bytes: 1 << 30,
        layers: None,
        globals: true,
        overwrite: false,
    };
    mb_surgery::hf_export::export_hf(&p, &pir, &hf, &opts).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(hf.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["num_hidden_layers"], 4);
    if let Some(types) = cfg["layer_types"].as_array() {
        assert_eq!(types.len(), 4);
    }
}

#[test]
fn yarn_sets_the_rope_scaling_keys() {
    let dir = tempfile::tempdir().unwrap();
    let (m, ir) = open(&mb_fixtures::gguf_llama(dir.path()));
    let out = dir.path().join("y.gguf");
    assert!(set_yarn(&m, &ir, 1.0, None, &out, false).is_err());
    set_yarn(&m, &ir, 4.0, None, &out, false).unwrap();
    let (_, y) = open(&out);
    assert_eq!(
        meta(&y, "llama.rope.scaling.type").unwrap().as_str(),
        Some("yarn")
    );
    let orig = meta(&ir, "llama.context_length").unwrap().as_u64().unwrap();
    assert_eq!(
        meta(&y, "llama.context_length").unwrap().as_u64(),
        Some(4 * orig)
    );
    assert!(y.rope.scaling.is_some(), "the IR sees the scaling");
}

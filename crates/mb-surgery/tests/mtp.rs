use mb_formats::dequant::dequantize;
use mb_ir::{Metadata, ModelIr};
use mb_surgery::mtp::{port_mtp_sidecar, MtpSidecarOptions};

fn open(p: &std::path::Path) -> (mb_formats::LoadedModel, ModelIr) {
    let m = mb_formats::open(p).unwrap();
    let ir = ModelIr::from_raw(m.raw.clone());
    (m, ir)
}

fn decode(m: &mb_formats::LoadedModel, name: &str) -> Vec<f32> {
    let t = m.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
    let mut v = Vec::new();
    dequantize(t.dtype, m.tensor_bytes(t).unwrap(), &mut v).unwrap();
    v
}

fn kv<'a>(ir: &'a ModelIr, key: &str) -> Option<&'a mb_ir::MetaValue> {
    let Metadata::Gguf { kv, .. } = &ir.raw.metadata else {
        panic!()
    };
    kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

#[test]
fn ports_the_head_into_an_mtp_only_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let (target, target_ir) = open(&mb_fixtures::gguf_bonsai_like(&dir.path().join("b")));
    let (reference, reference_ir) = open(&mb_fixtures::qwen_hybrid_matching_bonsai_like(
        &dir.path().join("q"),
    ));
    let out = dir.path().join("bonsai-like-mtp.gguf");
    let report = port_mtp_sidecar(
        &target,
        &target_ir,
        &reference,
        &reference_ir,
        &out,
        &MtpSidecarOptions::default(),
    )
    .unwrap();
    assert_eq!(report.tensors.len(), 3 + 15);

    let (side, side_ir) = open(&out);
    // The IR sees an MTP module at layer 8 and no trunk layers.
    let mtp = side_ir.mtp.as_ref().expect("MTP detected");
    assert_eq!((mtp.num_modules, mtp.tensor_count), (1, 15));
    assert_eq!(
        kv(&side_ir, "qwen35.block_count").unwrap(),
        &mb_ir::MetaValue::U32(9)
    );
    assert_eq!(
        kv(&side_ir, "qwen35.nextn_predict_layers").unwrap(),
        &mb_ir::MetaValue::U32(1)
    );
    assert!(
        side.tensor("blk.0.attn_norm.weight").is_none(),
        "MTP-only: no trunk"
    );

    // Matrices are the reference's BF16 bytes, untouched.
    let fc = reference.tensor("mtp.fc.weight").unwrap();
    let eh = side.tensor("blk.8.nextn.eh_proj.weight").unwrap();
    assert_eq!(
        side.tensor_bytes(eh).unwrap(),
        reference.tensor_bytes(fc).unwrap()
    );
    // Norms are +1 in F32.
    let src = decode(&reference, "mtp.layers.0.self_attn.q_norm.weight");
    let dst = decode(&side, "blk.8.attn_q_norm.weight");
    assert!(src
        .iter()
        .zip(&dst)
        .all(|(a, b)| (a + 1.0 - b).abs() < 1e-6));
    // Target tensors are copied byte for byte.
    for n in ["token_embd.weight", "output_norm.weight", "output.weight"] {
        assert_eq!(
            side.tensor_bytes(side.tensor(n).unwrap()).unwrap(),
            target.tensor_bytes(target.tensor(n).unwrap()).unwrap()
        );
    }
    // Rotation lists only name tensors present in the sidecar.
    let rot = side_ir.weight_rotation.as_ref().unwrap();
    assert_eq!((rot.rotated_tensors, rot.inverse_tensors), (1, 1));
    assert!(rot.is_rotated("output.weight") && !rot.is_rotated("blk.8.attn_q.weight"));
    assert_eq!(
        kv(&side_ir, "modelbuilder.surgery").unwrap().as_str(),
        Some("mtp-port")
    );
}

#[test]
fn refuses_unsafe_or_mismatched_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let target_path = mb_fixtures::gguf_bonsai_like(&dir.path().join("b"));
    let (target, target_ir) = open(&target_path);
    let (good, good_ir) = open(&mb_fixtures::qwen_hybrid_matching_bonsai_like(
        &dir.path().join("q"),
    ));
    let (small, small_ir) = open(&mb_fixtures::qwen_hybrid(&dir.path().join("s")));
    let opts = MtpSidecarOptions::default();

    // Writing over an input is refused, even with overwrite.
    let force = MtpSidecarOptions {
        overwrite: true,
        ..Default::default()
    };
    let e =
        port_mtp_sidecar(&target, &target_ir, &good, &good_ir, &target_path, &force).unwrap_err();
    assert!(e.to_string().contains("one of the input files"), "{e}");

    // Existing output without overwrite.
    let out = dir.path().join("out.gguf");
    std::fs::write(&out, b"x").unwrap();
    assert!(
        port_mtp_sidecar(&target, &target_ir, &good, &good_ir, &out, &opts)
            .unwrap_err()
            .to_string()
            .contains("exists")
    );
    assert!(port_mtp_sidecar(&target, &target_ir, &good, &good_ir, &out, &force).is_ok());

    // Shapes that don't match the target's attention blocks.
    let other = dir.path().join("other.gguf");
    let e = port_mtp_sidecar(&target, &target_ir, &small, &small_ir, &other, &opts).unwrap_err();
    assert!(e.to_string().contains("shape"), "{e}");
    assert!(!other.exists(), "nothing written on failure");

    // A sidecar already has MTP; a llama target has the wrong architecture.
    let (side, side_ir) = open(&out);
    assert!(port_mtp_sidecar(
        &side,
        &side_ir,
        &good,
        &good_ir,
        &dir.path().join("x.gguf"),
        &opts
    )
    .is_err());
    let (llama, llama_ir) = open(&mb_fixtures::gguf_mixed_quant(&dir.path().join("l")));
    let e = port_mtp_sidecar(
        &llama,
        &llama_ir,
        &good,
        &good_ir,
        &dir.path().join("y.gguf"),
        &opts,
    )
    .unwrap_err();
    assert!(e.to_string().contains("architecture"), "{e}");
}

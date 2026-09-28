use mb_features::estimate::Fit;
use mb_features::{Context, Detection};
use mb_ir::ModelIr;
use mb_plan::{parse_feature_spec, plan, resolve_hardware, FeaturePlan};

fn load(path: &std::path::Path) -> ModelIr {
    ModelIr::from_raw(mb_formats::open(path).unwrap().raw)
}

fn find<'a>(p: &'a mb_plan::Plan, id: &str) -> &'a FeaturePlan {
    p.features.iter().find(|f| f.id == id).unwrap()
}

#[test]
fn whole_catalog_on_hybrid_ternary() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::gguf_hybrid_ternary(dir.path()));
    let ctx = Context::new(&ir, None);
    let hw = resolve_hardware(&["1x24GB".into(), "8xH100".into()]).unwrap();
    let p = plan(&ctx, &[], &hw).unwrap();
    assert_eq!(p.features.len(), 3);

    // FP4 KV: 2 global layers × 2 × 2 kv heads × 32 dims = 256 elements/token.
    let fp4 = find(&p, "fp4-kv");
    assert!(fp4.compat.ok());
    let e = fp4.estimate.as_ref().unwrap();
    assert_eq!((e.effects[0].before, e.effects[0].after), (512.0, 144.0));
    assert_eq!(e.stages.len(), 1, "QAT by default");
    // Trainable: K and V (64 × 128 each) and q/k norms (32 each) in 2 layers.
    assert_eq!(e.stages[0].trainable_params, 2 * (2 * 64 * 128 + 2 * 32));
    assert_eq!(fp4.compute.len(), 2);
    assert!(fp4.compute.iter().all(|c| c.fits == Fit::Yes));

    // kv-share (default group 4 > 2 layers) is blocked; nothing is estimated.
    let share = find(&p, "kv-share");
    assert!(!share.compat.ok());
    assert!(share.estimate.is_none() && share.compute.is_empty());
    assert!(share.compat.blockers[0].contains("exceeds"));

    // MTP: absent, no reference given, head estimated from a full-attention block.
    let mtp = find(&p, "mtp");
    assert!(matches!(mtp.detection, Detection::Absent));
    assert!(mtp.compat.ok() && mtp.compat.warnings.iter().any(|w| w.contains("fresh head")));
    assert!(mtp
        .surgery
        .iter()
        .any(|s| s.contains("blk.8.nextn.eh_proj")));
    assert!(
        mtp.surgery.iter().any(|s| s.contains("unrotated")),
        "rotation handled: {:?}",
        mtp.surgery
    );
    let st = &mtp.estimate.as_ref().unwrap().stages[0];
    assert_eq!(
        st.trainable_params, st.backprop_params,
        "gradients stop at the head"
    );
}

#[test]
fn kv_share_with_group_two() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::gguf_hybrid_ternary(dir.path()));
    let ctx = Context::new(&ir, None);
    let hw = resolve_hardware(&["8xH100".into()]).unwrap();
    let p = plan(&ctx, &[parse_feature_spec("kv-share:group=2")], &hw).unwrap();
    let f = &p.features[0];
    assert!(f.compat.ok(), "{:?}", f.compat);
    assert!(f.compat.warnings.iter().any(|w| w.contains("layers apart")));
    let e = f.estimate.as_ref().unwrap();
    assert_eq!((e.effects[0].before, e.effects[0].after), (512.0, 256.0));
    assert!(f.surgery[0].contains("[3]") && f.surgery[1].contains("[7]"));
    assert_eq!(e.stages.len(), 2);
}

#[test]
fn mtp_detected_and_reference_checks() {
    let dir = tempfile::tempdir().unwrap();
    let qwen = load(&mb_fixtures::qwen_hybrid(&dir.path().join("q")));
    let target = load(&mb_fixtures::gguf_hybrid_ternary(&dir.path().join("b")));
    let hw = resolve_hardware(&["1x24GB".into()]).unwrap();
    let mtp = [parse_feature_spec("mtp")];

    let ctx = Context::new(&qwen, None);
    let p = plan(&ctx, &mtp, &hw).unwrap();
    assert!(matches!(p.features[0].detection, Detection::Present(_)));

    // Porting from a reference with different shapes is blocked.
    let ctx = Context::new(&target, Some(&qwen));
    let p = plan(&ctx, &mtp, &hw).unwrap();
    let blockers = &p.features[0].compat.blockers;
    assert!(
        blockers.iter().any(|b| b.contains("hidden/vocab differ")),
        "{blockers:?}"
    );

    // A `from` path that wasn't loaded is a blocker, not a panic.
    let ctx = Context::new(&target, None);
    let p = plan(&ctx, &[parse_feature_spec("mtp:from=/nowhere")], &hw).unwrap();
    assert!(p.features[0].compat.blockers[0].contains("was not loaded"));
}

#[test]
fn mla_model_and_bad_params() {
    let dir = tempfile::tempdir().unwrap();
    let ir = load(&mb_fixtures::deepseek_mla_moe(dir.path()));
    let ctx = Context::new(&ir, None);
    let hw = resolve_hardware(&["1x24GB".into()]).unwrap();
    let p = plan(
        &ctx,
        &[
            parse_feature_spec("fp4-kv:mode=ptq"),
            parse_feature_spec("kv-share"),
        ],
        &hw,
    )
    .unwrap();
    let fp4 = &p.features[0];
    assert!(fp4.compat.ok());
    let e = fp4.estimate.as_ref().unwrap();
    assert!(e.stages.is_empty(), "PTQ trains nothing");
    assert_eq!(fp4.compute[0].fits, Fit::NotApplicable);
    // MLA layers are not "global GQA" layers: nothing to share.
    assert!(p.features[1].compat.blockers[0].contains("at least 2"));

    assert!(plan(&ctx, &[parse_feature_spec("fp4-kv:mode=bogus")], &hw).is_err());
    assert!(
        plan(&ctx, &[parse_feature_spec("kv-share:grop=2")], &hw).is_err(),
        "unknown keys are rejected"
    );
    assert!(plan(&ctx, &[parse_feature_spec("nope")], &hw).is_err());
}

#[test]
fn example_recipes_parse() {
    // The generic example, and the task-specific one kept with the runbooks.
    for text in [
        include_str!("../../../examples/recipes/kv-and-mtp.toml"),
        include_str!("../../../docs/runbooks/bonsai2-kv-and-mtp.toml"),
    ] {
        example_recipe_is_valid(text);
    }
}

fn example_recipe_is_valid(text: &str) {
    let r = mb_plan::Recipe::parse(text).unwrap();
    assert_eq!(
        r.features.iter().map(|f| f.id.as_str()).collect::<Vec<_>>(),
        ["fp4-kv", "kv-share", "mtp"]
    );
    resolve_hardware(&r.hardware_ids()).unwrap();
    for f in &r.features {
        mb_features::feature(&f.id).unwrap();
    }
}

//! The Rust types must accept exactly what the schema examples contain.

use std::path::{Path, PathBuf};

use mb_jobs::{mtp_align_spec, Device, EventRecord, Hyper, JobSpec, MtpAlignInputs};

fn schema_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schema")
}

#[test]
fn example_spec_round_trips() {
    let text = std::fs::read_to_string(schema_dir().join("examples/mtp-align.job.json")).unwrap();
    let spec: JobSpec = serde_json::from_str(&text).unwrap();
    spec.validate().unwrap();
    let a: serde_json::Value = serde_json::from_str(&text).unwrap();
    let b = serde_json::to_value(&spec).unwrap();
    assert_eq!(a, b, "serializing must reproduce the example exactly");
}

#[test]
fn example_events_parse() {
    let text = std::fs::read_to_string(schema_dir().join("examples/events.jsonl")).unwrap();
    let events: Vec<EventRecord> = text
        .lines()
        .map(|l| EventRecord::parse(l).unwrap())
        .collect();
    assert_eq!(events.len(), 6);
    assert!(EventRecord::parse(
        r#"{"schema_version": 2, "event": "error", "time": 0, "message": "x"}"#
    )
    .is_err());
    assert!(EventRecord::parse(r#"{"schema_version": 1, "event": "bogus", "time": 0}"#).is_err());
}

#[test]
fn unknown_fields_and_bad_values_are_rejected() {
    let text = std::fs::read_to_string(schema_dir().join("examples/mtp-align.job.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["extra"] = 1.into();
    assert!(serde_json::from_value::<JobSpec>(v).is_err());

    let mut spec: JobSpec = serde_json::from_str(&text).unwrap();
    spec.stages[0].hyper.lr = 0.0;
    assert!(spec.validate().is_err());
    let mut spec: JobSpec = serde_json::from_str(&text).unwrap();
    spec.stages[0].mtp_align = None;
    assert!(spec.validate().is_err());
}

#[test]
fn builder_output_is_valid_and_matches_the_example_shape() {
    let dir = tempfile_dir();
    let spec = mtp_align_spec(MtpAlignInputs {
        job_id: "j".into(),
        output_dir: dir.join("out"),
        reference_config: "ref/config.json".into(),
        init_head: Some("ref".into()),
        frozen_tensors: dir.join("frozen.safetensors"),
        embedding_tensor: "token_embd.weight".into(),
        lm_head_tensor: "output.weight".into(),
        features: "feat".into(),
        hardware_profile: Some("1x24GB".into()),
        device: Device::Auto,
        hyper: Hyper::mtp_align_default(),
    });
    let path = dir.join("job.json");
    spec.write(&path).unwrap();
    let back = JobSpec::read(&path).unwrap();
    assert_eq!(back, spec);
    // Same keys as the schema example, so the Python side reads it the same way.
    let example: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(schema_dir().join("examples/mtp-align.job.json")).unwrap(),
    )
    .unwrap();
    let ours = serde_json::to_value(&spec).unwrap();
    let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(keys(&ours), keys(&example));
    assert_eq!(
        keys(&ours["stages"][0]["mtp_align"]),
        keys(&example["stages"][0]["mtp_align"])
    );
    assert_eq!(
        keys(&ours["stages"][0]["hyper"]),
        keys(&example["stages"][0]["hyper"])
    );
}

fn tempfile_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("mb-jobs-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

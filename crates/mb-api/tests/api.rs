use std::io::Write;
use std::path::Path;
use std::time::Duration;

use mb_api::fs::{EntryKind, ListRequest};
use mb_api::jobs::{Cursor, JobManager, JobSource, JobStatus};
use mb_api::{ApiError, InspectRequest, PlanRequest, StatsRequest};

const EVENTS: &str = include_str!("../../../schema/examples/events.jsonl");

fn s(p: &Path) -> String {
    p.display().to_string()
}

#[test]
fn inspect_stats_and_plan_a_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = s(&mb_fixtures::gguf_hybrid_ternary(dir.path()));

    let r = mb_api::inspect(&InspectRequest {
        path: path.clone(),
        tensors: true,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(r.report.architecture.family.as_deref(), Some("qwen35"));
    assert_eq!(r.layers.len(), r.report.architecture.num_layers);
    assert!(
        r.layers.iter().any(|l| l.label.starts_with("gqa")),
        "{:?}",
        r.layers[3].label
    );
    assert!(r
        .tensors
        .as_ref()
        .unwrap()
        .iter()
        .any(|t| t.info.name == "token_embd.weight"));

    let st = mb_api::stats(&StatsRequest {
        path: path.clone(),
        only: vec!["attn_q".into()],
        kv_spectra: false,
    })
    .unwrap();
    assert!(!st.tensors.is_empty());

    let plan = mb_api::plan(&PlanRequest {
        path: Some(path.clone()),
        features: vec!["kv-share:group=2".into()],
        hardware: vec!["1x24GB".into()],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(plan.features.len(), 1);
    assert_eq!(plan.hardware.len(), 1);

    // A recipe supplies the source and features; `path` overrides the source.
    let recipe = "[source]\npath = \"nowhere.gguf\"\n\n[[feature]]\nid = \"fp4-kv\"\n".to_string();
    let plan = mb_api::plan(&PlanRequest {
        path: Some(path),
        recipe: Some(recipe),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(plan.features[0].id, "fp4-kv");
}

#[test]
fn bad_inputs_are_bad_requests() {
    let e = mb_api::inspect(&InspectRequest {
        path: "/no/such/model.gguf".into(),
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(e, ApiError::BadRequest(_)), "{e:?}");
    let e = mb_api::plan(&PlanRequest::default()).unwrap_err();
    assert!(matches!(e, ApiError::BadRequest(_)), "{e:?}");
    let e = mb_api::plan(&PlanRequest {
        recipe: Some("not = [toml".into()),
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(e, ApiError::BadRequest(_)), "{e:?}");
}

#[test]
fn catalog_lists_features_and_hardware() {
    let c = mb_api::catalog();
    assert!(c.features.iter().any(|f| f.id == "mtp"));
    assert!(c.hardware.iter().any(|h| h.id == "1x24GB"));
}

#[test]
fn lists_directories_with_model_kinds() {
    let dir = tempfile::tempdir().unwrap();
    mb_fixtures::gguf_hybrid_ternary(&dir.path().join("g"));
    mb_fixtures::llama_gqa(&dir.path().join("hf"));
    std::fs::write(dir.path().join("recipe.toml"), "").unwrap();
    std::fs::write(dir.path().join(".hidden"), "").unwrap();

    let l = mb_api::fs::list(&ListRequest {
        path: s(dir.path()),
    })
    .unwrap();
    let kinds: Vec<_> = l
        .entries
        .iter()
        .map(|e| (e.name.as_str(), e.kind))
        .collect();
    assert_eq!(
        kinds,
        [
            ("g", EntryKind::Dir),
            ("hf", EntryKind::HfModel),
            ("recipe.toml", EntryKind::Recipe)
        ]
    );
    let g = mb_api::fs::list(&ListRequest {
        path: s(&dir.path().join("g")),
    })
    .unwrap();
    assert_eq!(g.entries[0].kind, EntryKind::Gguf);
    assert!(g.entries[0].size.unwrap() > 0);
}

fn wait_done(m: &JobManager, id: u32) -> mb_api::jobs::JobUpdate {
    let mut cursor = Cursor::default();
    let mut events = Vec::new();
    for _ in 0..100 {
        let u = m.wait(id, cursor, Duration::from_millis(200)).unwrap();
        events.extend(u.events.clone());
        cursor = u.cursor;
        if u.summary.status != JobStatus::Running {
            let mut u = u;
            u.events = events;
            return u;
        }
    }
    panic!("job {id} did not finish");
}

#[test]
fn follows_an_events_file_as_it_grows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let m = JobManager::new("python3");
    // Started before the file exists: it waits for it.
    let job = m
        .start(JobSource::Events {
            events_path: s(&path),
        })
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let mut f = std::fs::File::create(&path).unwrap();
    let lines: Vec<&str> = EVENTS.lines().collect();
    for l in &lines[..3] {
        writeln!(f, "{l}").unwrap();
    }
    f.flush().unwrap();
    let u = m
        .wait(job.id, Cursor::default(), Duration::from_secs(5))
        .unwrap();
    assert!(!u.events.is_empty());
    for l in &lines[3..] {
        writeln!(f, "{l}").unwrap();
    }
    let done = wait_done(&m, job.id);
    assert_eq!(done.summary.status, JobStatus::Succeeded);
    assert_eq!(done.summary.job_id.as_deref(), Some("mtp-align-example"));
    assert!(done.summary.latest_eval.is_some());
    // It stops at `finished`: the example's trailing `error` line is not read.
    let finished = lines
        .iter()
        .position(|l| l.contains("\"finished\""))
        .unwrap();
    assert_eq!(done.summary.events, finished + 1);
}

#[cfg(unix)]
fn fake_python(dir: &Path, body: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join("fake-python");
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    s(&p)
}

#[cfg(unix)]
#[test]
fn runs_a_spec_with_the_python_side_and_can_cancel() {
    let dir = tempfile::tempdir().unwrap();
    let spec = dir.path().join("job.json");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples/mtp-align.job.json"),
        &spec,
    )
    .unwrap();
    let events = dir.path().join("events.jsonl");
    std::fs::write(&events, EVENTS).unwrap();

    // Stands in for `python -m modelbuilder_train run <spec>`: replays the example events.
    let py = fake_python(
        dir.path(),
        &format!("echo starting >&2\ncat '{}'", events.display()),
    );
    let m = JobManager::new(py);
    let job = m
        .start(JobSource::Spec {
            spec_path: s(&spec),
            python: None,
        })
        .unwrap();
    assert_eq!(job.job_id.as_deref(), Some("mtp-align-example"));
    let done = wait_done(&m, job.id);
    assert_eq!(
        done.summary.status,
        JobStatus::Succeeded,
        "{:?}",
        done.summary.error
    );
    assert_eq!(
        done.summary.outputs.get("mtp_head").map(String::as_str),
        Some("runs/mtp-align-example/mtp-head")
    );
    let log = m.update(job.id, Cursor::default()).unwrap().log;
    assert!(log.iter().any(|l| l == "starting"), "{log:?}");

    // A job that never finishes, cancelled.
    let slow = fake_python(dir.path(), "sleep 30");
    let job = m
        .start(JobSource::Spec {
            spec_path: s(&spec),
            python: Some(slow),
        })
        .unwrap();
    let c = m.cancel(job.id).unwrap();
    assert_eq!(c.status, JobStatus::Cancelled);
    let done = wait_done(&m, job.id);
    assert_eq!(done.summary.status, JobStatus::Cancelled);
    assert_eq!(m.list().len(), 2);

    // A bad spec is refused up front.
    std::fs::write(dir.path().join("bad.json"), "{}").unwrap();
    let e = m
        .start(JobSource::Spec {
            spec_path: s(&dir.path().join("bad.json")),
            python: None,
        })
        .unwrap_err();
    assert!(matches!(e, ApiError::BadRequest(_)));
}

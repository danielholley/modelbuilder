use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use mb_server::{router, ServerConfig};
use serde_json::{json, Value};
use tower::ServiceExt;

fn app(web_dir: Option<std::path::PathBuf>) -> Router {
    router(&ServerConfig {
        web_dir,
        python: "python3".into(),
        extra_hosts: vec!["gpu-box".into()],
    })
}

async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    host: &str,
) -> (StatusCode, String) {
    let mut req = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::HOST, host);
    if body.is_some() {
        req = req.header(header::CONTENT_TYPE, "application/json");
    }
    let req = req
        .body(body.map_or(Body::empty(), |b| Body::from(b.to_string())))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn api(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let (s, text) = call(app, method, uri, body, "localhost:7878").await;
    (
        s,
        serde_json::from_str(&text).unwrap_or(Value::String(text)),
    )
}

#[tokio::test]
async fn serves_the_api_over_fixtures() {
    let dir = tempfile::tempdir().unwrap();
    let model = mb_fixtures::gguf_bonsai_like(dir.path())
        .display()
        .to_string();
    let app = app(None);

    let (s, v) = api(&app, "GET", "/api/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["web_ui"], false);

    let (s, v) = api(&app, "GET", "/api/catalog", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(v["features"].as_array().unwrap().len() >= 3);

    let (s, v) = api(
        &app,
        "POST",
        "/api/inspect",
        Some(json!({"path": model, "tensors": true})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["report"]["architecture"]["family"], "qwen35");
    assert!(v["tensors"].as_array().unwrap().len() > 10);

    let (s, v) = api(
        &app,
        "POST",
        "/api/stats",
        Some(json!({"path": model, "only": ["attn_k"], "kv_spectra": true})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(!v["kv_spectra"].as_array().unwrap().is_empty(), "{v}");

    let (s, v) = api(
        &app,
        "POST",
        "/api/plan",
        Some(json!({"path": model, "hardware": ["1x24GB"]})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["features"].as_array().unwrap().len() >= 3);

    let (s, v) = api(
        &app,
        "POST",
        "/api/fs/list",
        Some(json!({"path": dir.path()})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["entries"][0]["kind"], "gguf");

    // Errors are JSON with the right status.
    let (s, v) = api(
        &app,
        "POST",
        "/api/inspect",
        Some(json!({"path": "/nope.gguf"})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("nope"));
    let (s, _) = api(
        &app,
        "POST",
        "/api/inspect",
        Some(json!({"path": model, "bogus": 1})),
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _) = api(&app, "GET", "/api/jobs/99", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _) = api(&app, "GET", "/api/nothing", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn follows_a_job_and_streams_its_events() {
    let dir = tempfile::tempdir().unwrap();
    let events = dir.path().join("events.jsonl");
    std::fs::write(
        &events,
        include_str!("../../../schema/examples/events.jsonl"),
    )
    .unwrap();
    let app = app(None);

    let (s, job) = api(
        &app,
        "POST",
        "/api/jobs",
        Some(json!({"kind": "events", "events_path": events})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{job}");
    let id = job["id"].as_u64().unwrap();

    // The stream ends once the job has ended; the example file ends with `finished`.
    let (s, body) = call(
        &app,
        "GET",
        &format!("/api/jobs/{id}/stream"),
        None,
        "localhost",
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let updates: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let last = updates.last().unwrap();
    assert_eq!(last["summary"]["status"], "succeeded", "{body}");
    let n: usize = updates
        .iter()
        .map(|u| u["events"].as_array().unwrap().len())
        .sum();
    assert_eq!(n, 5, "each event is sent once");

    let (s, v) = api(
        &app,
        "GET",
        &format!("/api/jobs/{id}/updates?events=3&log=0&version=0"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["events"].as_array().unwrap().len(), 2);
    let (_, v) = api(&app, "GET", "/api/jobs", None).await;
    assert_eq!(v.as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn refuses_foreign_hosts_and_origins() {
    let app = app(None);
    let (s, _) = call(&app, "GET", "/api/health", None, "evil.example:7878").await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, _) = call(&app, "GET", "/api/health", None, "gpu-box:7878").await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = call(&app, "GET", "/api/health", None, "[::1]:7878").await;
    assert_eq!(s, StatusCode::OK);

    let req = |origin: &str| {
        Request::builder()
            .method("POST")
            .uri("/api/jobs")
            .header(header::HOST, "localhost:7878")
            .header(header::ORIGIN, origin)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"kind":"events","events_path":"/tmp/x"}"#))
            .unwrap()
    };
    let res = app
        .clone()
        .oneshot(req("https://evil.example"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let res = app
        .clone()
        .oneshot(req("http://localhost:5173"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn serves_the_web_build_with_spa_fallback() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("index.html"),
        "<!doctype html><title>ui</title>",
    )
    .unwrap();
    std::fs::create_dir(dir.path().join("assets")).unwrap();
    std::fs::write(dir.path().join("assets/app.js"), "console.log(1)").unwrap();
    let app = app(Some(dir.path().to_owned()));

    let (s, body) = call(&app, "GET", "/assets/app.js", None, "localhost").await;
    assert_eq!((s, body.as_str()), (StatusCode::OK, "console.log(1)"));
    let (s, body) = call(&app, "GET", "/plan", None, "localhost").await;
    assert_eq!(s, StatusCode::OK);
    assert!(body.contains("<title>ui</title>"));
    let (_, v) = api(&app, "GET", "/api/health", None).await;
    assert_eq!(v["web_ui"], true);

    // Without a build, `/` says how to make one.
    let (s, body) = call(&self::app(None), "GET", "/", None, "localhost").await;
    assert_eq!(s, StatusCode::OK);
    assert!(body.contains("npm run build"));
}

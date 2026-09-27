//! The local web API behind the React UI (`web/`), plus a static host for
//! its build. Every handler is a thin wrapper over `mb-api`.
//!
//! The server reads model files by path and starts training processes, so it
//! is meant for the machine it runs on: it binds to loopback by default and
//! rejects requests whose `Host` or `Origin` isn't an allowed host name, which
//! stops other web pages in the same browser (and DNS rebinding) from using it.

pub mod types;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::Stream;
use mb_api::jobs::{Cursor, JobId, JobManager, JobSource, JobStatus};
use mb_api::ApiError;
use serde::Serialize;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// The web UI build (`web/dist`); without it, `/` explains how to build it.
    pub web_dir: Option<PathBuf>,
    /// Default Python interpreter for training jobs.
    pub python: String,
    /// Host names accepted in `Host`/`Origin` besides localhost, 127.0.0.1 and ::1.
    pub extra_hosts: Vec<String>,
}

#[derive(Clone)]
struct AppState {
    jobs: JobManager,
    hosts: std::sync::Arc<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, ts_rs::TS)]
pub struct ErrorBody {
    pub error: String,
}

#[derive(Clone, Debug, Serialize, ts_rs::TS)]
pub struct Health {
    pub name: String,
    pub version: String,
    /// Whether the web UI build is being served.
    pub web_ui: bool,
}

struct HttpError(ApiError);

impl From<ApiError> for HttpError {
    fn from(e: ApiError) -> Self {
        Self(e)
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorBody {
                error: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

type ApiResult<T> = Result<Json<T>, HttpError>;

/// Runs a blocking `mb-api` call off the async runtime.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> mb_api::Result<T> + Send + 'static,
) -> ApiResult<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| HttpError(ApiError::Internal(e.to_string())))?
        .map(Json)
        .map_err(HttpError)
}

fn host_name(authority: &str) -> &str {
    if let Some(rest) = authority.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    authority.rsplit_once(':').map_or(authority, |(h, _)| h)
}

impl AppState {
    fn allowed(&self, authority: &str) -> bool {
        let h = host_name(authority).to_ascii_lowercase();
        ["localhost", "127.0.0.1", "::1"].contains(&h.as_str()) || self.hosts.contains(&h)
    }
}

async fn guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let headers: &HeaderMap = req.headers();
    let host_ok = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|h| st.allowed(h));
    let origin_ok = match headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        None => true,
        Some(o) => o
            .split_once("://")
            .is_some_and(|(_, rest)| st.allowed(rest.split('/').next().unwrap_or(rest))),
    };
    if !(host_ok && origin_ok) {
        let body = ErrorBody {
            error: "request refused: this server only accepts requests addressed to localhost (see `modelbuilder serve --allow-host`)".into(),
        };
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    }
    next.run(req).await
}

pub fn router(cfg: &ServerConfig) -> Router {
    let state = AppState {
        jobs: JobManager::new(cfg.python.clone()),
        hosts: std::sync::Arc::new(
            cfg.extra_hosts
                .iter()
                .map(|h| h.to_ascii_lowercase())
                .collect(),
        ),
    };
    let index = cfg
        .web_dir
        .as_ref()
        .map(|d| d.join("index.html"))
        .filter(|p| p.is_file());
    let web_ui = index.is_some();

    let api = Router::new()
        .route(
            "/health",
            get(move || async move {
                Json(Health {
                    name: "modelbuilder".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                    web_ui,
                })
            }),
        )
        .route("/catalog", get(|| async { Json(mb_api::catalog()) }))
        .route(
            "/inspect",
            post(|Json(r): Json<mb_api::InspectRequest>| blocking(move || mb_api::inspect(&r))),
        )
        .route(
            "/stats",
            post(|Json(r): Json<mb_api::StatsRequest>| blocking(move || mb_api::stats(&r))),
        )
        .route(
            "/plan",
            post(|Json(r): Json<mb_api::PlanRequest>| blocking(move || mb_api::plan(&r))),
        )
        .route(
            "/fs/list",
            post(|Json(r): Json<mb_api::fs::ListRequest>| blocking(move || mb_api::fs::list(&r))),
        )
        .route("/jobs", get(list_jobs).post(start_job))
        .route("/jobs/{id}", get(get_job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/jobs/{id}/updates", get(job_updates))
        .route("/jobs/{id}/stream", get(job_stream))
        .fallback(|| async { HttpError(ApiError::NotFound("no such API route".into())) });

    let app = Router::new().nest("/api", api);
    let app = match (index, &cfg.web_dir) {
        (Some(index), Some(dir)) => {
            app.fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)))
        }
        _ => app.fallback(|| async { Html(NO_UI) }),
    };
    app.layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

const NO_UI: &str = "<!doctype html><meta charset=utf-8><title>modelbuilder</title>\
<body style=\"font-family:system-ui;max-width:40rem;margin:4rem auto;line-height:1.5\">\
<h1>modelbuilder</h1><p>The API is running under <code>/api</code>, but the web UI hasn't been built.</p>\
<pre>cd web &amp;&amp; npm ci &amp;&amp; npm run build</pre>\
<p>then restart <code>modelbuilder serve</code> (or pass <code>--web-dir</code>).</p></body>";

async fn list_jobs(State(st): State<AppState>) -> Json<Vec<mb_api::jobs::JobSummary>> {
    Json(st.jobs.list())
}

async fn start_job(
    State(st): State<AppState>,
    Json(src): Json<JobSource>,
) -> ApiResult<mb_api::jobs::JobSummary> {
    blocking(move || st.jobs.start(src)).await
}

async fn get_job(
    State(st): State<AppState>,
    Path(id): Path<JobId>,
) -> ApiResult<mb_api::jobs::JobSummary> {
    Ok(Json(st.jobs.get(id)?))
}

async fn cancel_job(
    State(st): State<AppState>,
    Path(id): Path<JobId>,
) -> ApiResult<mb_api::jobs::JobSummary> {
    Ok(Json(st.jobs.cancel(id)?))
}

/// Everything since `?events=&log=&version=`, without waiting.
async fn job_updates(
    State(st): State<AppState>,
    Path(id): Path<JobId>,
    Query(since): Query<Cursor>,
) -> ApiResult<mb_api::jobs::JobUpdate> {
    Ok(Json(st.jobs.update(id, since)?))
}

/// Server-sent events: one `update` message (a `JobUpdate`) per change,
/// starting with the current state; the stream ends once the job has ended.
async fn job_stream(
    State(st): State<AppState>,
    Path(id): Path<JobId>,
    Query(since): Query<Cursor>,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, HttpError> {
    st.jobs.get(id)?;
    let jobs = st.jobs.clone();
    let stream =
        futures_util::stream::unfold((since, true, false), move |(cursor, first, done)| {
            let jobs = jobs.clone();
            async move {
                if done {
                    return None;
                }
                let update = tokio::task::spawn_blocking(move || {
                    if first {
                        jobs.update(id, cursor)
                    } else {
                        jobs.wait(id, cursor, Duration::from_secs(15))
                    }
                })
                .await
                .ok()?
                .ok()?;
                let ended = update.summary.status != JobStatus::Running;
                let event = if first || update.cursor != cursor {
                    SseEvent::default()
                        .event("update")
                        .json_data(&update)
                        .unwrap_or_else(|_| SseEvent::default().comment("serialization failed"))
                } else {
                    SseEvent::default().comment("no change")
                };
                Some((Ok(event), (update.cursor, false, ended)))
            }
        });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Binds `addr` and serves until Ctrl-C.
pub async fn serve(addr: SocketAddr, cfg: ServerConfig) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    let shown = if local.ip().is_unspecified() || local.ip().is_loopback() {
        format!("http://localhost:{}", local.port())
    } else {
        format!("http://{local}")
    };
    println!("modelbuilder dashboard: {shown}");
    if cfg
        .web_dir
        .as_ref()
        .is_none_or(|d| !d.join("index.html").is_file())
    {
        println!("  (web UI not built: cd web && npm ci && npm run build; the API is under /api)");
    }
    axum::serve(listener, router(&cfg))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

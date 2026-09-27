//! Training jobs as the dashboards see them.
//!
//! A job either runs a job spec with the Python side (`mb_jobs::JobProcess`)
//! or follows an events file that a job elsewhere is writing
//! (`python -m modelbuilder_train run job.json --events events.jsonl`), e.g. on
//! a GPU box reached over SSH. Either way the dashboards get the same thing:
//! a summary plus the event records, read incrementally with a [`Cursor`].

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mb_jobs::{Event, EventRecord, JobProcess, JobSpec};
use serde::{Deserialize, Serialize};

use crate::{ApiError, Result};

pub type JobId = u32;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum JobSource {
    /// Run a job spec (job.json) with the Python side.
    Spec {
        spec_path: String,
        /// Python interpreter with `modelbuilder_train` installed (default: the manager's).
        #[serde(default)]
        python: Option<String>,
    },
    /// Follow an events file (JSONL) written by a job running elsewhere.
    Events { events_path: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct JobSummary {
    pub id: JobId,
    pub source: JobSource,
    pub status: JobStatus,
    /// The spec's `job_id` (or the `started` event's, when following a file).
    pub job_id: Option<String>,
    /// Unix seconds.
    pub started_at: f64,
    pub finished_at: Option<f64>,
    pub events: usize,
    pub latest_progress: Option<EventRecord>,
    pub latest_eval: Option<EventRecord>,
    pub outputs: BTreeMap<String, String>,
    pub error: Option<String>,
}

/// How much of a job a client has seen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(default, deny_unknown_fields)]
pub struct Cursor {
    pub events: usize,
    pub log: usize,
    /// Bumped on every change to the job, including its status.
    pub version: u64,
}

/// What changed since a [`Cursor`].
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct JobUpdate {
    pub summary: JobSummary,
    /// Event records from `since.events` on.
    pub events: Vec<EventRecord>,
    /// Log lines (stderr and non-event stdout) from `since.log` on.
    pub log: Vec<String>,
    /// Pass this back to get the next update.
    pub cursor: Cursor,
}

struct Job {
    summary: JobSummary,
    events: Vec<EventRecord>,
    log: Vec<String>,
    version: u64,
    process: Option<Arc<JobProcess>>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    fn cursor(&self) -> Cursor {
        Cursor {
            events: self.events.len(),
            log: self.log.len(),
            version: self.version,
        }
    }

    fn update(&self, since: Cursor) -> JobUpdate {
        JobUpdate {
            summary: self.summary.clone(),
            events: self.events.get(since.events..).unwrap_or_default().to_vec(),
            log: self.log.get(since.log..).unwrap_or_default().to_vec(),
            cursor: self.cursor(),
        }
    }

    fn push_event(&mut self, rec: EventRecord) {
        match &rec.event {
            Event::Started { job_id, .. } if self.summary.job_id.is_none() => {
                self.summary.job_id = Some(job_id.clone());
            }
            Event::Progress { .. } => self.summary.latest_progress = Some(rec.clone()),
            Event::Eval { .. } => self.summary.latest_eval = Some(rec.clone()),
            Event::Error { message } => self.summary.error = Some(message.clone()),
            Event::Finished { outputs, .. } => self.summary.outputs = outputs.clone(),
            _ => {}
        }
        self.events.push(rec);
        self.summary.events = self.events.len();
    }

    fn finish(&mut self, status: JobStatus, error: Option<String>) {
        if self.summary.status == JobStatus::Running {
            self.summary.status = status;
            self.summary.finished_at = Some(now());
        }
        if error.is_some() && self.summary.error.is_none() {
            self.summary.error = error;
        }
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

struct Inner {
    jobs: Mutex<BTreeMap<JobId, Job>>,
    changed: Condvar,
    python: String,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, BTreeMap<JobId, Job>> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Applies `f` to job `id` and wakes waiters.
    fn with(&self, id: JobId, f: impl FnOnce(&mut Job)) {
        if let Some(job) = self.lock().get_mut(&id) {
            f(job);
            job.version += 1;
        }
        self.changed.notify_all();
    }
}

/// Owns the jobs started from a dashboard. Cheap to clone; clones share jobs.
#[derive(Clone)]
pub struct JobManager {
    inner: Arc<Inner>,
}

impl JobManager {
    /// `python` is the default interpreter for spec jobs.
    pub fn new(python: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                jobs: Mutex::new(BTreeMap::new()),
                changed: Condvar::new(),
                python: python.into(),
            }),
        }
    }

    pub fn start(&self, source: JobSource) -> Result<JobSummary> {
        let (job_id, process) = match &source {
            JobSource::Spec { spec_path, python } => {
                let path = Path::new(spec_path);
                let spec = JobSpec::read(path)
                    .map_err(|e| ApiError::BadRequest(format!("{spec_path}: {e}")))?;
                let python = python.as_deref().unwrap_or(&self.inner.python);
                let p = JobProcess::spawn_with(path, python, Stdio::piped())
                    .map_err(|e| ApiError::BadRequest(e.to_string()))?;
                (Some(spec.job_id), Some(Arc::new(p)))
            }
            JobSource::Events { events_path } => {
                if events_path.is_empty() {
                    return Err(ApiError::BadRequest("no events file given".into()));
                }
                (None, None)
            }
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let summary = {
            let mut jobs = self.inner.lock();
            let id = jobs.keys().next_back().map_or(1, |k| k + 1);
            let summary = JobSummary {
                id,
                source: source.clone(),
                status: JobStatus::Running,
                job_id,
                started_at: now(),
                finished_at: None,
                events: 0,
                latest_progress: None,
                latest_eval: None,
                outputs: BTreeMap::new(),
                error: None,
            };
            jobs.insert(
                id,
                Job {
                    summary: summary.clone(),
                    events: Vec::new(),
                    log: Vec::new(),
                    version: 0,
                    process: process.clone(),
                    cancel: cancel.clone(),
                },
            );
            summary
        };
        let id = summary.id;
        match (source, process) {
            (JobSource::Spec { .. }, Some(p)) => self.drive(id, p),
            (JobSource::Events { events_path }, _) => {
                self.follow(id, PathBuf::from(events_path), cancel)
            }
            _ => unreachable!("spec jobs always have a process"),
        }
        Ok(summary)
    }

    fn drive(&self, id: JobId, p: Arc<JobProcess>) {
        let stderr = p.take_stderr().map(|stderr| {
            let inner = self.inner.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(|l| l.ok()) {
                    inner.with(id, |j| j.log.push(line));
                }
            })
        });
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let result = p.drive(
                |rec| inner.with(id, |j| j.push_event(rec.clone())),
                |line| inner.with(id, |j| j.log.push(line.to_string())),
            );
            // The process has exited, so stderr is at EOF: take its last lines before
            // the job reads as finished.
            if let Some(h) = stderr {
                let _ = h.join();
            }
            inner.with(id, |j| match result {
                Ok(_) => j.finish(JobStatus::Succeeded, None),
                Err(e) => j.finish(JobStatus::Failed, Some(e.to_string())),
            });
        });
    }

    fn follow(&self, id: JobId, path: PathBuf, cancel: Arc<AtomicBool>) {
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let poll = Duration::from_millis(300);
            // The file may not exist yet if the job hasn't started.
            let file = loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                match std::fs::File::open(&path) {
                    Ok(f) => break f,
                    Err(_) => std::thread::sleep(poll),
                }
            };
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                match reader.read_line(&mut line) {
                    Ok(0) => std::thread::sleep(poll),
                    // A partial line: the writer hasn't finished it yet; read the rest next time.
                    Ok(_) if !line.ends_with('\n') => std::thread::sleep(poll),
                    Ok(_) => {
                        let text = line.trim_end();
                        let mut done = None;
                        match EventRecord::parse(text) {
                            Ok(rec) => {
                                if let Event::Finished { status, .. } = &rec.event {
                                    done = Some(if status == "ok" {
                                        JobStatus::Succeeded
                                    } else {
                                        JobStatus::Failed
                                    });
                                }
                                inner.with(id, |j| j.push_event(rec));
                            }
                            Err(_) if !text.is_empty() => {
                                let text = text.to_string();
                                inner.with(id, |j| j.log.push(text));
                            }
                            Err(_) => {}
                        }
                        line.clear();
                        if let Some(status) = done {
                            inner.with(id, |j| j.finish(status, None));
                            return;
                        }
                    }
                    Err(e) => {
                        inner.with(id, |j| {
                            j.finish(JobStatus::Failed, Some(format!("{}: {e}", path.display())))
                        });
                        return;
                    }
                }
            }
        });
    }

    pub fn list(&self) -> Vec<JobSummary> {
        self.inner
            .lock()
            .values()
            .map(|j| j.summary.clone())
            .collect()
    }

    pub fn get(&self, id: JobId) -> Result<JobSummary> {
        self.inner
            .lock()
            .get(&id)
            .map(|j| j.summary.clone())
            .ok_or_else(|| not_found(id))
    }

    /// Kills a spec job's process, or stops following an events file.
    pub fn cancel(&self, id: JobId) -> Result<JobSummary> {
        let process = {
            let jobs = self.inner.lock();
            let job = jobs.get(&id).ok_or_else(|| not_found(id))?;
            job.cancel.store(true, Ordering::Relaxed);
            job.process.clone()
        };
        // Mark it first, so the driver thread's "failed" (the process was killed) doesn't win.
        self.inner
            .with(id, |j| j.finish(JobStatus::Cancelled, None));
        if let Some(p) = process {
            p.kill();
        }
        self.get(id)
    }

    /// Everything since `since`, without waiting.
    pub fn update(&self, id: JobId, since: Cursor) -> Result<JobUpdate> {
        self.inner
            .lock()
            .get(&id)
            .map(|j| j.update(since))
            .ok_or_else(|| not_found(id))
    }

    /// Waits until job `id` changes after `since` (or `timeout` passes), then
    /// returns what changed. Blocks the calling thread.
    pub fn wait(&self, id: JobId, since: Cursor, timeout: Duration) -> Result<JobUpdate> {
        let deadline = Instant::now() + timeout;
        let mut jobs = self.inner.lock();
        loop {
            let job = jobs.get(&id).ok_or_else(|| not_found(id))?;
            let left = deadline.saturating_duration_since(Instant::now());
            if job.version != since.version || left.is_zero() {
                return Ok(job.update(since));
            }
            jobs = self
                .inner
                .changed
                .wait_timeout(jobs, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

fn not_found(id: JobId) -> ApiError {
    ApiError::NotFound(format!("no job {id}"))
}

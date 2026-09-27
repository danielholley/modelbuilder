//! Training jobs: the Rust side of the Rust ↔ Python contract.
//!
//! The contract lives in `schema/` (`job-spec.v1.schema.json`,
//! `events.v1.schema.json`). These types mirror it and reject unknown fields;
//! tests round-trip the files in `schema/examples/`, which the Python tests
//! validate too.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("invalid job spec: {0}")]
    Spec(String),
    #[error("could not start `{python}`: {source}. Install the Python side (python/, `pip install -e python[torch]`) or pass --python")]
    Launch {
        python: String,
        #[source]
        source: std::io::Error,
    },
    #[error("training failed: {0}")]
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub schema_version: u32,
    pub job_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    pub backend: Backend,
    pub device: Device,
    #[serde(default)]
    pub hardware_profile: Option<String>,
    pub output_dir: String,
    pub stages: Vec<Stage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Torch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    Auto,
    Cpu,
    Cuda,
    Mps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    MtpAlign,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub name: String,
    pub kind: StageKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp_align: Option<MtpAlign>,
    pub hyper: Hyper,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtpAlign {
    pub reference_config: String,
    #[serde(default)]
    pub init_head: Option<String>,
    pub frozen_tensors: String,
    pub embedding_tensor: String,
    pub lm_head_tensor: String,
    pub features: String,
    #[serde(default = "default_eval_fraction")]
    pub eval_fraction: f64,
}

fn default_eval_fraction() -> f64 {
    0.05
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainDtype {
    Float32,
    Bfloat16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hyper {
    pub lr: f64,
    pub steps: u64,
    pub seq_len: u64,
    pub batch_seqs: u64,
    #[serde(default)]
    pub warmup_steps: u64,
    #[serde(default)]
    pub weight_decay: f64,
    #[serde(default = "default_grad_clip")]
    pub grad_clip: Option<f64>,
    #[serde(default = "default_log_every")]
    pub log_every: u64,
    #[serde(default = "default_eval_every")]
    pub eval_every: u64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_dtype")]
    pub dtype: TrainDtype,
}

fn default_grad_clip() -> Option<f64> {
    Some(1.0)
}
fn default_log_every() -> u64 {
    10
}
fn default_eval_every() -> u64 {
    100
}
fn default_dtype() -> TrainDtype {
    TrainDtype::Float32
}

impl JobSpec {
    /// Checks what the schema checks and serde can't express.
    pub fn validate(&self) -> Result<(), JobError> {
        let bad = |m: String| Err(JobError::Spec(m));
        if self.schema_version != SCHEMA_VERSION {
            return bad(format!(
                "schema_version {} (expected {SCHEMA_VERSION})",
                self.schema_version
            ));
        }
        if self.stages.is_empty() {
            return bad("no stages".into());
        }
        for s in &self.stages {
            let h = &s.hyper;
            if h.lr.is_nan()
                || h.lr <= 0.0
                || h.steps == 0
                || h.seq_len < 4
                || h.batch_seqs == 0
                || h.log_every == 0
                || h.eval_every == 0
            {
                return bad(format!("stage {}: hyperparameters out of range", s.name));
            }
            match (s.kind, &s.mtp_align) {
                (StageKind::MtpAlign, None) => {
                    return bad(format!(
                        "stage {}: kind mtp_align needs an mtp_align table",
                        s.name
                    ))
                }
                (StageKind::MtpAlign, Some(m)) if !(0.0..=0.5).contains(&m.eval_fraction) => {
                    return bad(format!(
                        "stage {}: eval_fraction must be in [0, 0.5]",
                        s.name
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn write(&self, path: &Path) -> Result<(), JobError> {
        self.validate()?;
        std::fs::write(
            path,
            serde_json::to_string_pretty(self).expect("spec serializes") + "\n",
        )?;
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Self, JobError> {
        let spec: Self = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| JobError::Spec(e.to_string()))?;
        spec.validate()?;
        Ok(spec)
    }
}

/// A progress event from the Python side (`schema/events.v1.schema.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Started {
        job_id: String,
        device: String,
        #[serde(default)]
        trainable_params: Option<u64>,
        #[serde(default)]
        train_tokens: Option<u64>,
        #[serde(default)]
        eval_tokens: Option<u64>,
    },
    Progress {
        step: u64,
        steps: u64,
        #[serde(default)]
        tokens: Option<u64>,
        loss: f64,
        #[serde(default)]
        accuracy: Option<f64>,
        #[serde(default)]
        lr: Option<f64>,
        #[serde(default)]
        tokens_per_s: Option<f64>,
    },
    Eval {
        step: u64,
        loss: f64,
        accuracy: f64,
        #[serde(default)]
        tokens: Option<u64>,
    },
    Checkpoint {
        step: u64,
        path: String,
    },
    Finished {
        status: String,
        #[serde(default)]
        outputs: std::collections::BTreeMap<String, String>,
    },
    Error {
        message: String,
    },
}

/// An event with its envelope fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub schema_version: u32,
    pub time: f64,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(flatten)]
    pub event: Event,
}

impl EventRecord {
    pub fn parse(line: &str) -> Result<Self, JobError> {
        let r: Self = serde_json::from_str(line)
            .map_err(|e| JobError::Spec(format!("bad event line: {e}")))?;
        if r.schema_version != SCHEMA_VERSION {
            return Err(JobError::Spec(format!(
                "event schema_version {}",
                r.schema_version
            )));
        }
        Ok(r)
    }
}

/// Runs `python -m modelbuilder_train run <spec>`, calling `on_event` for each
/// event line as it arrives. Other stdout lines are passed to `on_other`;
/// stderr goes straight to the terminal. Returns the `finished` outputs.
pub fn run(
    spec_path: &Path,
    python: &str,
    mut on_event: impl FnMut(&EventRecord),
    mut on_other: impl FnMut(&str),
) -> Result<std::collections::BTreeMap<String, String>, JobError> {
    let mut child = Command::new(python)
        .args(["-m", "modelbuilder_train", "run"])
        .arg(spec_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|source| JobError::Launch {
            python: python.to_string(),
            source,
        })?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let mut finished = None;
    let mut last_error = None;
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        match EventRecord::parse(&line) {
            Ok(rec) => {
                match &rec.event {
                    Event::Finished { status, outputs } => {
                        finished = Some((status.clone(), outputs.clone()))
                    }
                    Event::Error { message } => last_error = Some(message.clone()),
                    _ => {}
                }
                on_event(&rec);
            }
            Err(_) => on_other(&line),
        }
    }
    let status = child.wait()?;
    match finished {
        Some((s, outputs)) if s == "ok" && status.success() => Ok(outputs),
        _ => Err(JobError::Failed(last_error.unwrap_or_else(|| {
            format!("the Python side exited with {status}")
        }))),
    }
}

/// Inputs for an `mtp_align` job.
#[derive(Clone, Debug)]
pub struct MtpAlignInputs {
    pub job_id: String,
    pub output_dir: PathBuf,
    pub reference_config: PathBuf,
    pub init_head: Option<PathBuf>,
    pub frozen_tensors: PathBuf,
    pub embedding_tensor: String,
    pub lm_head_tensor: String,
    pub features: PathBuf,
    pub hardware_profile: Option<String>,
    pub device: Device,
    pub hyper: Hyper,
}

impl Hyper {
    /// Defaults for aligning a ported MTP head: short warmup, modest LR.
    pub fn mtp_align_default() -> Self {
        Self {
            lr: 1e-4,
            steps: 2000,
            seq_len: 1024,
            batch_seqs: 4,
            warmup_steps: 100,
            weight_decay: 0.0,
            grad_clip: Some(1.0),
            log_every: 10,
            eval_every: 200,
            seed: 0,
            dtype: TrainDtype::Bfloat16,
        }
    }
}

pub fn mtp_align_spec(i: MtpAlignInputs) -> JobSpec {
    let s = |p: &Path| p.display().to_string();
    JobSpec {
        schema_version: SCHEMA_VERSION,
        job_id: i.job_id,
        created_by: Some(format!("modelbuilder {}", env!("CARGO_PKG_VERSION"))),
        backend: Backend::Torch,
        device: i.device,
        hardware_profile: i.hardware_profile,
        output_dir: s(&i.output_dir),
        stages: vec![Stage {
            name: "mtp-align".into(),
            kind: StageKind::MtpAlign,
            mtp_align: Some(MtpAlign {
                reference_config: s(&i.reference_config),
                init_head: i.init_head.as_deref().map(s),
                frozen_tensors: s(&i.frozen_tensors),
                embedding_tensor: i.embedding_tensor,
                lm_head_tensor: i.lm_head_tensor,
                features: s(&i.features),
                eval_fraction: 0.05,
            }),
            hyper: i.hyper,
        }],
    }
}

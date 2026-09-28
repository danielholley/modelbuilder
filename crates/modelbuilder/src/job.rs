//! `modelbuilder job`: prepare, run and finish training jobs.
//!
//! Rust does everything around the training loop (decoding the frozen
//! tensors, writing the job spec, turning the trained head into a GGUF
//! sidecar); the Python side (`python/modelbuilder_train`) runs the loop and
//! reports JSONL events, which are rendered here as they arrive.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use mb_ir::ModelIr;
use mb_jobs::{Device, Event, EventRecord, Hyper, JobSpec, MtpAlignInputs, TrainDtype};

use crate::render;

#[derive(Subcommand)]
pub enum JobOp {
    /// Train (align) an MTP head against a frozen target, then write the GGUF
    /// sidecar. Starts from the reference model's head; trunk features come
    /// from `modelbuilder-train extract-features`.
    MtpAlign(Box<MtpAlignArgs>),
    /// Run an existing job spec (job.json) and render its events.
    Run {
        spec: PathBuf,
        #[command(flatten)]
        py: PythonArgs,
    },
}

#[derive(Args)]
pub struct PythonArgs {
    /// Python interpreter with `modelbuilder_train` installed.
    #[arg(long, default_value = "python3", env = "MODELBUILDER_PYTHON")]
    python: String,
    /// Print raw JSONL events instead of the rendered view.
    #[arg(long)]
    raw_events: bool,
}

#[derive(Args)]
pub struct MtpAlignArgs {
    /// Target model (.gguf).
    target: PathBuf,
    /// Reference HF model directory with `mtp.*` tensors (config and initial head).
    #[arg(long)]
    from: PathBuf,
    /// Trunk feature directory (manifest.json + shards) computed from the target.
    #[arg(long)]
    features: PathBuf,
    /// Run directory: frozen tensors, job.json, the trained head and the sidecar go here.
    #[arg(long, short = 'o')]
    out: PathBuf,
    /// Train the head from random initialization instead of the reference's head.
    #[arg(long)]
    from_scratch: bool,
    #[arg(long, default_value = "auto")]
    device: String,
    /// Hardware profile to record in the spec (see `modelbuilder features`).
    #[arg(long)]
    hardware: Option<String>,
    #[arg(long)]
    steps: Option<u64>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    seq_len: Option<u64>,
    #[arg(long)]
    batch_seqs: Option<u64>,
    #[arg(long)]
    warmup_steps: Option<u64>,
    #[arg(long)]
    eval_every: Option<u64>,
    #[arg(long)]
    log_every: Option<u64>,
    /// Train in float32 instead of bfloat16 autocast.
    #[arg(long)]
    fp32: bool,
    /// Write the frozen tensors and job.json, then stop (e.g. to train on another machine).
    #[arg(long)]
    emit_only: bool,
    /// Replace existing outputs in the run directory.
    #[arg(long)]
    force: bool,
    #[command(flatten)]
    py: PythonArgs,
}

const FROZEN: &str = "frozen.safetensors";
const EMBED: &str = "token_embd.weight";
const OUTPUT: &str = "output.weight";

pub fn run(op: JobOp) -> Result<()> {
    match op {
        JobOp::Run { spec, py } => {
            JobSpec::read(&spec).with_context(|| format!("reading {}", spec.display()))?;
            let outputs = launch(&spec, &py)?;
            print_outputs(&outputs);
            Ok(())
        }
        JobOp::MtpAlign(a) => mtp_align(*a),
    }
}

fn open(p: &Path) -> Result<(mb_formats::LoadedModel, ModelIr)> {
    let m = mb_formats::open(p).with_context(|| format!("opening {}", p.display()))?;
    let ir = ModelIr::from_raw(m.raw.clone());
    Ok((m, ir))
}

fn absolute(p: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(p).with_context(|| format!("{} not found", p.display()))
}

fn mtp_align(a: MtpAlignArgs) -> Result<()> {
    let (target, target_ir) = open(&a.target)?;
    let reference_config = a.from.join("config.json");
    if !reference_config.is_file() {
        bail!("{} has no config.json", a.from.display());
    }
    if !has_manifest(&a.features) {
        bail!(
            "{} is not a feature directory (no manifest.json); create one with \
             `python -m modelbuilder_train extract-features`",
            a.features.display()
        );
    }
    std::fs::create_dir_all(&a.out)?;
    let out = absolute(&a.out)?;

    // Frozen pieces for the loss: decoded and in the primal basis.
    let lm_head = if target.tensor(OUTPUT).is_some() {
        OUTPUT
    } else {
        EMBED
    };
    let mut names = vec![EMBED.to_string()];
    if lm_head != EMBED {
        names.push(lm_head.to_string());
    }
    let frozen = out.join(FROZEN);
    if frozen.exists() && !a.force {
        println!(
            "using existing {} (pass --force to re-export)",
            frozen.display()
        );
    } else {
        let opts = mb_surgery::export::ExportOptions {
            names,
            dtype: mb_ir::DType::Bf16,
            overwrite: true,
        };
        let report = mb_surgery::export::export_primal(&target, &target_ir, &frozen, &opts)?;
        println!(
            "exported {} ({}) to {}",
            report
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if target_ir.weight_rotation.is_some() {
                "rotation undone"
            } else {
                "decoded"
            },
            frozen.display()
        );
    }

    let mut hyper = Hyper::mtp_align_default();
    macro_rules! set {
        ($($f:ident),*) => { $(if let Some(v) = a.$f { hyper.$f = v; })* };
    }
    set!(
        steps,
        lr,
        seq_len,
        batch_seqs,
        warmup_steps,
        eval_every,
        log_every
    );
    hyper.warmup_steps = hyper.warmup_steps.min(hyper.steps / 2);
    hyper.eval_every = hyper.eval_every.min(hyper.steps);
    if a.fp32 {
        hyper.dtype = TrainDtype::Float32;
    }
    let device = match a.device.as_str() {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        "mps" => Device::Mps,
        d => bail!("unknown device {d}; use auto, cpu, cuda or mps"),
    };
    let job_id = format!(
        "mtp-align-{}",
        a.target
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    );
    let spec = mb_jobs::mtp_align_spec(MtpAlignInputs {
        job_id,
        // Paths inside the run directory stay relative, so it can be moved.
        output_dir: ".".into(),
        reference_config: absolute(&reference_config)?,
        init_head: if a.from_scratch {
            None
        } else {
            Some(absolute(&a.from)?)
        },
        frozen_tensors: FROZEN.into(),
        embedding_tensor: EMBED.into(),
        lm_head_tensor: lm_head.into(),
        features: absolute(&a.features)?,
        hardware_profile: a.hardware,
        device,
        hyper,
    });
    let spec_path = out.join("job.json");
    spec.write(&spec_path)?;
    println!("wrote {}", spec_path.display());
    if a.emit_only {
        println!(
            "\nRun it with:\n  modelbuilder job run {}\nthen write the sidecar with:\n  \
             modelbuilder surgery mtp {} --from {} -o <name>-mtp.gguf",
            spec_path.display(),
            a.target.display(),
            out.join("mtp-head").display()
        );
        return Ok(());
    }

    let outputs = launch(&spec_path, &a.py)?;
    print_outputs(&outputs);
    let head = outputs
        .get("mtp_head")
        .map(PathBuf::from)
        .context("the job reported no mtp_head output")?;

    let (r, r_ir) = open(&head)?;
    let stem = a
        .target
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let sidecar = out.join(format!("{stem}-mtp-aligned.gguf"));
    let opts = mb_surgery::mtp::MtpSidecarOptions {
        overwrite: a.force,
        reference_label: Some(format!("aligned:{}", a.from.display())),
        aligned_to_target: true,
    };
    let report =
        mb_surgery::mtp::port_mtp_sidecar(&target, &target_ir, &r, &r_ir, &sidecar, &opts)?;
    print!("\n{}", render::surgery(&report));
    Ok(())
}

fn launch(spec: &Path, py: &PythonArgs) -> Result<std::collections::BTreeMap<String, String>> {
    let raw = py.raw_events;
    let outputs = mb_jobs::run(
        spec,
        &py.python,
        |rec| {
            if raw {
                println!("{}", serde_json::to_string(rec).unwrap_or_default());
            } else if let Some(line) = render_event(rec) {
                println!("{line}");
            }
        },
        |other| println!("  | {other}"),
    )?;
    Ok(outputs)
}

fn print_outputs(outputs: &std::collections::BTreeMap<String, String>) {
    for (k, v) in outputs {
        println!("output {k}: {v}");
    }
}

/// One line per event, or `None` for events not worth a line.
pub fn render_event(rec: &EventRecord) -> Option<String> {
    let stage = rec
        .stage
        .as_deref()
        .map(|s| format!("[{s}] "))
        .unwrap_or_default();
    let pct = |x: f64| format!("{:.1}%", 100.0 * x);
    let line = match &rec.event {
        Event::Started {
            job_id,
            device,
            trainable_params,
            train_tokens,
            eval_tokens,
        } => format!(
            "{stage}started {job_id} on {device}: {} trainable params, {} train / {} eval tokens",
            trainable_params.map_or("?".into(), render::count),
            train_tokens.map_or("?".into(), render::count),
            eval_tokens.map_or("?".into(), render::count),
        ),
        Event::Progress {
            step,
            steps,
            loss,
            accuracy,
            lr,
            tokens_per_s,
            ..
        } => format!(
            "{stage}step {step:>6}/{steps}  loss {loss:.4}  acc {}  lr {}  {} tok/s",
            accuracy.map_or("-".into(), pct),
            lr.map_or("-".into(), |l| format!("{l:.2e}")),
            tokens_per_s.map_or("-".into(), |t| format!("{t:.0}")),
        ),
        Event::Eval {
            step,
            loss,
            accuracy,
            tokens,
        } => format!(
            "{stage}eval @ {step:>6}  loss {loss:.4}  top-1 {}  ({} tokens)",
            pct(*accuracy),
            tokens.map_or("?".into(), render::count),
        ),
        Event::Checkpoint { step, path } => format!("{stage}checkpoint @ {step}: {path}"),
        Event::Finished { status, .. } => format!("finished: {status}"),
        Event::Error { message } => format!("{stage}error: {message}"),
    };
    Some(line)
}

/// A feature directory, or a directory of them (one per `extract-features --shard`).
fn has_manifest(dir: &Path) -> bool {
    dir.join("manifest.json").is_file()
        || std::fs::read_dir(dir).is_ok_and(|rd| {
            rd.flatten()
                .any(|e| e.path().join("manifest.json").is_file())
        })
}

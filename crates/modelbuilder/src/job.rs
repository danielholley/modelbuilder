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
use mb_jobs::{
    Device, Event, EventRecord, Hyper, JobSpec, KvFormat, MtpAlignInputs, TrainDtype,
    TrunkDistillInputs,
};

use crate::render;

#[derive(Subcommand)]
pub enum JobOp {
    /// Train (align) an MTP head against a frozen target, then write the GGUF
    /// sidecar. Starts from the reference model's head; trunk features come
    /// from `modelbuilder-train extract-features`.
    MtpAlign(Box<MtpAlignArgs>),
    /// Retrain some trunk tensors so the model tolerates a change (a quantized
    /// or shared KV cache), distilling from the unmodified model, then write
    /// them back into a copy of the GGUF in their original types. Exports the
    /// target as an HF checkpoint first (`export-hf`).
    TrunkDistill(Box<TrunkDistillArgs>),
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

#[derive(Args)]
pub struct TrunkDistillArgs {
    /// Target model (.gguf).
    target: PathBuf,
    /// Reference HF model directory (config.json, tokenizer) with the same architecture.
    #[arg(long)]
    reference: PathBuf,
    /// Training text: JSONL with `text` or `tokens` per line.
    #[arg(long)]
    texts: PathBuf,
    /// Run directory: the HF export, job.json, the updates and the new GGUF go here.
    #[arg(long, short = 'o')]
    out: PathBuf,
    /// Substrings of HF parameter names to train.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "k_proj.weight,v_proj.weight"
    )]
    trainable: Vec<String>,
    /// KV-cache format to train for: q8_0, q4_0 or nvfp4.
    #[arg(long)]
    kv_format: Option<String>,
    /// Share each attention layer's KV cache with the next `N - 1` attention layers.
    #[arg(long)]
    kv_share_group: Option<u32>,
    /// Train in full precision instead of through the source's weight format.
    /// The write-back then re-quantizes with a larger error.
    #[arg(long)]
    no_weight_fakequant: bool,
    /// Refuse the write-back if any tensor's re-quantization error (relative RMS) exceeds this.
    #[arg(long, default_value_t = 1e-3)]
    max_rel_error: f64,
    #[arg(long, default_value = "auto")]
    device: String,
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
    /// Train in float32 (e.g. GPUs without fast bf16).
    #[arg(long)]
    fp32: bool,
    /// Write the HF export and job.json, then stop.
    #[arg(long)]
    emit_only: bool,
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
        JobOp::TrunkDistill(a) => trunk_distill(*a),
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
    let device = parse_device(&a.device)?;
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

fn parse_device(d: &str) -> Result<Device> {
    Ok(match d {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        "mps" => Device::Mps,
        d => bail!("unknown device {d}; use auto, cpu, cuda or mps"),
    })
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn trunk_distill(a: TrunkDistillArgs) -> Result<()> {
    let (target, target_ir) = open(&a.target)?;
    let kv_format = match a.kv_format.as_deref() {
        None => None,
        Some("q8_0") => Some(KvFormat::Q8_0),
        Some("q4_0") => Some(KvFormat::Q4_0),
        Some("nvfp4") => Some(KvFormat::Nvfp4),
        Some(f) => bail!("unknown KV format {f}; use q8_0, q4_0 or nvfp4"),
    };
    if kv_format.is_none() && a.kv_share_group.is_none() && a.no_weight_fakequant {
        bail!("nothing to distill for: pass --kv-format and/or --kv-share-group");
    }
    std::fs::create_dir_all(&a.out)?;
    let out = absolute(&a.out)?;

    // The trunk in PyTorch's layout; float32 so training starts from the exact decoded values.
    let hf = out.join("hf");
    if hf.join("config.json").is_file() && !a.force {
        println!(
            "using existing {} (pass --force to re-export)",
            hf.display()
        );
    } else {
        let opts = mb_surgery::hf_export::HfExportOptions {
            reference: a.reference.clone(),
            dtype: mb_ir::DType::F32,
            shard_bytes: 5 << 30,
            layers: None,
            globals: true,
            overwrite: true,
        };
        let report = mb_surgery::hf_export::export_hf(&target, &target_ir, &hf, &opts)?;
        println!(
            "exported {} tensors to {}",
            report.tensors.len(),
            hf.display()
        );
    }

    let mut hyper = Hyper::trunk_distill_default();
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
    let name = match (kv_format, a.kv_share_group) {
        (Some(_), Some(_)) => "kv-quant-share-qat",
        (Some(_), None) => "kv-quant-qat",
        (None, Some(_)) => "kv-share-qat",
        (None, None) => "weight-qat",
    };
    let spec = mb_jobs::trunk_distill_spec(TrunkDistillInputs {
        job_id: format!("{name}-{}", stem(&a.target)),
        stage_name: name.into(),
        output_dir: ".".into(),
        model: "hf".into(),
        texts: absolute(&a.texts)?,
        trainable: a.trainable,
        kv_format,
        kv_share_group: a.kv_share_group,
        weight_fakequant: !a.no_weight_fakequant,
        hardware_profile: a.hardware,
        device: parse_device(&a.device)?,
        hyper,
    });
    let spec_path = out.join("job.json");
    spec.write(&spec_path)?;
    println!("wrote {}", spec_path.display());
    let new = out.join(format!("{}-{name}.gguf", stem(&a.target)));
    if a.emit_only {
        println!(
            "\nRun it with:\n  modelbuilder job run {}\nthen write the tensors back with:\n  \
             modelbuilder surgery replace {} --updates {} -o {}",
            spec_path.display(),
            a.target.display(),
            out.join("updates").display(),
            new.display()
        );
        return Ok(());
    }

    let outputs = launch(&spec_path, &a.py)?;
    print_outputs(&outputs);
    let updates = outputs
        .get("updates")
        .map(PathBuf::from)
        .context("the job reported no updates output")?;
    let (u, _) = open(&updates)?;
    let opts = mb_surgery::replace::ReplaceOptions {
        config: updates.join("config.json"),
        overwrite: a.force,
        max_rel_error: Some(a.max_rel_error),
    };
    let report = mb_surgery::replace::replace_tensors(&target, &target_ir, &u, &new, &opts)?;
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

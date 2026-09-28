use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mb_ir::ModelIr;

mod job;
mod render;

#[derive(Parser)]
#[command(
    name = "modelbuilder",
    version,
    about = "Inspect, plan and rebuild LLM checkpoints"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Break down a checkpoint: architecture, parameters, quantization, KV cache, provenance.
    Inspect {
        /// A .gguf file or an HF directory (config.json + .safetensors).
        path: PathBuf,
        /// Print the full report as JSON.
        #[arg(long)]
        json: bool,
        /// Also list every tensor.
        #[arg(long)]
        tensors: bool,
        /// Context length for KV-cache totals (defaults to the model's max positions).
        #[arg(long)]
        context: Option<u64>,
    },
    /// Stream the weights and compute statistics: norms, outliers, sparsity,
    /// ternary structure, and (with --kv-spectra) K/V singular-value spectra.
    Stats {
        /// A .gguf file or an HF directory (config.json + .safetensors).
        path: PathBuf,
        /// Only tensors whose name contains one of these (comma-separated).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Compute K/V spectra for each attention layer (the rank evidence for
        /// KV compression / MLA conversion).
        #[arg(long)]
        kv_spectra: bool,
        /// How many tensors to list in the outlier rankings.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Print the full report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Evaluate features against a model: already present? compatible? what
    /// it changes, what training it needs, and what that costs per hardware
    /// profile. With no --feature, evaluates the whole catalog.
    Plan {
        /// The model (optional with --recipe, which names its source).
        path: Option<PathBuf>,
        /// A recipe file (TOML) listing source, hardware and features.
        #[arg(long)]
        recipe: Option<PathBuf>,
        /// Feature spec, `id` or `id:key=value,...` (repeatable), e.g.
        /// `kv-share:group=2` or `mtp:from=models/base-model`.
        #[arg(long = "feature", short = 'f')]
        features: Vec<String>,
        /// Hardware profiles (comma-separated); default: all.
        #[arg(long, value_delimiter = ',')]
        hardware: Vec<String>,
        /// Print the plan as JSON.
        #[arg(long)]
        json: bool,
    },
    /// List the feature catalog and hardware profiles.
    Features,
    /// Export tensors decoded and un-rotated (primal basis) to safetensors,
    /// e.g. a low-bit model's embedding and output head for PyTorch training.
    ExportTensors {
        model: PathBuf,
        /// Tensor names (comma-separated), e.g. token_embd.weight,output.weight.
        #[arg(long, value_delimiter = ',', required = true)]
        names: Vec<String>,
        /// Output dtype: bf16 or f32.
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long, short = 'o')]
        out: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Export a GGUF as a Hugging Face checkpoint for PyTorch: decoded,
    /// un-rotated, with llama.cpp's converter transforms undone (architecture
    /// adapters: llama, qwen2, qwen3, qwen35). Needs a reference HF config.json.
    ExportHf {
        model: PathBuf,
        /// Reference HF model directory (config.json, tokenizer) with the same architecture.
        #[arg(long)]
        reference: PathBuf,
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// bf16 or f32 for the large weights.
        #[arg(long, default_value = "bf16")]
        dtype: String,
        /// Shard size in GiB.
        #[arg(long, default_value_t = 5.0)]
        shard_gib: f64,
        /// Only these decoder layers, `start..end` (for validation).
        #[arg(long)]
        layers: Option<String>,
        /// With --layers: leave out the embedding, final norm and LM head.
        #[arg(long)]
        no_globals: bool,
        #[arg(long)]
        force: bool,
    },
    /// Write a new checkpoint with a feature added. Inputs are never modified.
    Surgery {
        #[command(subcommand)]
        op: SurgeryOp,
    },
    /// Training jobs: prepare a job spec, run it on the Python side with live
    /// progress, and turn the result into a checkpoint.
    Job {
        #[command(subcommand)]
        op: job::JobOp,
    },
    /// Start the web dashboard (and its JSON API) on this machine.
    Serve {
        #[arg(long, default_value_t = 7878)]
        port: u16,
        /// Address to bind. The server reads files and starts training jobs:
        /// keep it on loopback unless the network is yours (then use an SSH tunnel if you can).
        #[arg(long, default_value = "127.0.0.1")]
        host: std::net::IpAddr,
        /// The web UI build (default: web/dist of this checkout, or $MODELBUILDER_WEB_DIR).
        #[arg(long, env = "MODELBUILDER_WEB_DIR")]
        web_dir: Option<PathBuf>,
        /// Python interpreter with `modelbuilder_train` installed, for training jobs.
        #[arg(long, default_value = "python3", env = "MODELBUILDER_PYTHON")]
        python: String,
        /// Extra host names to accept in Host/Origin headers (repeatable), e.g. a tunnel's.
        #[arg(long = "allow-host")]
        allow_hosts: Vec<String>,
    },
    /// Terminal dashboard: a model's breakdown, its plan, and a live training job.
    Tui {
        /// A .gguf file or HF directory.
        model: Option<PathBuf>,
        /// Run this job spec and watch it.
        #[arg(long, conflicts_with = "events")]
        job: Option<PathBuf>,
        /// Follow a job running elsewhere through its events file
        /// (`python -m modelbuilder_train run job.json --events FILE`).
        #[arg(long)]
        events: Option<PathBuf>,
        /// Hardware profiles for the plan (comma-separated); default: all.
        #[arg(long, value_delimiter = ',')]
        hardware: Vec<String>,
        #[arg(long, default_value = "python3", env = "MODELBUILDER_PYTHON")]
        python: String,
    },
    /// Write a tiny synthetic checkpoint, for trying the tool without a real model.
    #[command(hide = true)]
    Fixture { kind: FixtureKind, out: PathBuf },
}

#[derive(Subcommand)]
enum SurgeryOp {
    /// Write trained tensors (HF names, primal basis, e.g. from a QAT stage)
    /// back into a copy of a GGUF, re-rotated and re-encoded to each tensor's
    /// original type. Everything else is copied byte for byte.
    Replace {
        /// The GGUF to start from.
        target: PathBuf,
        /// HF-layout directory with the updated tensors (and a config.json).
        #[arg(long)]
        updates: PathBuf,
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Architecture config (default: the updates directory's config.json).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Refuse if any tensor's re-quantization error exceeds this (relative RMS, e.g. 0.01).
        #[arg(long)]
        max_rel_error: Option<f64>,
        #[arg(long)]
        force: bool,
    },
    /// Port an MTP head from a Hugging Face reference model into an MTP-only
    /// GGUF sidecar for a GGUF target whose architecture llama.cpp runs with
    /// nextn layers (run it with `-md <sidecar> --spec-type draft-mtp`).
    Mtp {
        /// Target model (.gguf).
        target: PathBuf,
        /// Reference HF model directory with `mtp.*` tensors (e.g. the target's base model).
        #[arg(long)]
        from: PathBuf,
        /// Output sidecar path. Put "mtp" in the name so the fork can find it next to the model.
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Replace the output if it exists.
        #[arg(long)]
        force: bool,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum FixtureKind {
    LlamaGqa,
    QwenHybrid,
    DeepseekMlaMoe,
    GgufMixedQuant,
    GgufHybridTernary,
    HybridMtpReference,
    GgufLlama,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect {
            path,
            json,
            tensors,
            context,
        } => {
            let model =
                mb_formats::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let ir = ModelIr::from_raw(model.raw);
            let report = mb_analyze::analyze(&ir, context);
            if json {
                let mut value = serde_json::to_value(&report)?;
                if tensors {
                    value["tensors"] = serde_json::to_value(
                        ir.tensors()
                            .map(|(t, r)| serde_json::json!({ "info": t, "role": r }))
                            .collect::<Vec<_>>(),
                    )?;
                }
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                print!("{}", render::report(&report));
                if tensors {
                    print!("{}", render::tensors(&ir));
                }
            }
        }
        Command::Stats {
            path,
            only,
            kv_spectra,
            top,
            json,
        } => {
            let model =
                mb_formats::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let ir = ModelIr::from_raw(model.raw.clone());
            let opts = mb_analyze::weights::WeightStatsOptions { only, kv_spectra };
            let report = mb_analyze::weights::weight_stats(&model, &ir, &opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", render::weight_stats(&report, &ir, top));
            }
        }
        Command::Plan {
            path,
            recipe,
            features,
            hardware,
            json,
        } => {
            let (source, mut requests, mut hw_ids) = match &recipe {
                Some(r) => {
                    let text = std::fs::read_to_string(r)
                        .with_context(|| format!("reading {}", r.display()))?;
                    let recipe = mb_plan::Recipe::parse(&text)?;
                    let hw = recipe.hardware_ids();
                    (PathBuf::from(&recipe.source.path), recipe.features, hw)
                }
                None => (
                    path.clone().context("give a model path or --recipe")?,
                    Vec::new(),
                    Vec::new(),
                ),
            };
            let source = path.unwrap_or(source);
            requests.extend(features.iter().map(|f| mb_plan::parse_feature_spec(f)));
            if !hardware.is_empty() {
                hw_ids = hardware;
            }
            let hw = mb_plan::resolve_hardware(&hw_ids)?;

            let open_ir = |p: &std::path::Path| -> Result<ModelIr> {
                let m = mb_formats::open(p).with_context(|| format!("opening {}", p.display()))?;
                Ok(ModelIr::from_raw(m.raw))
            };
            let ir = open_ir(&source)?;
            // A feature's `from` names a reference model (e.g. the base to port an MTP head from).
            let reference_path = requests.iter().find_map(|r| {
                r.params
                    .get("from")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
            });
            let reference = reference_path.as_deref().map(open_ir).transpose()?;
            let ctx = mb_features::Context::new(&ir, reference.as_ref());
            let plan = mb_plan::plan(&ctx, &requests, &hw)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                print!("{}", render::plan(&plan));
            }
        }
        Command::Features => {
            println!("FEATURES");
            for f in mb_features::catalog() {
                println!(
                    "  {:<10} {}\n             {}",
                    f.id(),
                    f.title(),
                    f.summary()
                );
            }
            println!("\nHARDWARE PROFILES");
            for p in mb_features::hardware::profiles() {
                println!("  {:<13} {}", p.id, p.description);
            }
        }
        Command::Surgery {
            op:
                SurgeryOp::Replace {
                    target,
                    updates,
                    out,
                    config,
                    max_rel_error,
                    force,
                },
        } => {
            let t = mb_formats::open(&target)
                .with_context(|| format!("opening {}", target.display()))?;
            let t_ir = ModelIr::from_raw(t.raw.clone());
            let u = mb_formats::open(&updates)
                .with_context(|| format!("opening {}", updates.display()))?;
            let opts = mb_surgery::replace::ReplaceOptions {
                config: config.unwrap_or_else(|| updates.join("config.json")),
                overwrite: force,
                max_rel_error,
            };
            let report = mb_surgery::replace::replace_tensors(&t, &t_ir, &u, &out, &opts)?;
            print!("{}", render::surgery(&report));
        }
        Command::Surgery {
            op:
                SurgeryOp::Mtp {
                    target,
                    from,
                    out,
                    force,
                    json,
                },
        } => {
            let open = |p: &std::path::Path| -> Result<(mb_formats::LoadedModel, ModelIr)> {
                let m = mb_formats::open(p).with_context(|| format!("opening {}", p.display()))?;
                let ir = ModelIr::from_raw(m.raw.clone());
                Ok((m, ir))
            };
            let (t, t_ir) = open(&target)?;
            let (r, r_ir) = open(&from)?;
            let opts = mb_surgery::mtp::MtpSidecarOptions {
                overwrite: force,
                reference_label: from.file_name().map(|n| n.to_string_lossy().into_owned()),
                aligned_to_target: false,
            };
            let report = mb_surgery::mtp::port_mtp_sidecar(&t, &t_ir, &r, &r_ir, &out, &opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", render::surgery(&report));
            }
        }
        Command::ExportHf {
            model,
            reference,
            out,
            dtype,
            shard_gib,
            layers,
            no_globals,
            force,
        } => {
            let dtype = match dtype.to_ascii_lowercase().as_str() {
                "bf16" => mb_ir::DType::Bf16,
                "f32" => mb_ir::DType::F32,
                other => anyhow::bail!("unsupported dtype {other}; use bf16 or f32"),
            };
            let layers = layers
                .map(|s| -> Result<(u32, u32)> {
                    let (a, b) = s.split_once("..").context("--layers takes start..end")?;
                    Ok((a.parse()?, b.parse()?))
                })
                .transpose()?;
            let m =
                mb_formats::open(&model).with_context(|| format!("opening {}", model.display()))?;
            let ir = ModelIr::from_raw(m.raw.clone());
            let opts = mb_surgery::hf_export::HfExportOptions {
                reference,
                dtype,
                shard_bytes: (shard_gib * (1u64 << 30) as f64) as u64,
                layers,
                globals: !no_globals,
                overwrite: force,
            };
            let report = mb_surgery::hf_export::export_hf(&m, &ir, &out, &opts)?;
            print!("{}", render::surgery(&report));
        }
        Command::ExportTensors {
            model,
            names,
            dtype,
            out,
            force,
        } => {
            let dtype = match dtype.to_ascii_lowercase().as_str() {
                "bf16" => mb_ir::DType::Bf16,
                "f32" => mb_ir::DType::F32,
                other => anyhow::bail!("unsupported dtype {other}; use bf16 or f32"),
            };
            let m =
                mb_formats::open(&model).with_context(|| format!("opening {}", model.display()))?;
            let ir = ModelIr::from_raw(m.raw.clone());
            let opts = mb_surgery::export::ExportOptions {
                names,
                dtype,
                overwrite: force,
            };
            let report = mb_surgery::export::export_primal(&m, &ir, &out, &opts)?;
            print!("{}", render::surgery(&report));
        }
        Command::Job { op } => job::run(op)?,
        Command::Tui {
            model,
            job,
            events,
            hardware,
            python,
        } => {
            use mb_api::jobs::JobSource;
            let job = match (job, events) {
                (Some(spec), _) => Some(JobSource::Spec {
                    spec_path: spec.display().to_string(),
                    python: None,
                }),
                (None, Some(e)) => Some(JobSource::Events {
                    events_path: e.display().to_string(),
                }),
                (None, None) => None,
            };
            if model.is_none() && job.is_none() {
                anyhow::bail!("give a model, --job <spec.json> or --events <file>");
            }
            mb_tui::run(mb_tui::Options {
                model,
                job,
                hardware,
                python,
            })?;
        }
        Command::Serve {
            port,
            host,
            web_dir,
            python,
            mut allow_hosts,
        } => {
            let web_dir = web_dir.or_else(|| {
                let dev = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/dist");
                dev.join("index.html").is_file().then_some(dev)
            });
            if !host.is_loopback() {
                eprintln!(
                    "warning: binding {host}: anyone who can reach it can read files and start jobs as you"
                );
                if !host.is_unspecified() {
                    allow_hosts.push(host.to_string());
                }
            }
            let cfg = mb_server::ServerConfig {
                web_dir,
                python,
                extra_hosts: allow_hosts,
            };
            tokio::runtime::Runtime::new()?
                .block_on(mb_server::serve(std::net::SocketAddr::new(host, port), cfg))?;
        }
        Command::Fixture { kind, out } => {
            let path = match kind {
                FixtureKind::LlamaGqa => mb_fixtures::llama_gqa(&out),
                FixtureKind::QwenHybrid => mb_fixtures::qwen_hybrid(&out),
                FixtureKind::DeepseekMlaMoe => mb_fixtures::deepseek_mla_moe(&out),
                FixtureKind::GgufMixedQuant => mb_fixtures::gguf_mixed_quant(&out),
                FixtureKind::GgufHybridTernary => mb_fixtures::gguf_hybrid_ternary(&out),
                FixtureKind::GgufLlama => mb_fixtures::gguf_llama(&out),
                FixtureKind::HybridMtpReference => mb_fixtures::hybrid_mtp_reference(&out),
            };
            println!("{}", path.display());
        }
    }
    Ok(())
}

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mb_ir::ModelIr;

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
        /// `kv-share:group=2` or `mtp:from=models/Qwen3.8-27B`.
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
    /// Write a new checkpoint with a feature added. Inputs are never modified.
    Surgery {
        #[command(subcommand)]
        op: SurgeryOp,
    },
    /// Write a tiny synthetic checkpoint, for trying the tool without a real model.
    #[command(hide = true)]
    Fixture { kind: FixtureKind, out: PathBuf },
}

#[derive(Subcommand)]
enum SurgeryOp {
    /// Port an MTP head from a Hugging Face reference model into an MTP-only
    /// GGUF sidecar for a qwen35 GGUF target (run it with `-md <sidecar>
    /// --spec-type draft-mtp` in the PrismML llama.cpp fork).
    Mtp {
        /// Target model (.gguf), e.g. Ternary-Bonsai-2-27B-PQ2_0.gguf.
        target: PathBuf,
        /// Reference HF model directory with `mtp.*` tensors, e.g. Qwen3.8-27B.
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
    GgufBonsaiLike,
    QwenHybridMatchingBonsaiLike,
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
            };
            let report = mb_surgery::mtp::port_mtp_sidecar(&t, &t_ir, &r, &r_ir, &out, &opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", render::surgery(&report));
            }
        }
        Command::Fixture { kind, out } => {
            let path = match kind {
                FixtureKind::LlamaGqa => mb_fixtures::llama_gqa(&out),
                FixtureKind::QwenHybrid => mb_fixtures::qwen_hybrid(&out),
                FixtureKind::DeepseekMlaMoe => mb_fixtures::deepseek_mla_moe(&out),
                FixtureKind::GgufMixedQuant => mb_fixtures::gguf_mixed_quant(&out),
                FixtureKind::GgufBonsaiLike => mb_fixtures::gguf_bonsai_like(&out),
                FixtureKind::QwenHybridMatchingBonsaiLike => {
                    mb_fixtures::qwen_hybrid_matching_bonsai_like(&out)
                }
            };
            println!("{}", path.display());
        }
    }
    Ok(())
}

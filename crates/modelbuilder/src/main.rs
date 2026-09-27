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
    /// Write a tiny synthetic checkpoint, for trying the tool without a real model.
    #[command(hide = true)]
    Fixture { kind: FixtureKind, out: PathBuf },
}

#[derive(Clone, Copy, ValueEnum)]
enum FixtureKind {
    LlamaGqa,
    QwenHybrid,
    DeepseekMlaMoe,
    GgufMixedQuant,
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
        Command::Fixture { kind, out } => {
            let path = match kind {
                FixtureKind::LlamaGqa => mb_fixtures::llama_gqa(&out),
                FixtureKind::QwenHybrid => mb_fixtures::qwen_hybrid(&out),
                FixtureKind::DeepseekMlaMoe => mb_fixtures::deepseek_mla_moe(&out),
                FixtureKind::GgufMixedQuant => mb_fixtures::gguf_mixed_quant(&out),
            };
            println!("{}", path.display());
        }
    }
    Ok(())
}

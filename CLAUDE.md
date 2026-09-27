# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

**modelbuilder** takes an existing LLM checkpoint, breaks it down to show what it
is and what it was trained for, and rebuilds it with new features. Two kinds of
change are in scope:

- **Additive:** bolt something new onto a frozen or mostly frozen trunk, e.g.
  multi-token-prediction (MTP) or speculative-decoding heads.
- **Restructuring:** change the architecture and retrain to recover quality, e.g.
  applying DeepSeek-style KV cache and attention changes (MLA, KV compression,
  sparse or hybrid attention) to a model like Bonsai 2 27B.

For each candidate feature, the tool must answer three questions: *is it
compatible with this model*, *what does it cost to add*, and *what could go
wrong*. It then carries the change through to an exported model.

## Status

The Rust core exists: `mb-ir`, `mb-formats`, `mb-analyze`, `mb-fixtures`, and
the `modelbuilder` binary with `inspect`. Everything else in the layout below is
the target design, not code yet. Update this file when the real layout differs.

Research on the first target (DeepSeek-V4.1-Flash KV techniques on Bonsai 2 27B)
is in `docs/research/targets.md`. Read it before working on attention, KV, or
MTP plugins. It was checked against the paper, model cards, and real file
headers, and it lists what is still unconfirmed.

PrismML's block formats and Hadamard contract are in
`docs/research/prismml-quant-formats.md`, read from the source of the
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork. Clone the
fork outside the repo when you need it, and regenerate the decoder golden
vectors with `scripts/prism-golden.sh <fork checkout>`.

The target checkpoint is `prism-ml/Ternary-Bonsai-2-27B-gguf`, not
`prism-ml/Bonsai-27B-gguf` (that one is the 1-bit model).

## Commands

```sh
cargo build
cargo test --workspace
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
cargo run -- inspect <model.gguf | hf-dir> [--json] [--tensors] [--context N]
cargo run -- fixture qwen-hybrid /tmp/qh   # hidden: writes a tiny test checkpoint
```

To inspect a real multi-GB model without downloading it, fetch only the
headers with HTTP range requests into sparse files of the true size. The
readers only touch header bytes. This is how the numbers in
`docs/research/targets.md` were measured.

## Core pipeline

```
inspect → analyze → plan → surgery → train → evaluate → export
```

1. **Inspect:** read the config, tokenizer, and tensor index into a normalized
   Model IR without loading any weights.
2. **Analyze:** run four layers of analysis, each optional and each more
   expensive than the one before:
   - *Static structure:* attention type (MHA/GQA/MQA/MLA), head dims, MoE
     layout, norm/activation, RoPE config, vocab, tied embeddings, parameter
     counts, and quantization scheme.
   - *Metadata/provenance:* model card, chat template, special tokens, trained
     context length, base vs. instruct lineage, and license.
   - *Weight statistics:* per-tensor norms, outliers, sparsity, and singular
     value spectra, e.g. the effective rank of K/V projections to judge how
     well a model will tolerate MLA or low-rank KV conversion. Computed by
     streaming tensors.
   - *Behavioral probes:* short perplexity and eval runs (long-context
     retrieval, code, tool use, languages) that infer what the model is
     actually good at. These need the Python side and a GPU.
3. **Plan:** the user picks features. The planner checks compatibility,
   orders the stages, and gives estimates for each feature:
   - **Compute:** tokens, GPU-hours, and peak VRAM per hardware profile.
   - **Data:** what data is needed and how much, and whether distillation from
     the original model is enough or fresh data is required.
   - **Quality risk:** expected regressions and the recovery strategy.
4. **Surgery:** Rust rewrites the checkpoint by adding, removing, or reshaping
   tensors, initializing new ones (e.g. SVD init for MLA projections, copies
   for MTP heads), and updating the config.
5. **Train:** Python runs the training stages from a job spec emitted by Rust.
6. **Evaluate:** compare against the original model using the same probes.
7. **Export:** write HF safetensors + config, and/or GGUF, re-quantizing when
   the source was low-bit.

## Architecture

### Rust workspace (inspection, planning, surgery, UI)

Planned crates:

| Crate | Responsibility |
|---|---|
| `mb-ir` ✅ | Normalized Model IR: layers, attention/MLP/MoE blocks, tensors, dtypes and quant formats. All other crates speak this. No IO. |
| `mb-formats` ✅ | Readers and writers for **HF safetensors + config.json** and **GGUF**. Uses mmap and streaming, and never materializes a full model. `dequant` decodes F32/F16/BF16/Q8_0/PQ2_0/PTQ1_0 to f32 one tensor at a time. |
| `mb-analyze` ✅ (static, quant, KV, provenance) | Static, metadata, and weight-statistics analyzers. Weight statistics are not built yet. |
| `mb-fixtures` ✅ | Tiny synthetic checkpoints for tests (Llama GQA, Qwen3.8-like hybrid, DeepSeek MLA+MoE, mixed-quant GGUF). |
| `mb-features` | Feature plugin trait plus the built-in features. |
| `mb-plan` | Recipe parsing, compatibility resolution, stage ordering, and cost models. |
| `mb-surgery` | Applies feature transforms to produce a modified checkpoint. |
| `mb-jobs` | Emits training job specs, launches and monitors the Python side. |
| `mb-server` | Local web API (axum) that serves the React UI. |
| `mb-tui` | Terminal dashboard (ratatui) for SSH and cloud boxes. |
| `modelbuilder` | Binary that launches `serve` (web), `tui`, and headless subcommands for scripting and CI. |

The web UI and TUI are both clients of the same core. Put logic in library
crates, never in the UI layers.

### Web UI

React + TypeScript + Vite, in `web/`. It talks to `mb-server` over a JSON API
whose types are generated from the Rust report types, so they are never
hand-written twice.

### IR conventions

- **Tensor names are the ground truth** for what a layer contains (mixer type,
  MoE, MTP). Config values supply sizes, and mismatches between the two become
  `ModelIr::warnings`, never panics.
- **Shapes are row-major** (`[out, in]` for linear weights) for every format.
  The GGUF reader reverses ggml's `ne` order, and the writer reverses it back.
- **Unknown GGUF tensor types are preserved**, not rejected, and sized from
  the gap to the next tensor offset with `bytes_exact = false`. Vendor types
  with a confirmed block layout (PrismML PQ2_0 = 142, PTQ1_0 = 143) are in the
  `GGML_TYPES` table in `mb-ir/src/dtype.rs`.
- **GGUF metadata keeps exact integer widths** (`MetaValue::U32` vs `U64`),
  because llama.cpp type-checks keys. A GGUF read→write round trip is
  byte-identical (tested).
- New naming conventions go in `mb-ir/src/naming.rs`, with a test case per
  name.

### Python package (training, probes)

`python/modelbuilder_train/` has a **pluggable backend** interface:

- `backends/hf`: PyTorch + transformers + accelerate (FSDP/DeepSpeed) + PEFT.
  This is the first backend.
- `backends/mlx`: Apple Silicon, planned.
- `backends/torchtitan`: multi-node continued pretraining, planned.

The Python side also runs behavioral probes and evals.

### Rust ↔ Python contract

- Rust writes a versioned **job spec** (JSON) with the checkpoint path,
  stages, which params are trainable or frozen, losses (e.g. distillation
  against the original model), data sources, hyperparameters, and the hardware
  profile.
- Python streams **JSONL events** (progress, loss, eval results, errors) on
  stdout or to a file. Rust consumes these to drive the dashboards.
- The schema lives in one place (`schema/`), with Rust types and Python
  (pydantic) types checked against it. Bump the version on any breaking change.

## Feature plugins

Each feature implements the same contract:

- `detect`: does the model already have this, or something equivalent?
- `check_compat`: can it be applied, and what blocks it?
- `estimate`: compute, data, and quality-risk estimates for a hardware profile.
- `surgery`: tensor and config transforms, plus initialization.
- `training_stages`: what to train, what to freeze, the losses, and the data.
- `export_notes`: runtime support implications, e.g. whether llama.cpp or
  vLLM can run the result.

First catalog:

- **MTP / speculative heads:** DeepSeek-V3-style MTP, Medusa, EAGLE. Usually
  trained with the trunk frozen.
- **KV cache / attention rework:** GQA→MLA conversion, KV compression,
  sliding-window or hybrid attention, sparse attention (DeepSeek-style).
- **Context extension:** RoPE scaling (e.g. YaRN) plus a long-context
  finetune.
- **MoE / structural:** dense→MoE upcycling, layer pruning and depth changes,
  vocab and tokenizer changes.

When implementing a technique from a specific model release (e.g. a DeepSeek
version), cite the paper or source code in the plugin, and base the plugin on
what that source actually specifies, not on what the name suggests.

## Recipes

Users compose features in a declarative recipe file (TOML). The UIs read and
write the same recipe format. A sketch:

```toml
[source]
path = "models/bonsai-2-27b"

[hardware]
profile = "1x24GB"   # or "8xH100", "m3-max-128gb", ...

[[feature]]
id = "mtp"
depth = 1

[[feature]]
id = "mla"
kv_lora_rank = 512

[export]
formats = ["safetensors", "gguf"]
```

## Hard constraints

- **Never load a whole model into memory** for inspection, analysis, or
  surgery. Stream tensor by tensor via mmap. Models may be 100B+ parameters.
- **Low-bit and ternary weights are first-class.** Surgery must not silently
  dequantize and re-save at a different precision. Training on low-bit
  sources uses quantization-aware training where needed, and export
  re-quantizes to the source scheme unless the recipe says otherwise.
- **Hardware range:** single consumer GPU, multi-GPU node, cloud cluster, and
  Apple Silicon. Estimates and training plans must be profile-aware; fall back
  to LoRA or partial-layer training when full training doesn't fit.
- **Be honest about estimates:** every estimate carries its assumptions and a
  confidence level. Don't present guesses as measurements.
- **Rotated weight bases are first-class.** Some checkpoints (PrismML Bonsai 2)
  store weights with an orthogonal Hadamard rotation folded in, declared in
  metadata (`ModelIr::weight_rotation`, `prism.hadamard.*`). Surgery must keep
  new or modified tensors in the same basis and keep that metadata in sync
  (contract and loader rules: `docs/research/prismml-quant-formats.md`).
  Per-input-channel statistics must undo the rotation first with
  `WeightRotation::to_primal`; singular-value spectra don't change under it.
- **Leave the source untouched:** surgery writes a new checkpoint and never
  modifies the input model in place.

## Conventions

- Rust: stable toolchain, `cargo fmt`, `cargo clippy -- -D warnings`,
  `cargo test` before committing. Use `thiserror` in libraries and `anyhow`
  only in binaries.
- Python: 3.11+, `ruff` for lint and format, `pytest`, type hints throughout.
  Heavy dependencies (torch, mlx) stay behind optional extras per backend.
- Tests use tiny synthetic checkpoints (generated fixtures, a few MB), never
  real model downloads. Keep golden tests for format round-trips
  (read → write → byte/semantic equality).
- Keep large files out of git: models, datasets, and run outputs go under
  `runs/` or `models/`, both git-ignored.

## Open questions

- See the "Still unconfirmed" list in `docs/research/targets.md`.
- Dataset sourcing and caching strategy for distillation and retraining.

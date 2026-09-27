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

The Rust core exists: `mb-ir`, `mb-formats`, `mb-analyze`, `mb-fixtures`,
`mb-features` (plugin trait, three plugins, cost model), `mb-plan` (recipes and
plans), `mb-surgery` (MTP head port, primal tensor export), `mb-jobs` (job
specs and the Python launcher), `mb-api` (the dashboards' operations and
types), `mb-server` (web API), `mb-tui` (terminal UI), and the `modelbuilder`
binary with `inspect`, `stats`, `plan`, `features`, `surgery mtp`,
`export-tensors`, `job`, `serve` and `tui`. The Python side
(`python/modelbuilder_train`) trains an MTP head against a frozen trunk
(`mtp_align`, see `docs/research/mtp-align.md`). The React UI is in `web/`.
Everything else in the layout below is
the target design, not code yet. Update this file when the real layout differs.

What's done, what's measured, and the plan for the next work are in
`docs/STATUS.md`. Keep it current when a piece of the plan lands.

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
cargo run --release -- stats <model> [--only substr,...] [--kv-spectra] [--top N] [--json]
cargo run -- plan <model> [-f id[:k=v,...]]... [--hardware ids] [--json]   # no -f: whole catalog
cargo run -- plan --recipe examples/recipes/bonsai2-kv-and-mtp.toml
cargo run -- features                        # catalog and hardware profiles
cargo run --release -- surgery mtp <target.gguf> --from <hf-reference> -o <name>-mtp.gguf
cargo run --release -- export-tensors <model> --names a,b [--dtype bf16|f32] -o out.safetensors
cargo run --release -- job mtp-align <target.gguf> --from <hf-reference> --features <dir> -o runs/<name> [--emit-only]
cargo run --release -- job run runs/<name>/job.json
cargo run --release -- serve [--port 7878] [--web-dir web/dist]   # web dashboard + JSON API on localhost
cargo run --release -- tui [<model>] [--job job.json | --events events.jsonl]
cargo run -- fixture qwen-hybrid /tmp/qh   # hidden: writes a tiny test checkpoint

# Web UI (web/): once, then check, test, build (serve picks up web/dist)
cd web && npm ci && npm run format:check && npm run typecheck && npm test && npm run build
npm run dev                                  # Vite on :5173, proxies /api to `serve` on :7878
UPDATE_TYPES=1 cargo test -p mb-server --test types   # regenerate web/src/api/types.ts after changing API types

# Python side (python/): once, then lint and test
python -m venv python/.venv && python/.venv/bin/pip install -e "python[torch,dev]"
cd python && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q
python -m modelbuilder_train extract-features --llama-bin <fork>/build/bin --model <target.gguf> --texts texts.jsonl --out <dir>
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
| `mb-analyze` ✅ | Static, metadata/provenance, KV-cache, and weight-statistics analyzers. `weights` streams tensors a chunk of rows at a time: moments, kurtosis and channel outliers in the primal basis; zeros and ternary structure in the stored basis; K/V singular-value spectra (`nalgebra`). Activation-aware statistics need calibration data and belong to the Python side. |
| `mb-fixtures` ✅ | Tiny synthetic checkpoints for tests (Llama GQA, Qwen3.8-like hybrid, DeepSeek MLA+MoE, mixed-quant GGUF). |
| `mb-features` ✅ | Feature plugin trait, hardware profiles, and the training cost model (`estimate`). Plugins: `fp4-kv`, `kv-share`, `mtp`. `surgery_outline` describes checkpoint changes; executing them belongs to `mb-surgery`. |
| `mb-plan` ✅ | Recipe parsing (TOML) and plan assembly: detection, compatibility, estimates priced per hardware profile. Stage ordering across features is not built yet. |
| `mb-surgery` ✅ (MTP) | Writes new checkpoints with a feature added, streaming from the inputs' mmaps. `mtp::port_mtp_sidecar` ports an HF MTP head into an MTP-only GGUF sidecar for `qwen35` targets, following the PrismML fork's converter; see `docs/research/mtp-port.md` for how it was verified end to end. `export::export_primal` writes selected tensors decoded and un-rotated to safetensors for training. |
| `mb-jobs` ✅ | Job spec and event types (mirroring `schema/`), spec builders (`mtp_align_spec`), and `run`, which launches `python -m modelbuilder_train run` and streams its events. |
| `mb-api` ✅ | What the dashboards can do, as plain functions and serde types: `inspect`, `stats`, `plan`, `catalog`, `fs::list` (model picker), and `jobs::JobManager` (run a spec with the Python side or follow an events file; cursor-based updates, cancel). With the `ts` feature every type derives `ts_rs::TS`. |
| `mb-server` ✅ | Local web API (axum) over `mb-api` plus the static web build. Binds loopback and rejects requests whose `Host`/`Origin` isn't localhost (or `--allow-host`), since it reads files and starts processes. Job updates stream as SSE. `types::typescript()` generates `web/src/api/types.ts`. |
| `mb-tui` ✅ | Terminal dashboard (ratatui 0.29, for the MSRV) over `mb-api`: overview, tensors, plan, live job. `App` is testable headless with `TestBackend`. |
| `modelbuilder` ✅ | Binary: `serve` (web), `tui`, and the headless subcommands for scripting and CI. |

The web UI and TUI are both clients of the same core. Put logic in library
crates, never in the UI layers.

### Web UI

React + TypeScript + Vite, in `web/`. It talks to `mb-server` over a JSON API
whose types are generated from the Rust report types, so they are never
hand-written twice: `web/src/api/types.ts` comes from `mb-server`'s
`types::typescript()`, and a Rust test fails when it is stale. Pages:
Inspect, Weights (stats and K/V spectra), Plan (feature picker or recipe
TOML), Jobs (start or follow a job, live curves over SSE).

- No UI framework or chart library: charts are small SVG components in
  `components/charts.tsx`, following the dataviz method (categorical slots in
  fixed order, one y axis, hover tooltips, legends plus direct labels, light
  and dark tokens in `styles.css`). Status badges always pair color with an
  icon and a label.
- Pure logic (formatting, scales, merging job updates) lives in `src/lib/`
  with Vitest tests. Prettier formats everything except the generated types.

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

`python/modelbuilder_train/` has a **pluggable backend** interface
(`backends.BACKENDS`, selected by the spec's `backend`):

- `torch` ✅: plain PyTorch on CUDA, MPS or CPU. Runs frozen-trunk stages
  (`mtp_align`) from precomputed trunk features, so it never loads the trunk:
  features come from the runtime that serves the model (`features.py`, llama.cpp
  for PrismML GGUFs), frozen tensors from `modelbuilder export-tensors`.
- `hf`: transformers + accelerate (FSDP/DeepSpeed) + PEFT, for stages that train
  the trunk. Planned.
- `mlx`: Apple Silicon, planned.
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
  `schema/examples/` is validated by the Rust tests (`mb-jobs/tests/contract.rs`),
  the Python tests (`test_contract.py`), and the JSON Schema itself. Change all
  three together. `python/tests/test_e2e.py` drives the Rust binary end to end
  (`MODELBUILDER_BIN`, default `target/debug/modelbuilder`) and runs in CI.

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

Users compose features in a declarative recipe file (TOML); the UIs will read
and write the same format. `mb_plan::Recipe` defines it, and
`examples/recipes/bonsai2-kv-and-mtp.toml` is a working example (a test keeps it
valid). Every key in a `[[feature]]` table other than `id` is passed to the
plugin as a parameter, and each plugin rejects unknown keys.

```toml
[source]
path = "models/Ternary-Bonsai-2-27B-PQ2_0.gguf"

[hardware]
profiles = ["1x24GB", "8xH100"]   # `modelbuilder features` lists them

[[feature]]
id = "mtp"
from = "models/Qwen3.8-27B"       # reference model to port the head from

[[feature]]
id = "kv-share"
group = 2
```

### Writing a feature plugin

- Implement `mb_features::Feature` in `crates/mb-features/src/features/` and
  add it to `CATALOG`.
- Parameters are a typed struct with `#[serde(deny_unknown_fields, default)]`,
  parsed with `parse_params`.
- `estimate` returns `Stage`s (what trains, how many tokens, the loss, the
  data), not hours: `estimate::cost` prices them per hardware profile.
- Every estimate lists its assumptions, a `Confidence`, and `references` to
  the paper sections or source files it follows. Token budgets that aren't
  from a source are called heuristics in the assumptions.
- Test it in `crates/mb-plan/tests/plan.rs` against the fixtures.

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

- Rust: stable toolchain (MSRV 1.85, checked in CI; check a new dependency's
  `rust-version` before adding it), `cargo fmt`, `cargo clippy -- -D warnings`,
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

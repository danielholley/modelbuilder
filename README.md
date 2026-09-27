# modelbuilder

A tool for taking LLM checkpoints apart and rebuilding them with new features
(MTP heads, KV-cache and attention changes, context extension, and so on).
Given a model, it shows what the model contains and what it was trained for,
and what adding a feature would cost.

Status: early. `inspect`, `stats` and `plan` work. Surgery, training and
the dashboards are still to come. See `CLAUDE.md` for the design and
`docs/research/targets.md` for the first target (Bonsai 2 27B).

## Build and test

You need a Rust toolchain, **1.85 or newer**. Nothing else is required: there
are no C dependencies and no Python yet.

```sh
# 1. Install Rust, if you don't have it (https://rustup.rs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # macOS / Linux
# Windows: download and run rustup-init.exe from https://rustup.rs

rustup update stable        # if Rust is already installed but older than 1.85
rustup component add clippy rustfmt

# 2. Build and run the tests (they generate tiny synthetic models; no downloads)
cargo build
cargo test --workspace

# 3. Lint, as CI would
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Try it

```sh
# On a tiny synthetic model
cargo run -- fixture qwen-hybrid /tmp/qh
cargo run -- inspect /tmp/qh

# On a real model: a .gguf file, or a Hugging Face directory with
# config.json + *.safetensors. Only headers are read, so this is fast even
# for very large models.
cargo run --release -- inspect path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
cargo run --release -- inspect path/to/Qwen3.8-27B/ --json > report.json
```

Weight statistics stream every tensor (decoding PQ2_0/PTQ1_0 and undoing
Bonsai's Hadamard rotation where needed). `--kv-spectra` adds per-layer K/V
singular-value spectra:

```sh
cargo run --release -- stats path/to/model.gguf --kv-spectra
cargo run --release -- stats path/to/model.gguf --only attn_k.,attn_v. --json
```

## What could I add, and what would it cost?

`plan` evaluates features against a model. For each one it reports:
- whether the model already has it, and whether it's compatible;
- what changes (e.g. KV bytes per token);
- the training stages needed;
- GPU-hours and memory per hardware profile;
- the quality risk, the assumptions, and a confidence level.

With no `-f`, it evaluates the whole catalog.

```sh
cargo run -- features                                   # catalog + hardware profiles
cargo run --release -- plan path/to/model.gguf          # everything, all profiles
cargo run --release -- plan path/to/model.gguf -f kv-share:group=2 -f mtp:from=path/to/base --hardware 1x24GB,8xH100
cargo run --release -- plan --recipe examples/recipes/bonsai2-kv-and-mtp.toml --json
```

Options for `inspect`: `--json` prints the full report as JSON, `--tensors` lists every
tensor with its role, and `--context N` sets the context length used for the
KV-cache totals.

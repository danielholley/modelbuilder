# modelbuilder

A tool for taking LLM checkpoints apart and rebuilding them with new features
(MTP heads, KV-cache and attention changes, context extension, and so on).
Given a model, it shows what the model contains and what it was trained for,
and what adding a feature would cost.

Status: early. `inspect`, `stats`, `plan` and the first surgery (`surgery mtp`)
work. Training and the dashboards are still to come. See `CLAUDE.md` for the design and
`docs/research/targets.md` for the first target (Bonsai 2 27B).

## Build and test

You need a Rust toolchain, **1.85 or newer**, with no C dependencies. The
training side is optional and needs Python 3.11+ (see *Training* below).

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

# 4. Optional: the Python training side and its tests (includes an end-to-end
#    run through the Rust binary built in step 2)
python3 -m venv python/.venv && python/.venv/bin/pip install -e "python[torch,dev]"
cd python && .venv/bin/ruff check . && .venv/bin/pytest -q && cd ..
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

## Surgery: add an MTP head to Bonsai 2

`surgery mtp` ports the multi-token-prediction head from Qwen3.8-27B into an
MTP-only GGUF sidecar for Ternary-Bonsai-2-27B. The sidecar holds about
1.4 GB of weights and is written in seconds; the 7.2 GB model file isn't
touched. PrismML's llama.cpp fork uses the sidecar as a speculative-decoding
draft (results in `docs/research/mtp-port.md`):

```sh
cargo run --release -- surgery mtp models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --from models/Qwen3.8-27B -o models/Ternary-Bonsai-2-27B-mtp.gguf

# with the PrismML fork (https://github.com/PrismML-Eng/llama.cpp):
llama-cli -m models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -md models/Ternary-Bonsai-2-27B-mtp.gguf --spec-type draft-mtp
```

The reference only needs its `mtp.*` tensors and `config.json`, not the whole
checkpoint. Header-only or sparse copies work.

## Training: align the MTP head to Bonsai 2

The ported head was trained against Qwen3.8's trunk. `job mtp-align` fine-tunes
it on Bonsai 2's own hidden states with the trunk frozen: only the head's 425M
parameters train. The trunk runs once, in llama.cpp, to produce features, and
PyTorch never loads the ternary model. See `docs/research/mtp-align.md` for the
design and how it was checked against the fork.

```sh
# Python side (Python 3.11+; install PyTorch for your platform first if you need CUDA/MPS)
python -m venv python/.venv && python/.venv/bin/pip install -e "python[torch]"

# 1. Trunk features: texts.jsonl has one {"text": ...} per line
python/.venv/bin/python -m modelbuilder_train extract-features \
    --llama-bin path/to/prism-llama.cpp/build/bin \
    --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf --texts texts.jsonl --out runs/bonsai2-feat

# 2. Export the frozen tensors, write job.json, train (live progress), write the sidecar
cargo run --release -- job mtp-align models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --from models/Qwen3.8-27B --features runs/bonsai2-feat -o runs/bonsai2-mtp \
    --python python/.venv/bin/python --steps 2000
```

`--emit-only` stops after writing `runs/<name>/job.json`. Copy the run
directory to a GPU machine and run `modelbuilder job run job.json` (or
`python -m modelbuilder_train run job.json`) there.

Options for `inspect`: `--json` prints the full report as JSON, `--tensors` lists every
tensor with its role, and `--context N` sets the context length used for the
KV-cache totals.

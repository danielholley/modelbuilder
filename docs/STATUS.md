# Status and plan

What has been built so far, what it was checked against, what is missing, and
the plan for the next three pieces of work. As of 2026-09-27, `main` at
`00740e5`. The design is in [`CLAUDE.md`](../CLAUDE.md), and the research
behind each technique is in [`docs/research/`](research/).

## Goal

Take an existing LLM checkpoint, show what it is and what it was trained for,
estimate what adding a feature would cost, and rebuild it with that feature.
The first target is PrismML's **Ternary-Bonsai-2-27B** (Qwen3.8-27B after
ternary QAT: PQ2_0 weights, a folded-in Hadamard rotation, 48 gated DeltaNet
layers plus 16 gated GQA layers). The first features are an **MTP head** for
speculative decoding and **DeepSeek-V4.1-Flash-style KV cache changes**.

## Done

| PR | What it added |
|---|---|
| #1 | `CLAUDE.md`; the Rust core: model IR, safetensors and GGUF readers and writers (streaming, mmap), static analysis, `modelbuilder inspect` |
| #2 | Research checked against primary sources (the DeepSeek-V4.1-Flash paper, model cards, real file headers), with fixes for issues found on real models |
| #3 | PrismML PQ2_0/PTQ1_0 decoding with golden vectors from the fork, undoing the Hadamard rotation, CI (fmt, clippy, tests on Linux and macOS, MSRV 1.85, weekly format-drift check), weight statistics and K/V singular-value spectra (`stats`) |
| #4 | Feature plugins (`fp4-kv`, `kv-share`, `mtp`), hardware profiles, the training cost model, recipes, `modelbuilder plan` |
| #5 | MTP surgery: port the Qwen3.8 MTP head into a GGUF sidecar for Bonsai 2 (`surgery mtp`) |
| #6 | The Python training side (`python/modelbuilder_train`, torch backend): the `mtp_align` stage, the job-spec and event schemas (`schema/`), `mb-jobs`, `export-tensors`, `modelbuilder job mtp-align` / `job run` |
| #7 | Dashboards: `mb-api` (operations and types), `mb-server` (web API, localhost-only, SSE), `web/` (React: Inspect, Weights, Plan, Jobs), `mb-tui` (terminal), `modelbuilder serve` / `tui`, TypeScript types generated from Rust |

### Commands that work today

```sh
modelbuilder inspect <model>                       # architecture, params, quantization, KV cache, provenance
modelbuilder stats <model> [--kv-spectra]          # streamed weight statistics and K/V spectra
modelbuilder plan <model> [-f feature[:k=v]]...    # compatibility, stages, cost per hardware profile
modelbuilder surgery mtp <target.gguf> --from <hf-ref> -o <sidecar.gguf>
modelbuilder export-tensors <model> --names a,b -o out.safetensors
modelbuilder job mtp-align <target.gguf> --from <hf-ref> --features <dir> -o runs/<name>
modelbuilder job run runs/<name>/job.json
modelbuilder serve                                 # web dashboard on http://localhost:7878
modelbuilder tui [<model>] [--job job.json | --events events.jsonl]
python -m modelbuilder_train extract-features | run | validate | evaluate-mtp
```

### Measured on the real model

- **PQ2_0 decoding:** decoded and un-rotated Bonsai 2 weights reach cosine
  0.88 against Qwen3.8-27B ([`prismml-quant-formats.md`](research/prismml-quant-formats.md)).
- **K/V spectra:** `[K;V]` needs about 1100–1300 of 2048 dimensions for 90% of
  the energy. Bonsai's spectrum is flatter than Qwen3.8's, so low-rank KV
  should start from the original weights ([`targets.md`](research/targets.md)).
- **Ported MTP head (no training):** 64% draft acceptance (174 of 272 tokens)
  and a 1.37–2.07× decode speedup on CPU in the PrismML fork. The control
  without the zero-centered-norm +1 accepted 0 of 192 ([`mtp-port.md`](research/mtp-port.md)).
- **Training pipeline:** the PyTorch head on llama.cpp features scores 76.0%
  t+2 top-1 over 337 targets (84.4% on generated tokens only). Again, the no-+1
  control scores 0% ([`mtp-align.md`](research/mtp-align.md)).
- **Dashboards:** the web UI was driven in a browser on the real GGUF. A
  training run was started from the Jobs page and followed to completion. The
  TUI was run in a real terminal on the model and on a training job.

### Test coverage

75 Rust tests, 25 Python tests (including an end-to-end run through the Rust
binary on fixtures) and 7 web tests, all in CI. Every test uses tiny synthetic
checkpoints; nothing downloads a model.

## Not done yet

- **A real alignment run on Bonsai 2.** The code works, but a useful run is
  impractical today (next section).
- **Draft acceptance of an aligned sidecar.** Only the unaligned port has been
  measured.
- **Any restructuring surgery.** `fp4-kv` and `kv-share` exist only as plans
  (detection, compatibility, cost). No checkpoint rewrite, training stage or
  export exists for them.
- **Stage ordering across features.** `plan` prices each feature
  independently and doesn't say what order to train them in.
- **Behavioral probes and evals** (perplexity, long-context retrieval, code,
  tool use). They are designed but not built.
- **Other catalog items:** Medusa/EAGLE/DSpark drafters, GQA→MLA, sparse
  attention, context extension (YaRN), MoE upcycling, pruning, vocab changes.
- **Other backends:** `hf` (transformers + FSDP/DeepSpeed + PEFT), `mlx`, `torchtitan`.

## Why a real training run isn't practical yet

A user asked whether they can pull `main` and train on their own servers.
Partly:

1. **No corpus.** `extract-features` needs a `texts.jsonl`, and nothing
   generates one. For alignment the right data is the model's *own*
   generations (self-distillation).
2. **Feature extraction doesn't scale.** It starts `llama-tokenize` and
   `llama-embedding` once per text, so the 7 GB model is reloaded every time.
   It also stores 10 KB per token (5120 bf16 values): 50 M tokens is 500 GB,
   and the plan's 500 M–2 B tokens would be 5–20 TB.
3. **No acceptance benchmark.** Nothing measures the aligned sidecar in
   llama.cpp against the unaligned port.
4. **No runbook** for a fresh server: building the fork with CUDA, installing
   CUDA PyTorch, fetching only the reference model's `mtp.*` tensors instead
   of 54 GB.

## Plan

The order puts the user's servers to work first.

### 1. Make a real MTP alignment run work on a GPU server

- **Corpus generation:** a new `modelbuilder_train generate-corpus` command.
  It drives the fork's `llama-server` (chat template, parallel slots) over a
  prompt set and writes the model's generations as JSONL, the
  self-distillation data the `mtp` plugin's estimate calls for.
- **Online trunk features:** stop storing hidden states. Run the frozen trunk
  in PyTorch during training and feed its post-norm hidden states straight to
  the head:
  - `modelbuilder export-hf`: stream a `qwen35` GGUF into an HF Qwen3.5
    checkpoint (config.json plus sharded bf16 safetensors, primal basis,
    norms with the +1 removed). It is the exact inverse of the fork's
    converter (`conversion/qwen.py`), so transformers' `Qwen3_5` model loads
    it.
  - `mtp_align` gains a `trunk` source (an HF directory) next to `features`.
    The trunk runs in bf16 with no gradients, split across GPUs with a device
    map when it doesn't fit on one. It needs about 54 GB for the trunk: one
    80 GB GPU or two 48 GB GPUs. Precomputed features stay for small cards
    and for validation.
  - Verify the export against llama.cpp. On the real model, compare the
    PyTorch trunk's hidden states with llama.cpp's on the texts already
    extracted. On a machine without the RAM, run it layer by layer. Pass
    means cosine ≈ 1 up to bf16 noise.
- **Faster extraction for the features path:** one long-running
  `llama-server --embeddings --pooling none`, fed token ids so the positions
  line up, instead of a process per text.
- **Acceptance benchmark:** a new `modelbuilder_train bench-draft` command
  that runs the fork's speculative decoding with each sidecar on the same
  prompts and reports acceptance and tokens/s. It compares the ported and
  aligned heads.
- **Runbook and script:** `docs/runbooks/bonsai2-mtp.md` and
  `scripts/bonsai2-mtp.sh`, covering the whole run: build, fetch, corpus,
  export, train, sidecar and benchmark, with the GPU memory each step needs.
- **Checked here where possible:** a short real run on this 16 GB CPU box to
  prove every step on the real model, then the full run on the servers.

### 2. Restructuring: FP4 KV first

The research rates FP4 KV as low risk and a direct fit for the 16 GQA layers
(64 → 18 KiB/token, 16 → 4.5 GiB at 262K). The fork already has quantized KV
cache types. Cross-layer KV sharing is medium–high risk, is untested as a
retrofit, and needs new runtime kernels, so it comes after.

- **The trunk in PyTorch** comes from the `export-hf` work in item 1.
- **A `kv_quant_qat` stage** on a new `hf` backend:
  - Fake-quantize K after RoPE and V, as the paper does, with a
    straight-through estimator.
  - Two formats: `q4_0`, what the fork's KV cache can run today, and
    `nvfp4` (E2M1 plus an E4M3 scale per 16 channels), the paper's format
    with no runtime yet.
  - Train only the full-attention layers' `k_proj`, `v_proj` and `k_norm`
    (168 M parameters). The loss is KL against the same model with an
    unquantized cache: the teacher is the same weights with fake-quant off,
    so there is no second copy.
  - Keep trained weights ternary: fake-quantize them in the rotated basis
    (rotate, ternarize per 128-group, rotate back), so export can re-encode
    them as PQ2_0 with no extra loss.
- **Surgery back to GGUF:** a PQ2_0 encoder in `mb-formats` and a new
  `surgery replace` step. It writes the trained tensors into a copy of the
  GGUF, re-rotated and re-quantized to the source type, and keeps the
  `prism.hadamard.*` metadata in sync.
- **Evaluate:** perplexity and needle retrieval at long context with the
  quantized cache, before and after QAT. This is also the start of the probes
  layer.
- **Then `kv-share`:** grouped K/V reuse across the 16 GQA layers, which needs
  kernels in the fork.

### 3. Stage ordering in the planner

- Each training stage says how it uses the trunk: it *restructures* it (e.g.
  `kv-share` reuse-adapt), *adapts* existing weights (e.g. `fp4-kv` QAT), or
  keeps it *frozen* and reads its outputs (e.g. `mtp-align`).
- The planner orders stages as restructure, then adapt, then frozen. A stage
  trained against the trunk is invalidated by any later change to it. So an
  MTP head is aligned last, and QAT runs after structural KV changes so it
  quantizes the final K/V. Within a feature, the plugin's order is kept.
- `Plan` gains a `schedule` (ordered stages, with the reason for each
  position) and totals per hardware profile (summed GPU-hours and wall-clock
  time, the peak memory, and the worst fit). All three UIs show it.
- Later, `plan` can emit one multi-stage job spec for the whole schedule.

## Target hardware (the user's cluster)

| Machines | Per machine | Good for |
|---|---|---|
| 2 GPU servers | 4 × Tesla P40 (24 GB, Pascal sm_61) | llama.cpp generation and extraction (int8 dp4a kernels); fp32 training of small heads |
| 9 CPU servers | 256 GB RAM, 32 threads (Xeon E5-2600 v2, AVX, no AVX2) | llama.cpp prefill for feature extraction in parallel (the fork has SSE/SSSE3 PQ2_0 kernels) |
| 1 CPU server | 768 GB RAM, 32 threads | large CPU jobs, e.g. streaming weight statistics or exports |
| Network and storage | dual 40 GbE, 50 TB disk array | shared corpus and feature store |

What that changes:

- **No bf16 on P40.** Pascal has no bf16 support, and fp16 runs at 1/64 rate.
  PyTorch training must run in **fp32**. The torch backend has to force
  float32 on GPUs below compute capability 8.0.
- **The trunk doesn't fit in PyTorch.** Bonsai 2 in fp32 is about 108 GB,
  more than one server's 96 GB of GPU memory. Training with the trunk online
  is out on this hardware; that path stays for the restructuring work and
  larger GPUs.
- **Precomputed features fit.** Features take 10 KB per token, so 50 M
  tokens is 0.5 TB and 500 M tokens is 5 TB, which fits on the array.
  Extraction becomes a sharded job:
  - each machine runs its own `llama-server` and writes its own feature
    directory to the array;
  - training reads all of them.
- **The MTP head trains in fp32 on the 8 P40s** with DDP:
  - Memory per GPU: head 1.7 GB, gradients 1.7 GB, AdamW 3.4 GB, fp32 LM head
    5 GB, fp16 embedding 2.5 GB, plus activations. That is about 15–18 GB of
    24 GB.
  - Compute is roughly 10 GFLOP per token, dominated by the 248K-vocab LM
    head. That gives an estimated 500 tokens/s per GPU and about 4,000
    tokens/s across 8, so 50 M tokens takes about 3.5 hours. These are
    estimates, not measurements.
- **Corpus generation is the slow step.** It is decode-bound.
  `llama-server` with parallel slots runs on each P40, and the CPU servers
  can add throughput. Throughput has to be measured on the first run.
- **KV-cache QAT of the 27B trunk isn't practical on P40s** at the planned
  100–500 M tokens (trunk forward in fp32, split across 4 GPUs, at roughly
  100 tokens/s per server). So step 2 starts by *measuring* the quality cost
  of the fork's quantized KV cache with no training. QAT is only worth it if
  that cost is large, and then with a much smaller token budget.

### Revised order

1. Force fp32 on pre-Ampere GPUs, and add a `4xP40` hardware profile to the
   planner.
2. `generate-corpus` over `llama-server`, sharded by machine.
3. `extract-features` over `llama-server`, sharded, with `FeatureSet`
   reading many directories.
4. DDP for `mtp_align` (`torchrun`).
5. A runbook for this cluster, plus `bench-draft` for acceptance.
6. Verify `export-hf` on the real model (per-tensor cosine against
   Qwen3.8-27B; hidden states against llama.cpp).
7. Stage ordering in the planner.
8. Measure quantized-KV quality (perplexity, long-context retrieval), then
   decide on QAT.

Done so far on this list: `export-hf` itself (commit `c4225d0`). It loads in
transformers on fixtures; the real-model check is item 6.

## Open questions

- ~~Which GPUs are on the servers?~~ Answered: see *Target hardware*.
- **What the fork's KV-cache types support** on the target GPUs (q4_0 cache
  with flash attention), to pick the QAT format that can actually run.
- **Where the prompt set comes from** (dataset sourcing and caching is already
  an open question in `CLAUDE.md`).
- The "Still unconfirmed" list in [`targets.md`](research/targets.md).

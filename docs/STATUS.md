# Status and plan

What has been built so far, what it was checked against, what is missing, and
what comes next. As of 2026-11-03 (PRs up to #9 merged; the branch
`claude/upbeat-babbage-9bw8ij` adds the work under *Added after PR #9*). The
design is in [`CLAUDE.md`](../CLAUDE.md), and the research behind each
technique is in [`docs/research/`](research/).

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
| #8 | The cluster path for MTP alignment: fp32 on pre-Ampere GPUs and a `4xP40` profile, `generate-corpus`, sharded `extract-features` over `llama-server`, DDP for `mtp_align`, `bench-draft`, the [runbook](runbooks/bonsai2-mtp.md), `export-hf` |
| #9 | Architecture adapters (`llama`, `qwen2`, `qwen3`, `qwen35`); ggml's reference encoders (Q8_0, PQ2_0, PTQ1_0); `surgery replace`; the `trunk_distill` stage (KV-cache QAT, cross-layer KV sharing, weights fake-quantized through their stored format); stage ordering in the planner; probes (`perplexity`, `needle`, `kv-cache`, `hf-vs-gguf`); the `mla`, `draft-head`, `yarn`, `prune-layers` and `moe-upcycle` plugins; `surgery prune` and `surgery yarn` |

## Added after PR #9 (branch `claude/upbeat-babbage-9bw8ij`)

- **`surgery upcycle`** (dense → MoE, Komatsuzaki et al.): every layer's MLP
  becomes `--experts` copies plus a random router. Expert tensors repeat the
  stored bytes (`gguf::write_parts` streams them from the mmap), so ternary
  and rotated sources stay exact and `prism.hadamard.weight_names` is renamed
  to the expert tensors. Maps `llama` (Mixtral layout) and `qwen3` →
  `qwen3moe`; llama.cpp renormalizes top-k weights for both, so the result
  computes the dense function until trained. `export-hf` writes MoE GGUFs as
  Mixtral or Qwen3-MoE checkpoints (one tensor per expert, routing config
  with renormalized top-k): in transformers the upcycled model matches the
  dense one exactly, and `trunk_distill` trains its experts and routers
  (transformers' fused expert parameters are mapped back to checkpoint
  names and fake-quantized in the source format) and `surgery replace`
  re-encodes every expert into the stacked GGUF tensors. `qwen35` hybrids
  become `qwen35moe` with routed experts plus a sigmoid-gated shared expert
  (the dense gate/up projections, a zero down projection, a zero gate), so
  the shared expert adds nothing until trained but still gets gradients; in
  transformers' Qwen3.5-MoE the upcycled ternary, rotated hybrid fixture
  matches the dense model to 1e-10.
- **`surgery mla`** (MHA2MLA, arXiv 2502.14837): partial RoPE per KV head
  (data-free `high` or `uniform` frequency choice) and a joint SVD of the
  RoPE-free K rows and V into a latent (`kv_a`, `kv_b`, `k_rope`), written to
  `mla.safetensors` next to links to the original HF weights, which are the
  teacher. `trunk_distill` switches the student to the latent through hooks
  on `k_proj`/`v_proj` and the rotary embedding (`hf/mla.py`), so Q/K norms,
  gating and partial-rotary models work unchanged. Exact at full rank on
  llama, qwen2 (biases), qwen3 (Q/K norms) and a Qwen3.5 hybrid (partial
  RoPE); `job trunk-distill --mla-rank` converts and trains in one go.
  `modelbuilder_train mla-rope-scores` ranks RoPE frequencies by query/key
  norms on calibration text (MHA2MLA's S_2-norm, one choice for all heads
  since they share a rotary table) for `surgery mla --freqs`. The
  `mla` plugin now counts RoPE dims per KV head, as MHA2MLA does.
- **The `draft_align` stage**: Medusa heads (zero-initialized residual
  blocks, own or shared LM head, 0.8^k weighting) and an EAGLE drafter (fc
  over [embedding; feature], decoder layers of the target's shape, no input
  norm in the first, SmoothL1 + 0.1·CE), trained from the same precomputed
  features as `mtp_align`. New stage kind in the schema, Rust and pydantic
  types; `modelbuilder job draft-align`.
- **`trunk_distill` options**:
  - `teacher`: a separate checkpoint to distill from (e.g. the unpruned
    original);
  - `loss = "ce"`: next-token training on the texts (long-context finetunes);
  - `lora_rank`: LoRA factors inside the fake-quant parametrization, so the
    merged tensor still writes back exactly;
  - `device_map`: the model and teacher spread layer by layer over one
    machine's GPUs (accelerate), for trunks that don't fit on one device.
    Only the CPU path is tested here;
  - `shard = "fsdp"`: decoder layers sharded over `torchrun` processes
    (FSDP2), on one or more machines; `--nproc-per-node` launches them
    locally. Plain tensors only (no LoRA or weight fake-quant), so it pairs
    with next-token loss or a separate teacher. Tested with two CPU ranks.

  `export-hf` writes a GGUF's YaRN scaling into the HF config, so a finetune
  after `surgery yarn` trains with the scaling transformers applies.
- **Plan pipeline**: each plugin lists the commands that apply it
  (`Feature::pipeline`); the planner orders them like the schedule and
  chains the GGUFs they produce. Shown by `plan` and on the web Plan page.
  Prune healing, KV-share long-context adaptation and the YaRN finetune now
  have runners.
- **LoRA in the cost model**: a `with_lora` fit tier, priced per profile,
  before the packed-trunk tier.
- **`probe skills`**: a short auto-graded suite (code output, reasoning,
  JSON tool calls, translations) through `llama-server`, reported per
  category, the probe layer's "what is it good at" check; `--items` takes a
  custom suite. A smoke test to compare before and after a change, not a
  benchmark.
- **The `mlx` backend** (Apple Silicon): `mtp_align` with the head in MLX,
  matching the PyTorch head to 1e-4 and writing the same `mtp.*` output;
  `job mtp-align --backend mlx`. Tested in CI with `mlx[cpu]` on Linux.

### Commands that work today

```sh
modelbuilder inspect <model>                       # architecture, params, quantization, KV cache, provenance
modelbuilder stats <model> [--kv-spectra]          # streamed weight statistics and K/V spectra
modelbuilder plan <model> [-f feature[:k=v]]...    # compatibility, stages, cost per profile, and the pipeline of commands
modelbuilder surgery mtp|prune|yarn|upcycle|replace ...   # GGUF surgery
modelbuilder surgery mla <hf-dir> --kv-rank R --rope-dim r -o <dir>
modelbuilder export-hf <model.gguf> --reference <hf-dir> -o <hf-out>
modelbuilder export-tensors <model> --names a,b -o out.safetensors
modelbuilder job mtp-align | draft-align | trunk-distill ...   # write a job spec, run it, finish (sidecar / write-back)
modelbuilder job run runs/<name>/job.json
modelbuilder serve                                 # web dashboard on http://localhost:7878
modelbuilder tui [<model>] [--job job.json | --events events.jsonl]
python -m modelbuilder_train generate-corpus | extract-features | run | validate | evaluate-mtp | bench-draft | probe ...
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
- **`export-hf` per tensor** (2026-11-03): layers 0–3 of the PQ2_0 GGUF
  (three DeltaNet layers and one gated-attention layer, 53 tensors) exported
  with `export-hf --layers 0..4 --no-globals --dtype f32` and compared with
  the same tensors of Qwen3.8-27B (fetched by range with
  `modelbuilder_train fetch-tensors`). Every ternary matrix is at cosine
  0.883–0.898, the level of the ternary QAT itself; `A_log` and `dt_bias`
  are at 1.0000 (the `-exp` inverse is right), `conv1d` at 0.995 and
  `in_proj_a/b` at ≥ 0.97 (the V-head reorder is right), and every norm at
  ≥ 0.987 (the +1 offset is removed). A transform or permutation error would
  collapse those numbers. The llama.cpp side (`probe hf-vs-gguf`) is still
  to run.
- **Dashboards:** the web UI was driven in a browser on the real GGUF. A
  training run was started from the Jobs page and followed to completion. The
  TUI was run in a real terminal on the model and on a training job.

### Test coverage

106 Rust tests, 80 Python tests (including end-to-end runs through the Rust
binary on fixtures: MTP alignment, KV QAT, MLA recovery, healing a pruned
model against its original, LoRA write-back, draft heads) and 7 web tests,
all in CI. Every test uses tiny synthetic checkpoints; nothing downloads a
model.

## Not done yet

These need hardware, data or runtime code that isn't in this repository:

- **Real-model checks of the newer pipeline** (steps in
  [`verify-export-and-kv-qat.md`](runbooks/verify-export-and-kv-qat.md)):
  - `probe hf-vs-gguf` on a pruned copy of the target (about 17 GB of disk;
    the per-tensor check against Qwen3.8-27B is done);
  - a `trunk_distill` run on GPUs;
  - the quantized-KV measurement (`probe kv-cache`), which decides whether
    KV QAT is worth its cost on the target.
- **Runtime support** that lives in llama.cpp (or the PrismML fork), not here:
  - KV-sharing reuse layers: training works, but no build runs them, so the
    written-back GGUF still carries every layer's K/V;
  - MLA outside DeepSeek's architecture: a converted model stays in HF form;
  - Medusa/EAGLE-1 heads in GGUF: the drafters are written as safetensors
    for runtimes that load them (e.g. vLLM).
- **A llama.cpp check of `surgery upcycle`**: transformers' MoE classes give
  the dense model's logits exactly, but no llama.cpp run has yet (building
  llama.cpp wasn't possible in the session that wrote it). The `qwen35moe`
  key and tensor names follow llama.cpp's qwen2moe/qwen3next conventions
  and need that run to confirm.
- **DSpark drafters**: the paper describes the drafter but there is no
  public reference code to follow.
- **Backends**: `mlx` runs `mtp_align` only (other stages use torch on MPS).
  Multi-node training is FSDP2 over `torchrun` in the `hf` stage; tensor,
  pipeline or context parallelism (a `torchtitan` backend) for continued
  pretraining at upcycling scale doesn't exist. Neither `device_map` nor
  FSDP has run on real GPUs yet.

## Next

1. Run the real-model checks above on the cluster (item 6 and 8 of the
   revised order below).
2. Run upcycled fixtures in llama.cpp (logits equal to the dense model's),
   including the `qwen35moe` loader.
3. Run `trunk_distill --device-map auto` and `--shard fsdp` on real GPUs.

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

1. ✅ Force fp32 on pre-Ampere GPUs; a `4xP40` hardware profile.
2. ✅ `generate-corpus` over `llama-server`, sharded by machine.
3. ✅ `extract-features` over `llama-server`, sharded, with `FeatureSet`
   reading many directories. Checked on the real model: features match the
   old path at cosine 1.000000 and the tokens match the HF tokenizer (it also
   exposed a bug in the old path: llama.cpp unescapes `\t` in prompt text).
4. ✅ DDP for `mtp_align` (`torchrun`).
5. ✅ The [runbook](runbooks/bonsai2-mtp.md) and `bench-draft`. On the real
   model the ported head accepts 20 of 23 drafts on a code prompt, a 1.93×
   speedup on CPU.
6. Verify `export-hf` on the real model. ✅ Per-tensor cosine against
   Qwen3.8-27B (see *Measured on the real model*); hidden states against
   llama.cpp with `probe hf-vs-gguf` still to run.
7. ✅ Stage ordering in the planner, and now the pipeline of commands.
8. Measure quantized-KV quality (`probe kv-cache`: perplexity, long-context
   retrieval), then decide on QAT.

## Open questions

- ~~Which GPUs are on the servers?~~ Answered: see *Target hardware*.
- **What the fork's KV-cache types support** on the target GPUs (q4_0 cache
  with flash attention), to pick the QAT format that can actually run.
- **Where the prompt set comes from** (dataset sourcing and caching is already
  an open question in `CLAUDE.md`).
- The "Still unconfirmed" list in [`targets.md`](research/targets.md).

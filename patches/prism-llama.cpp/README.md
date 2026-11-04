# Patches for the PrismML llama.cpp fork

Runtime support for checkpoints that modelbuilder writes but no released
llama.cpp build runs yet. Apply on a checkout of
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp):

```sh
git -C <fork> apply /path/to/modelbuilder/patches/prism-llama.cpp/0001-cross-layer-kv-sharing.patch
cmake -B <fork>/build -DCMAKE_BUILD_TYPE=Release && cmake --build <fork>/build -j
```

Made against fork commit `62c418ccd7aefee37997e94c2d9fac1eed7fc89b`.

## 0001: cross-layer KV sharing

Reads `{arch}.attention.kv_reuse_layer` (one `int32` per layer: the earlier
layer whose KV cache this layer reads, or -1), written by
`modelbuilder surgery kv-share` and by `modelbuilder job trunk-distill
--kv-share-group`:

- `llama_hparams::kv_reuse_layer`, validated at load (a reuse layer must name an
  earlier layer that keeps its own cache); `has_kv()` is false for reuse layers.
- The KV cache maps each reuse layer to its producer's cache through the
  existing `layer_reuse_cb` (as Gemma 3n does), for both the plain and the
  hybrid (attention + recurrent) memory, so reuse layers allocate nothing.
- `qwen35` graphs: a reuse layer still computes its own queries, Q norm, RoPE
  and output gate, but attends over the producer's cached K/V and writes none.
- `build_attn` (non-SWA) accepts null K/V, as the SWA overload already did.

Other architectures read the key (and share the cache) but their graphs still
pass their own K/V to `build_attn`, which would write into the producer's
cache; extend their graphs the same way before using the key with them.

Checked on the real model (2026-11-04): Ternary-Bonsai-2-27B pruned to 8
layers (attention layers 3 and 7) with `surgery kv-share --group 2`. The KV
cache halves (2 → 1 cache layers, 32 → 16 MiB at 4096 tokens), and
`probe hf-vs-gguf --kv-share-group 2` against the PyTorch model with the same
sharing gives cosine 0.99996 (min 0.99972) and 99.05% top-1 agreement over
1,266 tokens; without sharing on the PyTorch side, 0.972 and 79.8%.

## 0002: DSpark drafters capture their target layers

At the fork commit above, nothing calls `llama_set_capture_layers`, so
`--spec-type draft-dspark` fails on the first speculative batch for every
checkpoint ("llama_get_embeddings_capture_ith(0) returned null"), including
DeepSeek's published `deepseek-ai/dspark_qwen3_4b_block7`. The patch:

- adds `llama_model_dspark_get_target_layers`, and the DSpark host loop enables
  capture of the drafter's tap layers on the target context;
- DSpark's `begin()` keeps context rows that `process()` staged for exactly the
  prompt it is given, instead of clearing them (callers evaluate the prompt
  before `begin()`);
- `llama-server` and `llama-speculative-simple`: for DSpark, every prompt row
  is an output (the drafter reads the taps of all of them) and the output
  limit is raised to the batch.

With it, a DSpark drafter trained by `modelbuilder job draft-align --kind
dspark` against Qwen3.5-0.8B (200 CPU steps), converted with the fork's
`convert_hf_to_gguf.py --target-model-dir`, drafts in `llama-speculative-simple
--spec-type draft-dspark --no-repack`: 485 tokens drafted, 27 accepted (5.6%),
matching the drafter's 5–7% top-1 in training; `llama-server --spec-type
draft-dspark --no-repack`: 464 drafted, 26 accepted (5.6%).
Two more limits of the fork, found the same way and not patched:

- Only `qwen35` target graphs produce capture tensors, so DSpark targets must
  be Qwen3.5-family models (as PrismML's shipped drafters are).
- With weight repacking on (the CPU default), reading the drafter's Markov head
  calls a null function and crashes (`llama_model_dspark_get_markov`); run with
  `--no-repack`.

## 0003: multi-head latent attention (MLA) converted from GQA

Runs checkpoints from `modelbuilder surgery mla` (optionally trained with
`job trunk-distill --mla-rank`) after `modelbuilder surgery mla-gguf` writes
them into the GGUF. Design and cache layout: `0003-mla-design.md`. Per converted
layer the cache holds `[latent | roped, weighted RoPE key | ||RoPE key||² per
head]`; K and V are decompressed from it each step, K RMSNorm is finished from
the stored norm, and Q is permuted into the cached head order. `qwen35` graphs
only; F32 latent tensors.

Checked on Qwen3.5-0.8B (6 attention layers; 2 KV heads × 256, 64 rotary
dims), 2026-11-04:

- full rank (896, all 64 RoPE dims; exact by construction): perplexity
  47.4867 vs 47.4850 for the original GGUF;
- rank 256 with 32 RoPE dims per head: KV cache 24.00 → 7.59 MiB at 1,024
  tokens (1,024 → 322 values per token and layer), and the runtime matches
  the PyTorch MLA model (`hf/mla.py`) token by token: cosine 1.000000 (min
  0.999996), 100% top-1 over 1,266 tokens. Against the unconverted model:
  0.969 and 80.1%.

Apply 0001, 0002, then 0003 (each a diff on top of the previous one).

# Target models and techniques: research notes

Status: **secondary sources only**. arxiv.org and huggingface.co were blocked
from the dev environment when this was written, so every figure below comes
from search-result summaries of the paper, model cards, and press coverage.
Confirm against the primary sources (listed at the end) before implementing a
plugin, and replace "reported" with a citation once confirmed.

## Bonsai 2 27B (PrismML)

What it is: a **ternary quantization-aware-trained (QAT) version of
Qwen3.8 27B**, released 2026-09-18, Apache 2.0.

### Base architecture (Qwen3.8 27B, reported)

| Property | Value |
|---|---|
| Layers | 64 |
| Hidden size | 5120 |
| FFN intermediate | 17,408 (dense FFN) |
| Layer pattern | 16 × [3 × (Gated DeltaNet → FFN), 1 × (Gated Attention → FFN)], so **48 linear-attention layers and 16 full-attention layers** |
| Gated DeltaNet | 16 QK heads, 48 V heads, head dim 128. Fixed-size recurrent state, **no per-token KV cache** |
| Gated Attention | 24 Q heads, **4 KV heads** (GQA), head dim **256** |
| Context | 262,144 native, extendable to ~1M |
| MTP | **The base checkpoint ships an MTP head** trained with multiple steps |
| Modality | Vision-language (text + image input) |

### Quantization (reported, partly conflicting)

- Weights in {−1, 0, +1} with FP16 group-wise scales, group size 128.
- GGUF type **PQ2_0**: a PrismML-specific type that stores each trit in a
  2-bit slot, reported at 2.13 bits/weight. It is not an upstream ggml type,
  so readers must handle an unknown type id.
- Other figures reported: "1.72 true bits per weight", 5.9 GB (text only)
  vs. 7–8.6 GB. These may describe different packings, or text-only vs.
  with vision. **The inspector should measure bits/weight from the file
  rather than trust any of these numbers.**
- Runs on PrismML's llama.cpp fork, with custom low-bit kernels for the
  hybrid-attention stack on CUDA and Apple Silicon.
- Reported quality: 98.2% of the FP16 baseline averaged over 14 benchmarks.

## DeepSeek-V4.1-Flash (DeepSeek-AI, arXiv 2609.19969)

Title: *Pushing the Limits of KV Cache Compression*.

| Property | Value (reported) |
|---|---|
| Size | 552B backbone. 16B active per token at decode, 8B at prefill |
| MoE | 384 routed experts + 1 shared per layer, 6 routed active per token |
| Context | Up to 1M tokens |
| Layout | **Causal Encoder-Decoder (CED)**: 40 layers = 20-layer causal encoder + 20-layer decoder. The decoder's global KV is projected from the **final encoder hidden states**, not from each decoder layer |
| KV footprint | **890 bytes/token**, ~4× smaller than V4-Flash |

### KV techniques

1. **CSA2 (Compressed Sparse Attention 2).** Each layer is statically assigned
   one of three modes:
   - **Full:** computes global KV (main KV + indexer K) and runs the indexer
     to pick Top-K positions.
   - **Reindex:** reuses the global KV from a preceding layer, but uses its
     own indexer Q to rescore the shared indexer K and pick fresh Top-K
     positions.
   - **Reuse:** reuses both the most recent main KV and the latest Top-K
     indices, with no indexer compute.

   Still unconfirmed: the mode assignment pattern, the Top-K value, indexer
   dims, and the token compression ratio inherited from V4's CSA.
2. **FP4 KV cache:** E2M1 with one scale per 16 channels, applied on top of
   CSA2.
3. **SWA Bounded Replay:** rebuilds sliding-window KV by replaying the last
   `n_win` tokens instead of persisting it. Persistent KV is ~1/8 of V4-Flash.
   V4 is framed as "an SWA local backbone + compressed global context".

## What this means for "V4.1-Flash-style KV on Bonsai 2 27B"

**Baseline KV cost.** Only the 16 full-attention layers keep a per-token KV
cache:

```
16 layers × 2 (K,V) × 4 kv_heads × 256 head_dim × 2 bytes (BF16) = 65,536 B/token
```

At the full 262K context that is **16 GiB of KV**, about 3× the size of the
quantized weights. The GDN layers add a constant-size state per sequence.
Compared with 890 B/token, there is roughly 74× of headroom, so the payoff is
large. The inspector computes this number automatically (`mb-analyze`).

**How each technique transfers** (first-pass effort estimates, all to be
refined by the cost model):

| Technique | Fit for Bonsai 2 27B | Likely effort |
|---|---|---|
| FP4 KV cache (E2M1, per-16 scale) | Direct: applies to the 16 GQA layers | Low. Mostly inference-side, plus calibration and a short finetune if quality drops. Needs kernel support in the runtime |
| Cross-layer KV reuse (CSA2 Reuse-style) | Good: 16 full-attn layers could share KV across groups, keeping 4 KV sets instead of 16 | Medium. Remove K/V projections on Reuse layers, then a partial retrain with distillation from the original |
| Compressed sparse attention + indexer (CSA/CSA2 Full/Reindex) | Possible: needs new indexer modules and token compression on the full-attn layers | High. New params (trained ternary or kept higher precision), long-context retraining |
| SWA Bounded Replay | **Not directly applicable**: in Qwen3.8 the GDN layers do the local work, not SWA | n/a |
| CED encoder-decoder restructure | Fundamental change to the layer graph | Very high. Effectively continued pretraining |

**Cross-cutting constraints:**

- **Ternary.** Any new or reshaped projection must be QAT-trained to ternary,
  or be explicitly kept at higher precision and reported in the bits/weight
  budget.
- **Runtime.** Architectural changes need matching kernels in PrismML's
  llama.cpp fork (PQ2_0 + hybrid attention). Export must flag this.
- **MTP.** "Adding MTP" to Bonsai may really mean *recovering* the base
  model's MTP head: check whether the Bonsai checkpoint kept it, then
  re-quantize or re-train it against the ternary trunk. The inspector must
  detect MTP tensors (`mtp.*`, `nextn`).

## Sources

- [DeepSeek-V4.1-Flash paper (arXiv 2609.19969)](https://arxiv.org/abs/2609.19969)
- [deepseek-ai/DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [Paper review, Andrey Lukyanenko](https://andlukyane.com/blog/paper-review-deepseek-v41-flash)
- [MindStudio: V4.1 Flash KV compression](https://www.mindstudio.ai/blog/deepseek-v4-1-flash-kv-cache-compression)
- [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- [PrismML docs: Bonsai 27B](https://docs.prismml.com/models/bonsai-27b)
- [MarkTechPost: Bonsai 2 27B release](https://www.marktechpost.com/2026/09/18/prismml-releases-ternary-bonsai-2-27b-a-5-9-gb-apache-2-0-model-retaining-98-2-of-qwen3-8-27b-performance/)
- [MindStudio: Ternary Bonsai 2 27B](https://www.mindstudio.ai/blog/ternary-bonsai-2-27b-ternary-quantization)
- [MindStudio: Qwen3.8-27B architecture](https://www.mindstudio.ai/blog/qwen3-8-27b-architecture-benchmarks)
- [SGLang cookbook: Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)

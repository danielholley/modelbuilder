# Target models and techniques: research notes

Status: **checked against primary sources** on 2026-09-27: the DeepSeek-V4.1-Flash
paper (arXiv HTML), the Hugging Face model cards, `KNOWN_ISSUES.md`, the
Qwen3.8-27B `config.json` and safetensors index, and the GGUF headers of both
released Bonsai 2 packings, parsed by `modelbuilder inspect`. Items still
unconfirmed are listed at the end.

Measured means the figure came from running `modelbuilder inspect` on the
real file headers (HTTP range requests, no weights downloaded). The figures
below are reproducible that way.

## Bonsai 2 27B (PrismML)

Repo: [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(released 2026-09-16, Apache 2.0). Don't confuse it with `prism-ml/Bonsai-27B-gguf`,
which is **1-bit Bonsai 27B**, a different model: binary weights on a Qwen3.6-27B
base, Q1_0_g128.

### Architecture: unchanged Qwen3.8-27B

Measured from both the Bonsai GGUF and the Qwen3.8-27B safetensors headers:

| Property | Value |
|---|---|
| Layers | 64: `16 × [3 × gated DeltaNet + dense FFN, 1 × gated GQA + dense FFN]` |
| Hidden / FFN | 5120 / 17,408 (SwiGLU) |
| Full attention (16 layers) | 24 Q heads, 4 KV heads, head dim 256, sigmoid output gate (`attn_output_gate`), QK-norm, partial RoPE (`partial_rotary_factor` 0.25, i.e. 64 of 256 dims, mRoPE sections [11, 11, 10]) |
| Gated DeltaNet (48 layers) | 16 QK heads, 48 V heads, head dims 128/128, conv kernel 4 |
| Vocab / context | 248,320 / 262,144 |
| Params (measured) | 26.90B in the Bonsai language GGUF. Qwen3.8 base is 27.78B, including a 424.7M MTP module and a 460.7M vision tower |
| RoPE theta | 1e7 |

HF tensor names (Qwen3.8, `model_type: qwen3_5_text`): DeltaNet projections are
split into `linear_attn.in_proj_{qkv,z,a,b}`. llama.cpp (`general.architecture:
qwen35`) names the same weights `attn_qkv`, `attn_gate`, `ssm_alpha`,
`ssm_beta`, `ssm_out`, and so on.

### Quantization (confirmed, measured)

| Packing | GGUF type id | Layout | bits/weight | File size |
|---|---|---|---|---|
| **PQ2_0** | 142 | trit in a 2-bit slot + FP16 scale per 128 weights (34 B/block) | 2.125 (measured) | 7.21 GB |
| **PTQ1_0** | 143 | dense trits + FP16 scale per 128 weights (28 B/block) | 1.750 (measured) | 5.95 GB |

- The weights are ternary {−1, 0, +1}, trained with QAT. They cover embeddings,
  attention, MLP, and the LM head. In the files, 402 of 851 tensors are
  ternary, 26.87B params.
- **Higher precision:** 26.2M params (0.0976%) stay above ternary: the
  DeltaNet recurrent-state path (`ssm_alpha`, `ssm_beta` in bf16; `ssm_a`,
  `ssm_conv1d`, `ssm_dt` in f32) and all norms.
- Both types are **PrismML vendor types**. Stock llama.cpp rejects them;
  running them needs the [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork.
- **Hadamard-rotated weight basis.** Each matrix is rotated blockwise along
  its input dimension (normalized Sylvester-Walsh-Hadamard, block 1024,
  explicit ±1 signs for input widths 5120, 6144 and 17408). The rotation is
  folded into the stored weights, and the runtime rotates activations to match.
  It is declared in GGUF metadata under `prism.hadamard.*`: 401 rotated
  weights, plus `token_embd.weight` stored with the inverse rotation. Without
  that metadata, a PQ2_0/PTQ1_0 file silently produces garbage.

### MTP: not present in Bonsai 2

The measured Bonsai GGUF contains **no MTP tensors**, even though the Qwen3.8
base ships one MTP layer (`mtp_num_hidden_layers: 1`, 424.7M params, bf16).
`KNOWN_ISSUES.md` says "MTP is refused for these files — fixed in source (#205)",
so the PrismML runtime is adding MTP support, but the released weights don't
include a head. The sibling 1-bit Bonsai 27B instead ships a **DSpark
drafter**: 6 layers conditioned on hidden states tapped from 5 target layers,
reported at a 1.37× decode speedup on H100.

## DeepSeek-V4.1-Flash (arXiv 2609.19969)

| Property | Value |
|---|---|
| Size | 552B backbone + 196B Engram params. 8B active at prefill, 16B at decode |
| Layers / hidden | 40 / 5120. Every layer has global attention (CSA2) + SWA, except layers 0–1, which are SWA-only |
| MoE | DeepSeekMoE shared + fine-grained routed experts (384 routed + 1 shared, 6 active, per the model card) |
| Training | **From scratch**, 45T tokens, 64K sequence length extended to 1M at 34T. Muon optimizer |
| Attention | 64 query heads, head dim 512, query compression dim 1280, 8 output projection groups |

### Causal Encoder-Decoder (CED)

Inspired by YOCO. Layers 0–19 form the encoder and layers 20–39 the decoder.
Decoder **global** KV is projected from the *last encoder layer's* hidden state
with layer-dependent weights, so prefill only has to run the encoder. SWA KV
stays layer-local, which is why SWA Bounded Replay (below) is needed.

### CSA2 layer assignment (confirmed)

| Block | Layers | Compression m | Groups | Modes per group |
|---|---|---|---|---|
| Encoder | 2–19 (18 layers) | 2 | 3 × 6 | Full, then 5 × Reuse |
| Decoder | 20–39 | 1 (uncompressed) | 5 × 4 | group 1: Full + 3 × Reuse; groups 2–5: Reindex + 3 × Reuse |

- **Full** computes main KV, derives indexer K *by projecting main KV*, and
  selects Top-K. **Reindex** reuses main KV and indexer K and rescores with its
  own indexer Q. **Reuse** reuses both the KV and the Top-K indices. All modes
  compute their own Q and SWA KV.
- Indexer: 32 heads, head dim 128, **Top-K = 512**.
- Compared with V4's CSA, the compressor drops overlapping windows and
  absolute position embeddings.
- **Hierarchical Sparse Indexer** (decoder only, added in post-training): the
  first Full layer picks up to 2,048 blocks × 8 positions (16,384
  candidates), and later Reindex layers only score that pool.

### FP4 main KV cache (confirmed)

- The format is **E2M1 with one E4M3 scale per 16 channels**: NVFP4 without
  the second-level global scale, 4.5 bits per element. The indexer Q/K use
  MXFP4.
- It is added by **QAT during post-training**. The cache is quantized *after*
  RoPE, and the RoPE and non-RoPE parts use the same format.
- **SWA KV stays FP8** because it is sensitive to quantization.
- Result: **890 bytes/token** of global KV, about 1/4 of V4-Flash.

### SWA Bounded Replay

Instead of persisting SWA KV, the model replays only the last `n_win` tokens
and truncates SWA to that segment. The reconstructed state is approximate, so
replay is simulated during post-training. The persistent KV cache ends up at
about 1/8 of V4-Flash.

### Speculative decoding: DSpark, not MTP

V4.1 **drops the MTP module from pre-training**. It trains a DSpark drafter
after pre-training with the **backbone frozen**, then keeps training it
alongside post-training without backpropagating into the backbone. The drafter
is 3 blocks with a 128-token sliding window, drafts 5 positions per pass with a
Markov head plus a confidence head, and schedules verification length from
predicted acceptance.

## What this means for "V4.1-Flash-style KV on Bonsai 2 27B"

**Baseline (measured):** 16 GQA layers × 2 × 4 KV heads × 256 × 2 bytes =
**64 KiB/token** at bf16, so **16 GiB at 262K tokens**, 2.4× the 6.65 GiB of
PQ2_0 weights. The 48 DeltaNet layers add a fixed ~147 MiB/sequence of state.

| Technique | Fit for Bonsai 2 27B | Effort / risk |
|---|---|---|
| FP4 KV (E2M1 + E4M3/16) | Direct fit on the 16 GQA layers: 64 → 18 KiB/token, 16 → 4.5 GiB at 262K | **Low.** The PrismML fork already has quantized KV types (it documents q4/q5 KV and FA-quant build flags). DeepSeek adds QAT to recover quality; for us that would be a short post-training QAT with distillation from the unquantized-KV model. Quantize after RoPE, as in the paper |
| Cross-layer KV reuse (CSA2 Full/Reuse) | Plausible: share KV within groups of the 16 full-attention layers. Groups of 4 (V4.1 decoder style) give 4 KV sets: 16 KiB/token at bf16, ~4.5 KiB with FP4 | **Medium–high, and untested.** V4.1 trained sharing from scratch over 45T tokens. Converting a trained model is our extrapolation. Bonsai's full-attention layers sit 4 blocks apart with DeltaNet layers between them, unlike V4.1's contiguous groups. Needs retraining the Reuse layers' Q/O with distillation, ternary QAT, and runtime support in the fork |
| Sparse indexer (Top-K 512, Reindex) | Would need new indexer modules on the full-attention layers | **High.** New params to train (ternary or kept at higher precision), long-context retraining, new kernels |
| CED encoder-decoder | Would restructure the layer graph so upper-half global KV comes from the middle layer | **Very high.** Effectively continued pretraining |
| SWA Bounded Replay | **Doesn't apply**: Qwen3.8 has no SWA. DeltaNet does the local work and its state is already fixed-size | n/a |

MTP and drafter options (from the MTP section above):

1. **Port the Qwen3.8 MTP head** (424.7M params): ternarize it with the same
   Hadamard rotation, then finetune it against the frozen Bonsai trunk.
   Cheapest, and it reuses the base model's training. Needs the fork's MTP
   support (#205).
2. **Train a DSpark-style drafter** with the trunk frozen, as both DeepSeek
   V4.1 and 1-bit Bonsai 27B did. More work, but the reported payoff is proven
   (1.37× on H100).

### Constraints that apply to every Bonsai plugin

- **Hadamard basis** (full contract and loader rules in
  [`prismml-quant-formats.md`](prismml-quant-formats.md)). Any new or modified weight that reads the residual
  stream (width 5120) or the 6144- or 17408-wide activations must be stored
  rotated with the same signs, and `prism.hadamard.weight_names` must be
  updated. The rotation is orthogonal on the input dimension, so
  **singular-value spectra (effective rank of K/V projections) don't change**
  and can be computed directly on the dequantized weights. Per-channel
  statistics (outliers, norms per input channel) *do* change and need the
  rotation undone.
- **Ternary.** New projections are either QAT-trained to ternary or kept at
  higher precision, and either way counted in the bits/weight budget.
- **Runtime.** Architectural changes need kernels in the PrismML llama.cpp
  fork. Stock llama.cpp can't load the files at all.

## Still unconfirmed

- The numeric value of V4.1's SWA window `n_win`: the paper keeps it
  symbolic.
- The per-entry size of V4.1's main KV (head dim 512 is stated; how the 890
  B/token breaks down is not).
- ~~The bit layout inside PQ2_0/PTQ1_0 blocks~~: **resolved**. See
  [`prismml-quant-formats.md`](prismml-quant-formats.md). It was read from the
  fork's source and verified on the real weights: decoded and un-rotated, they
  reach cosine 0.88 against Qwen3.8-27B.
- Whether the fork's MTP support (#205) expects a specific tensor layout.

## Sources

- [DeepSeek-V4.1-Flash paper (arXiv 2609.19969)](https://arxiv.org/abs/2609.19969): §2.1–2.4, §3.2.2, §4.2
- [prism-ml/Ternary-Bonsai-2-27B-gguf model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and [KNOWN_ISSUES.md](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/KNOWN_ISSUES.md)
- [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (1-bit Bonsai 27B, DSpark drafter)
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): `config.json`, `model.safetensors.index.json`
- [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)

# PrismML quantization formats and Hadamard rotation

This comes from the source of [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp),
branch `prism`, commit `adfffbe` (2026-09-25). It was verified against the
released Ternary-Bonsai-2-27B weights (see [Verification](#verification)). The
implementation is in `crates/mb-formats/src/dequant.rs` (block decoding) and
`crates/mb-ir/src/rotation.rs` (rotation).

## Block layouts

Both types have one FP16 scale per 128 weights. Blocks run along a row (the
input dimension), so every input width must be a multiple of 128.

### PQ2_0 (ggml type 142, 34 bytes/block, 2.125 bpw)

Source: `ggml-common.h` `block_pq2_0`, `ggml-quants.c` `dequantize_row_pq2_0`.

```c
typedef struct { ggml_half d; uint8_t qs[32]; } block_pq2_0;
// element j: q = (qs[j/4] >> 2*(j%4)) & 3;   value = (q - 1) * d
//            00 = -1, 01 = 0, 10 = +1, 11 = +2 (never used by ternary checkpoints)
```

This is the same codec as the fork's Q2_0 (type 42, group 64), but with a
group of 128. Older Bonsai files that declare type 42 are the group-64 variant.

### PTQ1_0 (ggml type 143, 28 bytes/block, 1.75 bpw)

Source: `block_ptq1_0`, `dequantize_row_ptq1_0`, `ptq1_0_stages = {32, 16, 8}`.

```c
typedef struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; } block_ptq1_0;  // note: scale LAST
```

The trit packing is the same as upstream TQ1_0, but with a group of 128
instead of 256. Each byte stores trits in base 3, scaled by 256/243 with
ceiling division, most significant trit first. Trit `n` of byte `b` is
`((uint8_t)(b * 3^n) * 3) >> 8`, which gives 0, 1 or 2, decoding to −1, 0 or +1.

| Elements | Source |
|---|---|
| 0–79 | `qs[0..16]`, 5 trits each: element `n*16 + m` is trit `n` of `qs[m]` |
| 80–119 | `qs[16..24]`: element `80 + n*8 + m` is trit `n` of `qs[16+m]` |
| 120–127 | `qh[0..2]`, 4 trits each: element `120 + n*2 + h` is trit `n` of `qh[h]` |

The chunking comes from the stages loop: a stage `c` applies while
`j + c <= 24`, so with 24 bytes the chunks are 16 and then 8. For
already-ternary group-128 weights, the format is lossless relative to PQ2_0.

## Hadamard weight rotation (`prism.hadamard.*`)

Source: `src/llama-model.cpp` (loader, around lines 1196–1355 and 1972–2140)
and `src/llama-graph.cpp` (`build_lora_mm`, `build_embd_rows`).

**Runtime math.** For each folded weight, the fork computes

```
y = W_stored · H(s ⊙ P(x))
```

- `H` is the normalized Sylvester-Walsh-Hadamard matrix, applied to each
  `block_size` chunk of the input dimension:
  `H[r][c] = (-1)^popcount(r & c) / sqrt(B)`. It is symmetric and orthogonal.
- `s` is the ±1 sign vector for the weight's **input width** (`ne[0]`).
- `P` is the identity, except for `ssm_out` when `gdn_v_grouped` is set: then
  it permutes activations from llama.cpp's tiled V-head order
  `[hd, nk, rep]` to grouped order `[hd, rep, nk]`.

**Recovering the original weights.** Per row, `w = s ⊙ (H · w_stored)`,
implemented as `WeightRotation::to_primal`. For `ssm_out`, the result is in
grouped (HF training) order. That matches `linear_attn.out_proj.weight` in the
Qwen3.8 checkpoint directly.

**Inverse tables.** `token_embd.weight` stores latent rows. After lookup, the
runtime computes `h = s ⊙ (H z)`, which is the same formula.

### Metadata keys and the loader's checks

The loader refuses files that break any of these:

| Key | Rule |
|---|---|
| `version` | 1 or 2. Version 2 requires `tied_output = true` and no `output.weight` |
| `block_size` | Must divide every folded weight's input dimension. Bonsai 2 uses 1024 |
| `transform` | Only `normalized-sylvester-walsh-hadamard` |
| `axis` | Only `input-last-dimension` |
| `sign_mode` | `identity` or `explicit`. Explicit requires non-empty `sign_widths`, and `sign_values` must be the concatenated ±1 vectors (lengths must match exactly) |
| `weight_names` | Must be foldable kinds only: `attn_{q,k,v,qkv,gate,output}`, `ffn_{gate,up,down}` (+ `_exps`, `_shexp`, `gate_up_exps`), `ssm_out`, and `output.weight`. Anything else is refused |
| `inverse_weight_names` | Only `token_embd.weight` |
| architecture | Only `llama`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next` and `dspark` |
| `gdn_v_grouped` | Optional bool, used for `ssm_out` |

### What this means for surgery

- A new weight on the residual stream (e.g. an MTP or drafter projection) can
  be left unrotated, since it simply doesn't go in `weight_names`. To store it
  ternary like its neighbours, fold it with `w_stored = H(s ⊙ w)` using the
  signs for its input width, and add it to `weight_names`.
- Tensor kinds outside the foldable list (e.g. `ssm_alpha`, `ssm_beta`, norms,
  any new custom kind) **cannot** be folded. The loader refuses them, so they
  must stay in the primal basis.
- A new input width needs its own sign vector appended to
  `sign_widths`/`sign_values`, or the loader fails with "no sign vector for
  width".
- `dspark` drafters may only fold `output.weight`.

## Verification

Checked on 2026-09-27 against the real files. Tensors were fetched with HTTP
range requests into sparse copies of the GGUFs and of the Qwen3.8-27B
safetensors shards.

| Tensor (width) | cos(stored, Qwen3.8 original) | cos(`to_primal`, original) | PQ2_0 = PTQ1_0 |
|---|---|---|---|
| `blk.3.attn_k` (5120) | −0.0005 | **0.883** | 5,242,880 / 5,242,880 identical |
| `blk.0.ffn_down` (17408) | −0.0000 | **0.880** | 89,128,960 / 89,128,960 identical |
| `blk.0.ssm_out` (6144, grouped) | −0.0002 | **0.886** | 31,457,280 / 31,457,280 identical |

Decoding, block order, signs, width selection and `ssm_out` ordering are all
correct: an error in any of them would push the cosine to about 0. The two
packings decode to the same weights. PQ2_0 code 3 (+2) never occurs, and
**32.8%** of the ternary weights are zero. The ~0.88 cosine is the similarity
between the QAT-trained ternary weights and the original BF16 weights; this is
itself a useful statistic for judging how far QAT moved the model.

The golden vectors in `crates/mb-formats/src/dequant_golden.rs` come from the
fork's own reference quantizer and dequantizer, via `scripts/prism-golden.sh`.

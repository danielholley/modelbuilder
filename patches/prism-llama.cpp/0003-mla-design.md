# Design: running `surgery mla` checkpoints in the fork

Status: implemented as `0003-mla.patch`, following this design, with two
changes: Q is permuted into the cached `[rope; nope]` head order by a constant
matrix (`attn_mla_perm`) instead of permuting weights offline, and the kept
RoPE frequencies come from per-layer frequency-factor tensors (Q: a huge factor
switches a frequency off; K: factors map the r-dim NeoX rope onto the kept
frequencies). The V cache row is 1 wide (zeros) so the cache stays uniform.

## What has to run

`modelbuilder surgery mla` (MHA2MLA, arXiv 2502.14837) gives each converted
layer a latent `c = kv_a·x` (`kv_rank` values), `[k_nope; v] = kv_b·c`, and
`k_rope = k_rope_proj·x` on the `rope_dim` kept dimensions per KV head. The
rest of the layer is unchanged: Qwen3.5's per-head K RMSNorm runs on the
reassembled key, then partial RoPE on the kept frequency pairs only.

DeepSeek's MLA absorbs `kv_b` into the query and output projections and
attends in latent space. That is impossible here: the K RMSNorm sits between
`kv_b` and the dot product. So the cache holds the latent and each step
decompresses the cached rows, which trades compute for the cache saving.

## Cache row (per token, per converted layer)

`[ c (kv_rank) | rope(k_rope ∘ w_rope) (n_kv·rope_dim) | ‖k_rope‖² (n_kv) ]`

- RMSNorm is a per-(token, head) scalar `s = 1/rms(k)` times the weight, and
  RoPE is linear, so `rope(norm(k))_rope = s · rope(k_rope ∘ w_rope)`. The
  roped, weighted part can be written at store time with the token's own
  position. Cached tokens' positions are then not needed when reading.
- `rms(k)` needs `‖k_rope‖² + ‖k_nope‖²`. RoPE keeps pair norms, but the stored
  part is weighted, so store `‖k_rope‖²` (one float per head) next to it;
  `‖k_nope‖²` comes from the decompressed rows.

Cache width: `kv_rank + n_kv·(rope_dim + 1)` instead of `2·n_kv·head_dim`;
for Bonsai 2 at `kv_rank 512`, `rope_dim 32`: 644 vs 2048 values.

## Layout trick for RoPE and Q

Permute each head's dimensions offline to `[kept rope dims as NeoX pairs |
nope dims]`, applying the same permutation to `q_proj` rows (per head, query
half only for gated attention), `q_norm`/`k_norm` weights, and `kv_b`/`k_rope`
rows. Attention is invariant to a shared permutation of Q and K dims. RoPE then
acts on the first `rope_dim` dims with `n_rot = rope_dim`; the kept
frequencies come from `ggml_rope_ext`'s `freq_factors`:
`factor[j] = base^(-2j/rope_dim) / base^(-2·kept[j]/rotary_dim)`.

## Graph, per converted layer

1. `c = kv_a·x`, `kr = k_rope_proj·x` (per head), `n2 = sum(kr²)` per head;
   store `[c | rope(kr ∘ w_rope) | n2]` with `cpy_k` (the V cache is unused:
   give these layers a 1-wide V row, or add per-layer cache widths).
2. Read the cache view `[width, n_kv_cells]`; `[k_nope; v] = kv_b·c`.
3. `s = 1/sqrt((n2 + sum(k_nope²))/head_dim + eps)` per (cell, head);
   `K = s · [rope_part ; k_nope ∘ w_nope]`, `V = v`.
4. `Q`: projection, permuted Q norm, RoPE as above; then `build_attn_mha`
   with the explicit `K`, `V` and the layer's mask; output gate and `wo` as
   now.

## Work items

- hparams: per-layer `mla_rank`, `mla_rope_dim`, kept frequencies; per-layer
  cache widths (`n_embd_k_gqa(il)` / `n_embd_v_gqa(il)`).
- loader: `attn_kv_a`, `attn_kv_b`, `attn_k_rope` for converted layers;
  `attn_k`/`attn_v` optional there.
- graph: the path above in `qwen35.cpp`'s attention.
- modelbuilder: a `surgery mla-gguf` step that writes the trained latent
  tensors (from `job trunk-distill --mla-rank`) into the GGUF in the permuted
  layout, encoded in the source types, plus the metadata.
- check: `probe hf-vs-gguf` against the PyTorch MLA student (`hf/mla.py`),
  as done for KV sharing.

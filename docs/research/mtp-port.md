# MTP head port: Qwen3.8-27B → Ternary-Bonsai-2-27B

First end-to-end surgery: `modelbuilder surgery mtp` ports the base model's
multi-token-prediction head into an MTP-only GGUF sidecar. The sidecar runs
unchanged in PrismML's llama.cpp fork as a speculative-decoding draft for
Bonsai 2.

Measured on 2026-09-27 with:
- `prism-ml/Ternary-Bonsai-2-27B-gguf` PQ2_0 (7,206,168,928 bytes, the full
  file);
- the `mtp.*` tensors of `Qwen/Qwen3.8-27B` (BF16);
- PrismML-Eng/llama.cpp `prism` @ adfffbe, built for CPU;
- a 4-core x86 box with 15 GB RAM.

## What the sidecar contains

`modelbuilder surgery mtp Ternary-Bonsai-2-27B-PQ2_0.gguf --from Qwen3.8-27B -o …-mtp.gguf`
writes 1.43 GiB in 3.7 s:

| Tensors | Source | Transform |
|---|---|---|
| `token_embd`, `output_norm`, `output` | Bonsai 2 | copied byte for byte (PQ2_0 / F32) |
| `blk.64.attn_{q,k,v,output}`, `ffn_{gate,up,down}`, `nextn.eh_proj` | Qwen3.8 `mtp.*` | copied as BF16 |
| `blk.64.{attn_norm, post_attention_norm, attn_q_norm, attn_k_norm}`, `nextn.{enorm, hnorm, shared_head_norm}` | Qwen3.8 `mtp.*` | +1 → F32 |

Metadata:
- `qwen35.block_count` goes from 64 to 65, and `qwen35.nextn_predict_layers = 1`
  is added.
- `prism.hadamard.weight_names` is filtered to the one rotated tensor in the
  file (`output.weight`), and `inverse_weight_names` keeps `token_embd.weight`.
- `modelbuilder.*` provenance keys record where the head came from.
- The head's own weights are unrotated and unlisted, so the runtime uses a
  plain matmul for them.

The conversion rules come from the fork's `conversion/qwen.py`. The "+1 on
every norm" rule was also checked against Bonsai 2's own F32 norms: they match
Qwen3.8's plus one with cosine 1.0000 and mean absolute error of about 0.005,
for `attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm` and
`output_norm`.

## Results

Run command:

```sh
llama-speculative-simple -m Bonsai-2-27B-PQ2_0.gguf -md Bonsai-2-27B-mtp.gguf \
    --spec-type draft-mtp -n 64 --temp 0 -t 4
```

The plain baseline runs with `--spec-type ngram-simple`, which the model card's
known issues confirm is a no-op on this model. That gives the same binary and
settings, just without drafting.

| Prompt | Plain | Ported MTP head | Accepted | Speedup | Control: norms without +1 |
|---|---|---|---|---|---|
| Python function (raw completion) | 1.21 t/s | 2.24 t/s | 48/57 (84%) | 1.86× | 0/192, 0.81 t/s |
| "Why is the sky blue" (raw) | 1.34 t/s | 1.84 t/s | 37/85 (44%) | 1.37× | 0/192, 0.80 t/s |
| Story opening (chat template) | 1.36 t/s | 2.20 t/s | 41/70 (59%) | 1.62× | 0/192, 0.81 t/s |
| Speed word problem (chat template) | 1.32 t/s | 2.74 t/s | 48/60 (80%) | 2.07× | 0/192, 0.81 t/s |

Overall, 174 of 272 drafted tokens were accepted (64%), for a 1.37–2.07×
decode speedup on this CPU. A first "capital of France" smoke test accepted
20 of 38 (53%) at 1.44×.

- **Lossless:** in every run the plain output is an exact prefix of the MTP
  output. The MTP runs can overshoot the 64-token limit by 1–3 tokens, because
  accepted drafts land in batches.
- **The port is what produces the acceptance.** The control sidecar, identical
  except that its 7 norms skip the +1, accepted 0 of 192 drafts in every
  prompt, and the wasted drafts made it slower than plain decoding.

## What this means for the `mtp` plugin

- The plugin's plan treats the ported head as an initialization that needs the
  `mtp-align` stage (50–300M tokens). The measurement shows the head is
  already useful without that stage: Bonsai 2's QAT kept the trunk's hidden
  states close enough to Qwen3.8's that the base model's head drafts well.
  Alignment should still raise acceptance, particularly on prose, where it is
  lowest (44%).
- Speedups are hardware-dependent. On this memory-bound CPU, verifying a
  3-token draft costs little more than one decode step. PrismML reports its
  DSpark drafter is not a net win on Apple Silicon at batch 1, and gives
  1.37× on H100. Measure on the deployment target.

## Not measured yet

- GPU (CUDA/Metal) speedups, and longer generations.
- Draft depth other than the fork's default (`n_draft = 3`).
- The PTQ1_0 packing. Its tensors decode to the same values as PQ2_0 (see
  `prismml-quant-formats.md`), so acceptance should match, but kernel speed
  differs.
- Acceptance after the `mtp-align` training stage.

## Reproduce

1. Download the Bonsai 2 PQ2_0 GGUF, plus Qwen3.8-27B's `config.json`,
   `model.safetensors.index.json` and the shard ranges holding `mtp.*`. Sparse
   shard files with only those ranges filled are enough; see "Commands" in
   `CLAUDE.md`.
2. Build the PrismML fork (`cmake -B build -DGGML_NATIVE=ON && cmake --build build`).
3. Run `modelbuilder surgery mtp …`, then the `llama-speculative-simple`
   command above.

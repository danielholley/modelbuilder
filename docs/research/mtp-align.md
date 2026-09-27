# MTP alignment: training a head against a frozen low-bit trunk

How `modelbuilder job mtp-align` trains an MTP head for a target such as
Ternary-Bonsai-2-27B, and how the pieces were checked against the PrismML
llama.cpp fork. The port itself (HF head → GGUF sidecar) is in
[mtp-port.md](mtp-port.md).

## Why align at all

A ported head was trained against its reference trunk (Qwen3.8-27B). Bonsai 2 is
the same architecture after ternary QAT, so its final hidden states are close to
the reference's but not identical. The port already drafts well (64% acceptance
in `mtp-port.md`), and `mtp_align` fine-tunes the head on the *target's* hidden
states to close the rest of the gap. Only the head trains: 425M parameters on
Bonsai 2, against a 27B frozen trunk.

## Design

```
target.gguf ──llama.cpp (PrismML fork)──► features/          (tokens, trunk hidden states)
target.gguf ──modelbuilder export-tensors──► frozen.safetensors (token_embd, output; primal basis)
reference/  ──────────────────────────────► config.json, initial mtp.* head
                         │
                job.json (schema/job-spec.v1)
                         │
             python -m modelbuilder_train run   ──► JSONL events (schema/events.v1)
                         │
                   mtp-head/model.safetensors
                         │
             modelbuilder surgery (port_mtp_sidecar) ──► <target>-mtp-aligned.gguf
```

- **The trunk never runs in PyTorch.** The target's ternary PQ2_0 weights with a
  folded-in Hadamard rotation have no PyTorch loader, and a frozen trunk only
  contributes its outputs anyway. Features are computed once with the runtime
  that serves the model: `llama-embedding --pooling none --embd-normalize -1`
  returns, for every token, the hidden state after the final norm. That is the
  tensor the fork's `graph_mtp` feeds the head (`h_nextn`,
  `src/models/qwen35.cpp`). `-np 2` is required on hybrid models: without it the
  tool sizes its recurrent-state cache for 256 sequences (38 GB for Bonsai 2 and
  a crash).
- **The frozen embedding and output head are exported decoded and un-rotated**
  (`export-tensors`, bf16). The head sees `embed(x[t+1])` in the primal basis,
  and the loss is computed with the primal `output.weight`.
- **The head is a PyTorch copy of `graph_mtp`** (`python/modelbuilder_train/mtp/model.py`):
  `fc(concat(enorm(e), hnorm(h)))`, then one gated-attention decoder block (Q/K
  norms, partial NeoX RoPE, sigmoid output gate interleaved per head in
  `q_proj`), then SwiGLU and the shared-head norm. Norms are zero-centered
  (`x/rms·(1+w)`) as in the HF checkpoint. The sidecar writer adds the +1.
- **Training pairs** are `(h[t], x[t+1]) → x[t+2]` at RoPE positions `1..n-1`,
  which is what the runtime feeds the draft head. Logits are computed in
  256-row chunks with activation checkpointing, so the 248K-vocab output head
  never holds more than a chunk of logits.

## Validation against the fork

These checks use the Bonsai 2 PQ2_0 GGUF, the Qwen3.8-27B `mtp.*` tensors and the
fork built for CPU.

| Check | Result |
|---|---|
| Trunk sanity: argmax of the exported `output.weight` over the llama.cpp features reproduces the tokens the fork generated | 95–100% per text |
| Ported head, t+2 top-1 on generated tokens (scratch script) | 84.4% |
| Same, through the shipped pipeline (`extract-features` + `evaluate-mtp`), all 337 targets of 4 texts including prompt tokens | 76.0% (CE 1.18) |
| Control: head with norms applied as `x/rms·w` (no +1) | 0% |

The ported head's accuracy tracks the 64% runtime acceptance, and the no-+1
control collapses. Together these show that the features, the frozen tensors and
the head's math match the runtime graph. Accuracy on prompt tokens is lower than
on the model's own generations, as expected.

## What has not been run yet

- **An actual alignment run on Bonsai 2.** The training loop is tested on
  synthetic fixtures (`python/tests/test_mtp.py`, and `test_e2e.py` through the
  Rust binary): loss falls, and the sidecar is written and inspected. On Bonsai 2
  it needs a real text corpus (ideally the target's own generations, as for
  distillation) and more memory than the 16 GB CPU box used here. The head plus
  AdamW state is about 7 GB in fp32, and the embedding and output head are 2.5 GB
  each in bf16.
- **Acceptance of an aligned sidecar in llama.cpp.** This is the number that
  matters. `evaluate-mtp` top-1 is a proxy for it.

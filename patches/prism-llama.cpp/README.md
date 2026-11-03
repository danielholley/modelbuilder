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

Other architectures read the key (and share the cache) but their graphs still
pass their own K/V to `build_attn`, which would write into the producer's
cache; extend their graphs the same way before using the key with them.

Checked on the real model: see `docs/STATUS.md` (a pruned Ternary-Bonsai-2-27B
with `kv-share --group 2`, compared token by token with the PyTorch model that
`trunk_distill` trains with the same sharing).

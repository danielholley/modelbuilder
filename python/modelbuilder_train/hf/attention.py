"""An attention implementation for any transformers model: KV-cache fake-quantization and cross-layer KV sharing.

Modern transformers models call ``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``
with the key and value *after* RoPE and Q/K norms, i.e. exactly what the KV
cache stores. Registering ``"modelbuilder"`` there changes the cache semantics
without touching any model's code:

- **Quantized cache** (``kv_format``): K and V are fake-quantized as the cache
  would store them (straight-through gradients).
- **Shared cache** (``share``): a map from each reuse layer to its producer
  layer. A producer stores its (quantized) K/V; a reuse layer attends over the
  producer's instead of its own, as in cross-layer KV sharing (CSA2 Full/Reuse,
  YOCO-style).

The setting lives on each attention module (``module._mb``), so one process can
run a teacher (plain) and a student (modified) pass by toggling ``enabled``.
Linear-attention layers (no KV cache) never call this and are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from modelbuilder_train.hf import fakequant

NAME = "modelbuilder"


@dataclass
class CacheBehavior:
    kv_format: str | None = None
    #: reuse layer index -> producer layer index
    share: dict[int, int] = field(default_factory=dict)
    enabled: bool = True
    _cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def reset(self) -> None:
        self._cache.clear()


def _cache_quant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Fake-quantizes ``[batch, kv_heads, seq, head_dim]`` the way llama.cpp stores the cache.

    A cache row is one token's ``kv_heads * head_dim`` values, so blocks run
    along the heads of a token and may span heads when ``head_dim`` is smaller
    than the block.
    """
    b, h, t, d = x.shape
    rows = x.transpose(1, 2).reshape(b, t, h * d)
    return fakequant.kv(rows, fmt).reshape(b, t, h, d).transpose(1, 2)


def _attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    b: CacheBehavior | None = getattr(module, "_mb", None)
    if b is not None and b.enabled:
        layer = getattr(module, "layer_idx", None)
        producer = b.share.get(layer)
        if producer is not None and producer != layer:
            if producer not in b._cache:
                raise RuntimeError(f"layer {layer} reuses layer {producer}'s KV, which hasn't run yet")
            key, value = b._cache[producer]
        else:
            if b.kv_format:
                key, value = _cache_quant(key, b.kv_format), _cache_quant(value, b.kv_format)
            if b.share:
                b._cache[layer] = (key, value)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def register() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, _attention)


def attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Modules with a KV cache: those with ``k_proj``/``v_proj`` and a ``layer_idx``."""
    return [m for m in model.modules() if hasattr(m, "k_proj") and hasattr(m, "v_proj") and hasattr(m, "layer_idx")]


def attach(model: torch.nn.Module, behavior: CacheBehavior) -> list[int]:
    """Uses ``behavior`` in every attention module; returns their layer indices."""
    mods = attention_modules(model)
    if not mods:
        raise ValueError("the model has no attention modules with k_proj/v_proj and layer_idx")
    for m in mods:
        m._mb = behavior
    return sorted(m.layer_idx for m in mods)


def share_groups(layers: list[int], group: int) -> dict[int, int]:
    """Groups consecutive attention layers ``group`` at a time; the first of each group produces."""
    out = {}
    for i in range(0, len(layers), group):
        g = layers[i : i + group]
        for li in g:
            out[li] = g[0]
    return out

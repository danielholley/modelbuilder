"""DSpark drafters (DeepSeek-V4.1-Flash, arXiv 2609.19969), following DeepSeek's DeepSpec.

This is our own implementation of the semantics of the reference trainer,
DeepSpec (MIT License, Copyright (c) 2026 The DeepSpec Authors):
``deepspec/modeling/dspark/{common.py, markov_head.py, loss.py,
qwen3/modeling.py, qwen3/config.py}`` and ``scripts/data/prepare_target_cache.py``.
Parameter names match DeepSpec's ``Qwen3DSparkModel`` state dict (and so the
published ``deepseek-ai/dspark_*`` checkpoints), which is what the PrismML
llama.cpp fork's converter reads (``conversion/qwen.py``, ``DSparkModel``).

The drafter, for a target with hidden size ``h``:

- **Context.** Hidden states tapped at ``target_layer_ids`` (decoder-layer
  outputs; -1 is the embedding) are concatenated per position and projected
  ``fc: n·h → h`` then ``hidden_norm``. They serve only as keys and values.
- **Blocks.** Anchor positions are sampled where the anchor and the token
  after it are both supervised. Each block is ``[x[anchor], MASK, ..., MASK]``
  (``block_size`` tokens) through the frozen target embedding, at positions
  ``anchor .. anchor + block_size - 1``. A block attends to the context
  strictly before its anchor and to all of its own rows (non-causal).
- **Decoder layers** of the target's shape (Qwen3: Q/K RMSNorm, RoPE, GQA,
  SwiGLU), then ``norm`` and the frozen target LM head. Slot ``i`` predicts
  ``x[anchor + i + 1]``.
- **Markov head** (vanilla): logits of slot ``i`` get ``markov_w2(markov_w1[prev])``
  with ``prev`` the anchor for slot 0 and ``x[anchor + i]`` after it.
- **Confidence head**: ``proj([hidden; markov_w1[prev]]) → 1`` predicts each
  slot's acceptance rate.
- **Loss** over supervised slots (a contiguous prefix of each block), weighted
  ``exp(-i / gamma)``: ``ce_alpha·CE + l1_alpha·‖p_draft − p_target‖₁ +
  confidence_alpha·BCE(confidence, 1 − ½‖p_draft − p_target‖₁)``, with
  ``p_target`` the target's own next-token distribution at the same position.

Departures from the reference, none of which changes the math: a dense boolean
attention mask instead of flex attention; blocks with no valid anchor attend
to themselves (instead of nothing), since they are excluded from the loss and
an empty softmax row would produce NaNs; one process (no loss all-reduce).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from modelbuilder_train.mtp.model import RMSNorm, SwiGLU, apply_partial_rope

#: DeepSpec's defaults (config/dspark/dspark_qwen3_*.py).
DEFAULTS = {
    "block_size": 7,
    "num_draft_layers": 5,
    "num_anchors": 512,
    "markov_rank": 256,
    "loss_decay_gamma": 4.0,
    "ce_loss_alpha": 0.1,
    "l1_loss_alpha": 0.9,
    "confidence_head_alpha": 1.0,
}


@dataclass(frozen=True)
class DSparkConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rotary_dim: int
    rms_eps: float
    attention_bias: bool
    num_layers: int
    target_layer_ids: tuple[int, ...]
    block_size: int
    mask_token_id: int
    num_anchors: int
    markov_rank: int
    confidence: bool
    confidence_with_markov: bool = True

    @classmethod
    def from_target(cls, target_cfg: dict, **kw) -> DSparkConfig:
        """The drafter takes the target's layer shape (``build_draft_config`` copies the target config)."""
        t = target_cfg.get("text_config", target_cfg)
        rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
        head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        partial = t.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0))
        n_target = int(t["num_hidden_layers"])
        ids = kw.pop("target_layer_ids", None) or default_target_layers(n_target)
        validate_target_layer_ids(ids, n_target)
        return cls(
            hidden_size=t["hidden_size"],
            num_heads=t["num_attention_heads"],
            num_kv_heads=t.get("num_key_value_heads", t["num_attention_heads"]),
            head_dim=head_dim,
            intermediate_size=t["intermediate_size"],
            vocab_size=t["vocab_size"],
            rope_theta=float(t.get("rope_theta", rope.get("rope_theta", 10_000.0))),
            rotary_dim=int(head_dim * partial),
            rms_eps=float(t.get("rms_norm_eps", 1e-6)),
            attention_bias=bool(t.get("attention_bias", False)),
            target_layer_ids=tuple(ids),
            **kw,
        )

    def hf_config(self, target_cfg: dict) -> dict:
        """``config.json`` in the ``Qwen3DSparkModel`` format (the target's config plus the DSpark fields)."""
        t = dict(target_cfg.get("text_config", target_cfg))
        for k in ("text_config", "vision_config", "mtp_num_hidden_layers", "layer_types", "auto_map"):
            t.pop(k, None)
        t.update(
            architectures=["Qwen3DSparkModel"],
            model_type="qwen3",
            num_target_layers=int(target_cfg.get("text_config", target_cfg)["num_hidden_layers"]),
            num_hidden_layers=self.num_layers,
            layer_types=["full_attention"] * self.num_layers,
            block_size=self.block_size,
            tie_word_embeddings=False,
            mask_token_id=self.mask_token_id,
            target_layer_ids=list(self.target_layer_ids),
            num_anchors=self.num_anchors,
            enable_confidence_head=self.confidence,
            markov_rank=self.markov_rank,
            head_dim=self.head_dim,
        )
        if self.confidence:
            t["confidence_head_with_markov"] = self.confidence_with_markov
        if self.markov_rank > 0:
            t["markov_head_type"] = "vanilla"
        return t


def default_target_layers(n: int, taps: int = 5) -> list[int]:
    """Five taps from layer 1 in steps of ``(n - 4) // 4``, which reproduces the published configs:
    DeepSpec's Qwen3-4B ([1, 9, 17, 25, 33] of 36) and a Qwen3.6-27B DSpark ([1, 16, 31, 46, 61] of 64)."""
    step = max(1, (n - 4) // (taps - 1)) if taps > 1 else 0
    return sorted({min(n - 1, 1 + k * step) for k in range(taps)})


def validate_target_layer_ids(ids, n: int) -> None:
    prev = None
    for i in ids:
        if not (i == -1 or 0 <= i < n):
            raise ValueError(f"target layer {i} is outside -1 ∪ [0, {n - 1}]")
        if prev is not None and i <= prev:
            raise ValueError("target_layer_ids must be strictly increasing")
        prev = i
    if not ids:
        raise ValueError("target_layer_ids must not be empty")


class Attention(nn.Module):
    """Qwen3 attention whose keys/values are [context; block] and queries the block only."""

    def __init__(self, c: DSparkConfig) -> None:
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.hidden_size, c.num_heads * c.head_dim, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=c.attention_bias)
        self.o_proj = nn.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=c.attention_bias)
        self.q_norm = RMSNorm(c.head_dim, c.rms_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_eps)

    def forward(self, x, ctx, q_pos, kv_pos, mask) -> torch.Tensor:
        c = self.c
        b, q_len, _ = x.shape
        kv = torch.cat([ctx, x], dim=1)
        q = self.q_norm(self.q_proj(x).view(b, q_len, c.num_heads, c.head_dim))
        k = self.k_norm(self.k_proj(kv).view(b, kv.shape[1], c.num_kv_heads, c.head_dim))
        v = self.v_proj(kv).view(b, kv.shape[1], c.num_kv_heads, c.head_dim)
        q = apply_partial_rope(q, q_pos, c.rotary_dim, c.rope_theta)
        k = apply_partial_rope(k, kv_pos, c.rotary_dim, c.rope_theta)
        rep = c.num_heads // c.num_kv_heads
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask[None, None],
            scale=1.0 / math.sqrt(c.head_dim),
        )  # fmt: skip
        return self.o_proj(out.transpose(1, 2).reshape(b, q_len, c.num_heads * c.head_dim))


class DecoderLayer(nn.Module):
    def __init__(self, c: DSparkConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x, ctx, q_pos, kv_pos, mask):
        x = x + self.self_attn(self.input_layernorm(x), ctx, q_pos, kv_pos, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class VanillaMarkov(nn.Module):
    def __init__(self, vocab: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, vocab, bias=False)

    def bias(self, prev: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev))


class AcceptRatePredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)


class DSpark(nn.Module):
    def __init__(self, c: DSparkConfig) -> None:
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_eps)
        self.fc = nn.Linear(len(c.target_layer_ids) * c.hidden_size, c.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(c.hidden_size, c.rms_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.markov_head = VanillaMarkov(c.vocab_size, c.markov_rank) if c.markov_rank > 0 else None
        self.confidence_head = None
        if c.confidence:
            dim = c.hidden_size + (c.markov_rank if c.confidence_with_markov else 0)
            self.confidence_head = AcceptRatePredictor(dim)

    def init_from_target(self, embed: torch.Tensor, lm_head: torch.Tensor) -> None:
        """Copies the target's embedding and LM head and freezes them (``initialize_embeddings_and_head``)."""
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed)
            self.lm_head.weight.copy_(lm_head)
        self.embed_tokens.requires_grad_(False)
        self.lm_head.requires_grad_(False)


def sample_anchors(loss_mask: torch.Tensor, num_anchors: int, gen: torch.Generator | None = None):
    """``sample_anchor_positions``: anchors where the anchor and its next token are supervised (one sequence)."""
    n = loss_mask.shape[0]
    cand = max(n - 1, 0)
    valid = (loss_mask[:cand] > 0.5) & (loss_mask[1 : cand + 1] > 0.5)
    idx = valid.nonzero().flatten()
    if idx.numel() > num_anchors:
        idx = idx[torch.randperm(idx.numel(), generator=gen, device="cpu").to(idx.device)[:num_anchors]]
    anchors = idx.sort().values
    keep = torch.ones_like(anchors, dtype=torch.bool)
    if anchors.numel() < num_anchors:  # pad with masked-out dummy anchors
        pad = num_anchors - anchors.numel()
        anchors = torch.cat([anchors, anchors.new_zeros(pad)])
        keep = torch.cat([keep, keep.new_zeros(pad)])
    return anchors, keep


def attention_mask(anchors, keep, seq_len: int, block: int) -> torch.Tensor:
    """``create_dspark_attention_mask`` as a dense [Q, KV] boolean mask (True = attend)."""
    a = anchors.shape[0]
    q_block = torch.arange(a * block, device=anchors.device) // block
    kv = torch.arange(seq_len + a * block, device=anchors.device)
    ctx = (kv[None, :] < seq_len) & (kv[None, :] < anchors[q_block][:, None])
    own = (kv[None, :] >= seq_len) & (((kv[None, :] - seq_len) // block) == q_block[:, None])
    return (ctx & keep[q_block][:, None]) | own  # invalid blocks still see themselves (no empty rows)


def forward_blocks(model: DSpark, ids, taps, last_hidden, loss_mask, anchors, keep):
    """One sequence: logits, targets, supervision mask, confidence and the target's aligned logits per slot."""
    c = model.config
    n, block, a = ids.shape[0], c.block_size, anchors.shape[0]
    noise = torch.full((a, block), c.mask_token_id, dtype=torch.long, device=ids.device)
    noise[:, 0] = torch.where(keep, ids[anchors], torch.full_like(anchors, c.mask_token_id))
    x = model.embed_tokens(noise.view(1, -1))
    ctx = model.hidden_norm(model.fc(taps[None].to(x.dtype)))
    q_pos = (anchors[:, None] + torch.arange(block, device=ids.device)).view(-1)
    kv_pos = torch.cat([torch.arange(n, device=ids.device), q_pos])
    mask = attention_mask(anchors, keep, n, block)
    for layer in model.layers:
        x = layer(x, ctx, q_pos, kv_pos, mask)
    hidden = model.norm(x).view(a, block, -1)

    label = anchors[:, None] + torch.arange(1, block + 1, device=ids.device)
    safe = torch.where(keep[:, None], label.clamp(max=n - 1), torch.zeros_like(label))
    targets = ids[safe]
    supervised = (label < n) & (loss_mask[safe] > 0.5) & keep[:, None]
    supervised = supervised.int().cumprod(-1).bool()  # a contiguous prefix per block
    prev = torch.cat([ids[anchors][:, None], targets[:, :-1]], dim=1)
    logits = model.lm_head(hidden)
    if model.markov_head is not None:
        logits = logits + model.markov_head.bias(prev)
    conf = None
    if model.confidence_head is not None:
        feats = hidden
        if c.confidence_with_markov:
            feats = torch.cat([hidden, model.markov_head.markov_w1(prev).to(hidden.dtype)], dim=-1)
        conf = model.confidence_head(feats).float()
    with torch.no_grad():
        aligned = model.lm_head(last_hidden[(safe - 1).clamp(min=0)].to(hidden.dtype))
    return logits, targets, supervised, conf, aligned


def dspark_loss(logits, targets, supervised, conf, aligned, *, gamma, ce_alpha, l1_alpha, conf_alpha):
    """``compute_dspark_loss`` for one process. Returns (loss, metrics)."""
    block = logits.shape[1]
    w = supervised.float()
    if gamma and gamma > 0:
        w = w * torch.exp(-torch.arange(block, device=w.device).float() / gamma)
    den = w.sum() + 1e-6
    ce = (F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none") * w.flatten()).sum() / den
    p, q = logits.float().softmax(-1), aligned.float().softmax(-1)
    l1_tok = (p - q).abs().sum(-1)
    l1 = (l1_tok * w).sum() / den
    accept = (1.0 - 0.5 * l1_tok).clamp(0.0, 1.0)
    loss = ce_alpha * ce + l1_alpha * l1
    if conf is not None:
        bce = F.binary_cross_entropy_with_logits(conf, accept.detach(), reduction="none")
        loss = loss + conf_alpha * (bce * w).sum() / den
    with torch.no_grad():
        valid = supervised.float()
        hits = ((logits.argmax(-1) == targets).float() * valid).sum()
        # Expected accepted draft tokens per block (+1 for the target's own token): DeepSpec's tau_probabilistic.
        blocks = supervised.any(-1)
        tau = ((accept * valid).cumprod(-1).sum(-1) + 1.0)[blocks].mean() if blocks.any() else accept.new_zeros(())
    return loss, {"hits": float(hits), "count": int(valid.sum()), "tau": float(tau), "ce": float(ce.detach())}


def tap_target(target, ids: torch.Tensor, layer_ids) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder-layer outputs at ``layer_ids`` (-1: the embedding) concatenated, and the final normed hidden.

    As ``prepare_target_cache.py``: forward hooks on the backbone's layers.
    """
    decoder = target.get_decoder()
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook(i):
        def f(_m, _inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()

        return f

    try:
        for i in layer_ids:
            mod = decoder.embed_tokens if i == -1 else decoder.layers[i]
            handles.append(mod.register_forward_hook(hook(i)))
        with torch.no_grad():
            last = decoder(input_ids=ids[None]).last_hidden_state[0]
        taps = torch.cat([captured[i][0] for i in layer_ids], dim=-1)
    finally:
        for h in handles:
            h.remove()
    return taps, last

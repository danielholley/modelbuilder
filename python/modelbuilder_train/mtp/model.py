"""A DeepSeek-V3-style MTP (next-n) head in PyTorch, configured from an HF config.

The block matches llama.cpp's nextn graph operation for operation, so a head
trained here drafts the same way at inference:

    e = enorm(embed(x[t+1]))            h = hnorm(trunk_hidden[t])
    x = eh_proj(concat(e, h))           # concat order: embedding first
    x = x + attn(input_layernorm(x))    # full attention: optional output gate, Q/K norms, partial RoPE
    x = x + ffn(post_attention_layernorm(x))   # SwiGLU
    logits = lm_head(norm(x))           # predicts x[t+2]

``trunk_hidden`` is the trunk's output *after* its final norm (llama.cpp's
``h_nextn``, the same tensor ``llama-embedding --pooling none`` returns).

What varies between model families is data in :data:`FAMILIES`, keyed by the
HF ``model_type`` and overridable by config keys: whether RMSNorm weights are
zero-centered (applied as ``x·(1+w)``; the GGUF stores ``w+1``), whether the
attention output is gated (``attn_output_gate``), whether Q/K are normed, and
whether projections have biases. Weights use HF names (``mtp.*``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

#: Per-family defaults, keyed by HF ``model_type``. Config keys win when present.
FAMILIES: dict[str, dict] = {
    "qwen3_5": {"norm_offset": 1.0, "attn_gate": True, "qk_norm": True, "attention_bias": False},
    "qwen3_5_text": {"norm_offset": 1.0, "attn_gate": True, "qk_norm": True, "attention_bias": False},
    "qwen3_next": {"norm_offset": 1.0, "attn_gate": True, "qk_norm": True, "attention_bias": False},
    "qwen3": {"norm_offset": 0.0, "attn_gate": False, "qk_norm": True, "attention_bias": False},
    "qwen2": {"norm_offset": 0.0, "attn_gate": False, "qk_norm": False, "attention_bias": True},
    "llama": {"norm_offset": 0.0, "attn_gate": False, "qk_norm": False, "attention_bias": False},
    "mistral": {"norm_offset": 0.0, "attn_gate": False, "qk_norm": False, "attention_bias": False},
}


@dataclass(frozen=True)
class MtpConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rotary_dim: int
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    #: 1.0 for zero-centered RMSNorm (``x·(1+w)``), 0.0 for the plain kind (``x·w``).
    norm_offset: float = 0.0
    #: Sigmoid output gate per head, packed with Q in ``q_proj`` (``[q, gate]`` per head).
    attn_gate: bool = False
    qk_norm: bool = False
    attention_bias: bool = False

    @classmethod
    def from_hf_config(cls, cfg: dict) -> MtpConfig:
        """Reads a decoder-only HF ``config.json`` (``text_config`` if nested)."""
        t = cfg.get("text_config", cfg)
        model_type = t.get("model_type") or cfg.get("model_type", "")
        if model_type not in FAMILIES:
            raise ValueError(
                f"model_type {model_type!r} has no MTP-head family defaults; add it to FAMILIES "
                f"in modelbuilder_train/mtp/model.py (known: {', '.join(sorted(FAMILIES))})"
            )
        fam = FAMILIES[model_type]
        rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
        partial = t.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0))
        head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        return cls(
            hidden_size=t["hidden_size"],
            num_heads=t["num_attention_heads"],
            num_kv_heads=t.get("num_key_value_heads", t["num_attention_heads"]),
            head_dim=head_dim,
            intermediate_size=t["intermediate_size"],
            rotary_dim=int(head_dim * partial),
            rope_theta=float(t.get("rope_theta", rope.get("rope_theta", 10_000.0))),
            rms_eps=float(t.get("rms_norm_eps", 1e-6)),
            norm_offset=float(fam["norm_offset"]),
            attn_gate=bool(t.get("attn_output_gate", fam["attn_gate"])),
            qk_norm=bool(fam["qk_norm"]),
            attention_bias=bool(t.get("attention_bias", fam["attention_bias"])),
        )


class RMSNorm(nn.Module):
    """``x / rms(x) · (offset + w)``: offset 1 for zero-centered weights (initialized to 0), 0 otherwise."""

    def __init__(self, dim: int, eps: float, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset
        self.weight = nn.Parameter(torch.zeros(dim) if offset else torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (self.offset + self.weight.float())).to(dtype)


def apply_partial_rope(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """NeoX-style RoPE on the first ``rotary_dim`` dims of ``x`` [B, T, H, D]."""
    half = rotary_dim // 2
    inv_freq = theta ** (-torch.arange(0, half, dtype=torch.float32, device=x.device) * 2 / rotary_dim)
    ang = positions.float()[:, None] * inv_freq[None, :]  # [T, half]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    rot, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    x1, x2 = rot[..., :half], rot[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rotated.to(x.dtype), rest], dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.c = c
        q_out = (2 if c.attn_gate else 1) * c.num_heads * c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, q_out, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=c.attention_bias)
        self.o_proj = nn.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=False)
        if c.qk_norm:
            self.q_norm = RMSNorm(c.head_dim, c.rms_eps, c.norm_offset)
            self.k_norm = RMSNorm(c.head_dim, c.rms_eps, c.norm_offset)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        c = self.c
        b, t, _ = x.shape
        if c.attn_gate:
            # Per head: [q (head_dim), gate (head_dim)] — llama.cpp views stride 2·head_dim.
            qg = self.q_proj(x).view(b, t, c.num_heads, 2 * c.head_dim)
            q, gate = qg[..., : c.head_dim], qg[..., c.head_dim :]
        else:
            q, gate = self.q_proj(x).view(b, t, c.num_heads, c.head_dim), None
        k = self.k_proj(x).view(b, t, c.num_kv_heads, c.head_dim)
        if c.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        v = self.v_proj(x).view(b, t, c.num_kv_heads, c.head_dim)
        q = apply_partial_rope(q, positions, c.rotary_dim, c.rope_theta)
        k = apply_partial_rope(k, positions, c.rotary_dim, c.rope_theta)
        rep = c.num_heads // c.num_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=1.0 / math.sqrt(c.head_dim)
        ).transpose(1, 2)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out.reshape(b, t, c.num_heads * c.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.self_attn = GatedAttention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.mlp = SwiGLU(c)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class MtpHead(nn.Module):
    """The trainable MTP block. Parameter names match HF ``mtp.*`` keys."""

    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.config = c
        self.pre_fc_norm_embedding = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.pre_fc_norm_hidden = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.fc = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        self.layers = nn.ModuleList([DecoderLayer(c)])
        self.norm = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)

    def forward(self, trunk_hidden: torch.Tensor, next_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Returns the normed hidden state that the (frozen) LM head reads.

        trunk_hidden: [B, T, H], the trunk's post-final-norm hidden at t.
        next_embeds:  [B, T, H], token embeddings of x[t+1].
        positions:    [T], RoPE positions.
        """
        e = self.pre_fc_norm_embedding(next_embeds)
        h = self.pre_fc_norm_hidden(trunk_hidden)
        x = self.fc(torch.cat([e, h], dim=-1))
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


MTP_PREFIX = "mtp."


def load_mtp_state(paths: list[Path], dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Reads ``mtp.*`` tensors (prefix removed) from safetensors files."""
    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    for p in paths:
        with safe_open(str(p), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open handles have no __iter__
                for prefix in (MTP_PREFIX, "model." + MTP_PREFIX):
                    if key.startswith(prefix):
                        state[key[len(prefix) :]] = f.get_tensor(key).to(dtype)
    return state


def save_mtp_state(head: MtpHead, path: Path, dtype: torch.dtype = torch.bfloat16) -> None:
    """Writes the head as ``mtp.*`` tensors, the layout ``modelbuilder surgery mtp`` reads."""
    from safetensors.torch import save_file

    tensors = {MTP_PREFIX + k: v.detach().to(dtype).contiguous() for k, v in head.state_dict().items()}
    save_file(tensors, str(path), metadata={"format": "pt", "modelbuilder": "mtp-head"})

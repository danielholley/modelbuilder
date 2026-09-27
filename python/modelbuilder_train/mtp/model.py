"""Qwen3.5/3.8-style MTP head in PyTorch.

Mirrors ``graph_mtp`` in the PrismML llama.cpp fork (``src/models/qwen35.cpp``)
operation for operation, so a head trained here drafts the same way at
inference:

    e = enorm(embed(x[t+1]))            h = hnorm(trunk_hidden[t])
    x = eh_proj(concat(e, h))           # concat order: embedding first
    x = x + attn(input_layernorm(x))    # gated full attention, Q/K norms, partial RoPE
    x = x + ffn(post_attention_layernorm(x))   # SwiGLU
    logits = lm_head(norm(x))           # predicts x[t+2]

``trunk_hidden`` is the trunk's output *after* its final norm (llama.cpp's
``h_nextn``, the same tensor ``llama-embedding --pooling none`` returns).

Weights use Hugging Face names and conventions (``mtp.*`` without the prefix):
RMSNorm weights are zero-centered, i.e. applied as ``x / rms(x) * (1 + w)``.
The GGUF sidecar stores them with the +1 folded in (see ``mb-surgery``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MtpConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rotary_dim: int
    rope_theta: float = 10_000_000.0
    rms_eps: float = 1e-6

    @classmethod
    def from_hf_config(cls, cfg: dict) -> MtpConfig:
        """Reads a Qwen3.5-family ``config.json`` (``text_config`` if nested)."""
        t = cfg.get("text_config", cfg)
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
            rope_theta=float(t.get("rope_theta", rope.get("rope_theta", 10_000_000.0))),
            rms_eps=float(t.get("rms_norm_eps", 1e-6)),
        )


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


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
        self.q_proj = nn.Linear(c.hidden_size, 2 * c.num_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(c.head_dim, c.rms_eps)
        self.k_norm = ZeroCenteredRMSNorm(c.head_dim, c.rms_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        c = self.c
        b, t, _ = x.shape
        # Per head: [q (head_dim), gate (head_dim)] — llama.cpp views stride 2·head_dim.
        qg = self.q_proj(x).view(b, t, c.num_heads, 2 * c.head_dim)
        q, gate = qg[..., : c.head_dim], qg[..., c.head_dim :]
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(b, t, c.num_kv_heads, c.head_dim))
        v = self.v_proj(x).view(b, t, c.num_kv_heads, c.head_dim)
        q = apply_partial_rope(q, positions, c.rotary_dim, c.rope_theta)
        k = apply_partial_rope(k, positions, c.rotary_dim, c.rope_theta)
        rep = c.num_heads // c.num_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=1.0 / math.sqrt(c.head_dim)
        ).transpose(1, 2)
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
        self.input_layernorm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_eps)
        self.self_attn = GatedAttention(c)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class MtpHead(nn.Module):
    """The trainable MTP block. Parameter names match HF ``mtp.*`` keys."""

    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.config = c
        self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(c.hidden_size, c.rms_eps)
        self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(c.hidden_size, c.rms_eps)
        self.fc = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        self.layers = nn.ModuleList([DecoderLayer(c)])
        self.norm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_eps)

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

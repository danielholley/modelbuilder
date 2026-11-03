"""The MTP head of ``mtp/model.py`` in MLX, for Apple Silicon.

Same modules, same parameter names (``mtp.*`` without the prefix) and the same
math as the PyTorch head, so a head trained here loads into the PyTorch head,
``modelbuilder surgery mtp`` and llama.cpp unchanged. A test checks both give
the same output for the same weights.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from modelbuilder_train.mtp.model import MtpConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset
        self.eps = eps
        self.weight = mx.zeros((dim,)) if offset else mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * (self.offset + self.weight.astype(mx.float32))).astype(dtype)


def apply_partial_rope(x: mx.array, positions: mx.array, rotary_dim: int, theta: float) -> mx.array:
    """NeoX-style RoPE on the first ``rotary_dim`` dims of ``x`` [B, T, H, D]."""
    half = rotary_dim // 2
    inv_freq = theta ** (-mx.arange(0, half, dtype=mx.float32) * 2 / rotary_dim)
    ang = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    cos, sin = mx.cos(ang)[None, :, None, :], mx.sin(ang)[None, :, None, :]
    rot, rest = x[..., :rotary_dim].astype(mx.float32), x[..., rotary_dim:]
    x1, x2 = rot[..., :half], rot[..., half:]
    rotated = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return mx.concatenate([rotated.astype(x.dtype), rest], axis=-1)


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

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        c = self.c
        b, t, _ = x.shape
        if c.attn_gate:
            qg = self.q_proj(x).reshape(b, t, c.num_heads, 2 * c.head_dim)
            q, gate = qg[..., : c.head_dim], qg[..., c.head_dim :]
        else:
            q, gate = self.q_proj(x).reshape(b, t, c.num_heads, c.head_dim), None
        k = self.k_proj(x).reshape(b, t, c.num_kv_heads, c.head_dim)
        if c.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        v = self.v_proj(x).reshape(b, t, c.num_kv_heads, c.head_dim)
        q = apply_partial_rope(q, positions, c.rotary_dim, c.rope_theta)
        k = apply_partial_rope(k, positions, c.rotary_dim, c.rope_theta)
        rep = c.num_heads // c.num_kv_heads
        k = mx.repeat(k, rep, axis=2)
        v = mx.repeat(v, rep, axis=2)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(c.head_dim)
        causal = mx.triu(mx.full((t, t), -mx.inf), k=1)
        out = (mx.softmax((scores + causal).astype(mx.float32), axis=-1).astype(v.dtype) @ v).transpose(0, 2, 1, 3)
        if gate is not None:
            out = out * mx.sigmoid(gate)
        return self.o_proj(out.reshape(b, t, c.num_heads * c.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.self_attn = GatedAttention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.mlp = SwiGLU(c)

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class MtpHead(nn.Module):
    def __init__(self, c: MtpConfig) -> None:
        super().__init__()
        self.config = c
        self.pre_fc_norm_embedding = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.pre_fc_norm_hidden = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)
        self.fc = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        self.layers = [DecoderLayer(c)]
        self.norm = RMSNorm(c.hidden_size, c.rms_eps, c.norm_offset)

    def __call__(self, trunk_hidden: mx.array, next_embeds: mx.array, positions: mx.array) -> mx.array:
        e = self.pre_fc_norm_embedding(next_embeds)
        h = self.pre_fc_norm_hidden(trunk_hidden)
        x = self.fc(mx.concatenate([e, h], axis=-1))
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)

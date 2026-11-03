"""Multi-head latent attention (MLA) as a student path on any transformers model.

``modelbuilder surgery mla`` writes the latent projections (``mla.safetensors``)
next to the original weights and records the conversion under
``modelbuilder_mla`` in ``config.json`` (MHA2MLA, arXiv 2502.14837: partial RoPE
plus a joint SVD of the RoPE-free key and the value projections). This module
switches a loaded model between the two:

- **Teacher:** the original ``k_proj`` / ``v_proj`` and full RoPE.
- **Student:** ``k_proj`` and ``v_proj`` outputs are rebuilt from the latent,
  ``c = kv_a(x)`` and ``[k_nope; v] = kv_b(c)``, while the kept RoPE key
  dimensions come from ``k_rope(x)``. Rotary frequencies that were not kept
  are switched off (``cos = 1``, ``sin = 0``) for Q and K alike.

The swap is done with forward hooks on the existing modules, so Q/K norms,
gating and the attention kernel run unchanged and parameter names stay those
of the original checkpoint. The new parameters live at
``…self_attn.mla.{kv_a,kv_b,k_rope}``, the names they have in
``mla.safetensors``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

CONFIG_KEY = "modelbuilder_mla"


class _Mode(Protocol):
    student: bool


def load_config(model_dir: Path) -> dict | None:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return cfg.get("text_config", cfg).get(CONFIG_KEY) or cfg.get(CONFIG_KEY)


class Latent(nn.Module):
    def __init__(self, hidden: int, rank: int, kv_rows: int, rope_rows: int, bias: bool) -> None:
        super().__init__()
        self.kv_a = nn.Linear(hidden, rank, bias=False)
        self.kv_b = nn.Linear(rank, kv_rows, bias=bias)
        self.k_rope = nn.Linear(hidden, rope_rows, bias=bias)


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    layers = getattr(model.get_decoder(), "layers", None)
    if layers is None:
        raise ValueError("the model's decoder has no `layers`")
    return layers


def _rotary(model: nn.Module) -> nn.Module:
    found = [m for _, m in model.named_modules() if type(m).__name__.endswith("RotaryEmbedding")]
    if len(found) != 1:
        raise ValueError(f"expected one rotary embedding module, found {len(found)}")
    return found[0]


def attach(model: nn.Module, model_dir: Path, cfg: dict, mode: _Mode) -> list[str]:
    """Adds the latent modules (from ``mla.safetensors``) and the student hooks; returns the new parameter names."""
    from safetensors.torch import load_file

    n_kv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
    rope_dims, nope_dims = cfg["rope_dims"], cfg["nope_dims"]
    rot, freqs = cfg["rotary_dim"], set(cfg["rope_freqs"])
    knope_rows = n_kv * len(nope_dims)
    # Position of each original K row in cat([k_nope, k_rope]).
    perm = torch.empty(n_kv * hd, dtype=torch.long)
    for h in range(n_kv):
        for j, d in enumerate(nope_dims):
            perm[h * hd + d] = h * len(nope_dims) + j
        for j, d in enumerate(rope_dims):
            perm[h * hd + d] = knope_rows + h * len(rope_dims) + j

    stored = load_file(str(Path(model_dir) / cfg.get("tensors", "mla.safetensors")))
    layers = _decoder_layers(model)
    names: list[str] = []
    module_names = {m: n for n, m in model.named_modules()}
    for idx in cfg["layers"]:
        attn = layers[idx].self_attn
        k_proj, v_proj = attn.k_proj, attn.v_proj
        ref = k_proj.weight
        key = next((k for k in stored if k.endswith(f"layers.{idx}.self_attn.mla.kv_a.weight")), None)
        if key is None:
            raise ValueError(f"{cfg.get('tensors')} has no latent for layer {idx}")
        base = key[: -len("kv_a.weight")]
        part = {k[len(base) :]: v for k, v in stored.items() if k.startswith(base)}
        lat = Latent(
            ref.shape[1],
            cfg["kv_rank"],
            part["kv_b.weight"].shape[0],
            part["k_rope.weight"].shape[0],
            bias="kv_b.bias" in part,
        )
        lat.load_state_dict(part)
        attn.mla = lat.to(device=ref.device, dtype=ref.dtype)
        names += [f"{module_names[attn]}.mla.{k}" for k in part]
        p = perm.to(ref.device)

        def k_hook(_m, inputs, out, lat=lat, attn=attn, p=p):
            if not mode.student:
                return out
            x = inputs[0]
            kv = attn.mla.kv_b(attn.mla.kv_a(x))
            k = torch.cat([kv[..., :knope_rows], attn.mla.k_rope(x)], -1)
            return k[..., p]

        def v_hook(_m, inputs, out, attn=attn):
            if not mode.student:
                return out
            return attn.mla.kv_b(attn.mla.kv_a(inputs[0]))[..., knope_rows:]

        k_proj.register_forward_hook(k_hook)
        v_proj.register_forward_hook(v_hook)

    drop = torch.tensor([i for i in range(rot // 2) if i not in freqs], dtype=torch.long)
    drop = torch.cat([drop, drop + rot // 2])

    def rope_hook(_m, _inputs, out):
        if not mode.student or drop.numel() == 0:
            return out
        cos, sin = out
        if cos.shape[-1] != rot:
            raise ValueError(f"rotary embedding has {cos.shape[-1]} dims, the conversion expects {rot}")
        d = drop.to(cos.device)
        cos, sin = cos.clone(), sin.clone()
        cos[..., d] = 1.0
        sin[..., d] = 0.0
        return cos, sin

    _rotary(model).register_forward_hook(rope_hook)
    return names

"""Fake quantization for QAT: the values a format can hold, with straight-through gradients.

Every function mirrors a deterministic reference quantizer, so a tensor trained
through it re-encodes to the same values when ``modelbuilder surgery replace``
writes it back (``mb-formats/src/quant.rs``, ggml's ``quantize_row_*_ref``):

- KV-cache formats (blocks along the head dimension, as llama.cpp stores the
  cache): ``q8_0``, ``q4_0``, and ``nvfp4`` (E2M1 values, one E4M3 scale per 16:
  the paper format of DeepSeek-V4.1-Flash's FP4 KV cache, no runtime support
  in llama.cpp yet).
- Weight formats (blocks along the input dimension): ``Q8_0``, and the ternary
  ``PQ2_0``/``PTQ1_0`` (absmax scale per 128). When the source folded an
  orthogonal Hadamard rotation into a weight (PrismML), quantization happens
  in the rotated basis: ``w_stored = H(s ⊙ w)``, quantize, then ``w = s ⊙ H(q)``.
"""

from __future__ import annotations

import torch


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Forward: q. Backward: identity (straight-through)."""
    return x + (q - x).detach()


def _f16(d: torch.Tensor) -> torch.Tensor:
    return d.to(torch.float16).to(torch.float32)


def _blocks(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] % size:
        raise ValueError(f"last dimension {x.shape[-1]} is not a multiple of the block size {size}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // size, size)


def q8_0(x: torch.Tensor) -> torch.Tensor:
    b = _blocks(x.float(), 32)
    d = _f16(b.abs().amax(-1, keepdim=True) / 127.0)
    id_ = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    return (torch.round(b * id_).clamp(-127, 127) * d).reshape(x.shape).to(x.dtype)


def q4_0(x: torch.Tensor) -> torch.Tensor:
    b = _blocks(x.float(), 32)
    idx = b.abs().argmax(-1, keepdim=True)
    mx = torch.gather(b, -1, idx)  # the signed value of largest magnitude
    d = _f16(mx / -8.0)
    id_ = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.trunc(b * id_ + 8.5), max=15)  # C's (int8_t) cast truncates toward zero
    return ((q - 8.0) * d).reshape(x.shape).to(x.dtype)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "float8_e4m3fn"):
        return x.clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
    return x  # older torch: keep the scale in fp32 (slightly optimistic)


def nvfp4(x: torch.Tensor) -> torch.Tensor:
    """E2M1 values with one E4M3 scale per 16 (no second-level global scale)."""
    b = _blocks(x.float(), 16)
    scale = _e4m3(b.abs().amax(-1, keepdim=True) / 6.0)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    grid = _E2M1.to(x.device)
    mag = (b.abs() / safe).unsqueeze(-1)
    nearest = grid[(mag - grid).abs().argmin(-1)]
    return (
        (torch.sign(b) * nearest * torch.where(scale > 0, scale, torch.zeros_like(scale))).reshape(x.shape).to(x.dtype)
    )


KV_FORMATS = {"q8_0": q8_0, "q4_0": q4_0, "nvfp4": nvfp4}


def kv(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Fake-quantizes cache entries ([..., head_dim]) with straight-through gradients."""
    if fmt in ("f16", "fp16"):
        return ste(x, x.to(torch.float16).to(x.dtype))
    if fmt not in KV_FORMATS:
        raise ValueError(f"unknown KV cache format {fmt!r} (have: f16, {', '.join(KV_FORMATS)})")
    return ste(x, KV_FORMATS[fmt](x))


def ternary(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """PQ2_0 / PTQ1_0: absmax scale (fp16) per block, codes in {-1, 0, +1}."""
    b = _blocks(x.float(), block)
    d = _f16(b.abs().amax(-1, keepdim=True))
    id_ = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    return (torch.round(b * id_).clamp(-1, 1) * d).reshape(x.shape).to(x.dtype)


WEIGHT_FORMATS = {"Q8_0": q8_0, "PQ2_0": ternary, "PTQ1_0": ternary}


def fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """Normalized Sylvester-Walsh-Hadamard transform on each ``block`` of the last dim (an involution)."""
    shape = x.shape
    y = x.reshape(-1, shape[-1] // block, block)
    h = 1
    while h < block:
        y = y.reshape(*y.shape[:-1], block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*y.shape[:-3], block)
        h *= 2
    return (y / block**0.5).reshape(shape)


class WeightFormat:
    """How one weight is stored in the source checkpoint (from ``modelbuilder_quantization.json``)."""

    def __init__(self, dtype: str, rotated: bool, block_size: int | None, signs: torch.Tensor | None) -> None:
        self.dtype = dtype
        self.rotated = rotated
        self.block_size = block_size
        self.signs = signs

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        """The weight as the source format can hold it, with straight-through gradients."""
        if self.dtype in ("F32",):
            return w
        if self.dtype in ("F16", "BF16"):
            dt = torch.float16 if self.dtype == "F16" else torch.bfloat16
            return ste(w, w.to(dt).to(w.dtype))
        if self.dtype not in WEIGHT_FORMATS:
            raise ValueError(f"no fake-quantizer for source type {self.dtype}")
        q = WEIGHT_FORMATS[self.dtype]
        if not self.rotated:
            return ste(w, q(w))
        s = self.signs.to(w.device, torch.float32) if self.signs is not None else 1.0
        stored = fwht(w.float() * s, self.block_size)  # w_stored = H(s ⊙ w)
        back = fwht(q(stored), self.block_size) * s  # w = s ⊙ H(q)
        return ste(w, back.to(w.dtype))


def load_manifest(path) -> dict[str, WeightFormat]:
    """Reads ``modelbuilder_quantization.json`` (written by ``modelbuilder export-hf``)."""
    import json
    from pathlib import Path

    m = json.loads(Path(path).read_text())
    if m.get("format") != "modelbuilder.source-quantization":
        raise ValueError(f"{path}: not a modelbuilder source-quantization manifest")
    rot = m.get("rotation") or {}
    signs = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in (rot.get("signs") or {}).items()}
    out = {}
    for name, t in m["tensors"].items():
        out[name] = WeightFormat(t["dtype"], t["rotated"], rot.get("block_size"), None)
        out[name]._width_signs = signs  # resolved per tensor width on first use
        # The rotation acts on the GGUF column order, which the manifest doesn't carry.
        out[name]._unsupported = t["rotated"] and t["cols_reordered"]
    return out


def resolve_signs(fmt: WeightFormat, width: int, name: str = "") -> WeightFormat:
    if getattr(fmt, "_unsupported", False):
        raise ValueError(f"{name}: rotated and column-reordered in the source; QAT in that basis is not supported")
    signs = getattr(fmt, "_width_signs", {})
    if fmt.rotated and fmt.signs is None and width in signs:
        fmt.signs = signs[width]
    return fmt

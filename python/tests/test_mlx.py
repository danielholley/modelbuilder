"""The mlx backend: same head math as PyTorch, and an mtp_align run that learns."""

import json

import pytest
import torch
from conftest import job
from test_mtp import run_cli

from modelbuilder_train.mtp.model import MtpHead, load_mtp_state, save_mtp_state

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("gate", [True, False])
def test_mlx_head_matches_the_torch_head(tiny, tmp_path, gate):
    from mlx.utils import tree_unflatten

    from modelbuilder_train.backends.mlx_backend import load_head_state
    from modelbuilder_train.mtp.mlx_model import MtpHead as MlxHead

    cfg = tiny["cfg"]
    if not gate:  # the plain (non-gated, plain-norm) family too
        from dataclasses import replace

        cfg = replace(cfg, attn_gate=False, norm_offset=0.0, qk_norm=False)
    torch.manual_seed(1)
    th = MtpHead(cfg).eval()
    with torch.no_grad():
        for p in th.parameters():
            p.add_(0.05 * torch.randn_like(p))
    save_mtp_state(th, tmp_path / "h.safetensors", dtype=torch.float32)
    mh = MlxHead(cfg)
    mh.update(tree_unflatten(list(load_head_state([tmp_path / "h.safetensors"]).items())))
    h, e = torch.randn(1, 9, cfg.hidden_size), torch.randn(1, 9, cfg.hidden_size)
    pos = torch.arange(3, 12)
    with torch.no_grad():
        want = th(h, e, pos).numpy()
    got = mh(mx.array(h.numpy()), mx.array(e.numpy()), mx.array(pos.numpy()))
    assert abs(got.__array__() - want).max() < 1e-4


def test_mlx_backend_learns_and_writes_a_torch_loadable_head(tiny):
    spec = job(tiny, steps=60)
    spec["backend"] = "mlx"
    code, events, err = run_cli(tiny, spec)
    assert code == 0, err
    assert events[0]["device"].startswith("mlx:")
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"] * 0.5, (evals[0], evals[-1])
    out = tiny["dir"] / events[-1]["outputs"]["mtp_head"]
    state = load_mtp_state([out / "model.safetensors"])
    MtpHead(tiny["cfg"]).load_state_dict(state, strict=True)
    assert json.loads((out / "config.json").read_text())


def test_mlx_backend_refuses_other_stages(tiny):
    spec = job(tiny)
    spec["backend"] = "mlx"
    st = spec["stages"][0]
    st["kind"], st["draft_align"] = (
        "draft_align",
        {
            "kind": "medusa",
            "frozen_tensors": "frozen.safetensors",
            "embedding_tensor": "token_embd.weight",
            "lm_head_tensor": "output.weight",
            "features": "features",
        },
    )
    del st["mtp_align"]
    code, events, _ = run_cli(tiny, spec)
    assert code != 0
    assert "mlx backend" in events[-1].get("message", "") or any("mlx backend" in e.get("message", "") for e in events)

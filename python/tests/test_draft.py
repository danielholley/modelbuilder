"""The ``draft_align`` stage: Medusa and EAGLE drafters learn from trunk features."""

import json

import pytest
import torch
from conftest import job
from test_e2e import BIN, mb
from test_mtp import run_cli

from modelbuilder_train.draft.model import Eagle, Medusa
from modelbuilder_train.mtp.model import MtpConfig


def draft_job(tiny: dict, kind: str, steps: int = 60, **extra) -> dict:
    spec = job(tiny, steps=steps)
    st = spec["stages"][0]
    m = st.pop("mtp_align")
    st["name"], st["kind"] = f"{kind}-align", "draft_align"
    st["draft_align"] = {
        "kind": kind,
        "reference_config": m["reference_config"] if kind == "eagle" else None,
        "frozen_tensors": m["frozen_tensors"],
        "embedding_tensor": m["embedding_tensor"],
        "lm_head_tensor": m["lm_head_tensor"],
        "features": m["features"],
        "eval_fraction": 0.2,
    } | extra
    st["hyper"]["lr"] = 1e-2
    return spec


def test_medusa_heads_start_as_the_target_head():
    lm = torch.randn(10, 6)
    m = Medusa(6, 3, lm)
    h = torch.randn(2, 5, 6)
    for k, out in enumerate(m(h)):
        assert torch.allclose(out, h)  # zero-initialized residual blocks
        assert torch.equal(m.head_weight(k, lm), lm)
    assert set(m.state_dict()) == {f"blocks.{k}.layers.0.{p}" for k in range(3) for p in ("weight", "bias")} | {
        f"lm_heads.{k}.weight" for k in range(3)
    }
    shared = Medusa(6, 2, None)
    assert shared.head_weight(1, lm) is lm and not any("lm_heads" in k for k in shared.state_dict())


def test_eagle_shape_and_causality(tiny):
    cfg: MtpConfig = tiny["cfg"]
    e = Eagle(cfg, 2).eval()
    assert isinstance(e.layers[0].input_layernorm, torch.nn.Identity)
    h, x = torch.randn(1, 7, cfg.hidden_size), torch.randn(1, 7, cfg.hidden_size)
    out = e(h, x, torch.arange(7))
    assert out.shape == h.shape
    h2 = h.clone()
    h2[0, 5:] += 1.0
    out2 = e(h2, x, torch.arange(7))
    assert torch.allclose(out[0, :5], out2[0, :5], atol=1e-5)


@pytest.mark.parametrize(
    ("kind", "extra"), [("medusa", {"heads": 3}), ("medusa", {"heads": 2, "share_lm_head": True}), ("eagle", {})]
)
def test_drafters_learn(tiny, kind, extra):
    code, events, err = run_cli(tiny, draft_job(tiny, kind, **extra))
    assert code == 0, err
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"] * 0.7, (evals[0], evals[-1])
    assert evals[-1]["accuracy"] > evals[0]["accuracy"]
    out = tiny["dir"] / events[-1]["outputs"]["draft_head"]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["kind"] == kind and cfg["vocab_size"] == 50
    from safetensors.torch import load_file

    state = {k: v.float() for k, v in load_file(str(out / "model.safetensors")).items()}
    if kind == "medusa":
        lm = None if cfg["share_lm_head"] else torch.zeros(50, 32)
        Medusa(cfg["hidden_size"], cfg["num_heads"], lm).load_state_dict(state)
    else:
        Eagle(tiny["cfg"], cfg["num_hidden_layers"]).load_state_dict(state)


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_job_draft_align_emits_a_valid_spec(tiny, tmp_path):
    target = mb("fixture", "gguf-hybrid-ternary", str(tmp_path / "t")).strip()
    ref = tmp_path / "r"
    mb("fixture", "hybrid-mtp-reference", str(ref))
    run = tmp_path / "run"
    out = mb("job", "draft-align", target, "--kind", "eagle", "--features", str(tiny["dir"] / "features"),
             "-o", str(run), "--emit-only", check=False)  # fmt: skip
    assert "--reference" in out  # EAGLE needs the layer shape
    mb("job", "draft-align", target, "--kind", "eagle", "--features", str(tiny["dir"] / "features"),
       "--reference", str(ref), "-o", str(run), "--emit-only", "--steps", "5")  # fmt: skip
    spec = json.loads((run / "job.json").read_text())
    d = spec["stages"][0]["draft_align"]
    assert spec["stages"][0]["kind"] == "draft_align" and d["kind"] == "eagle"
    assert (run / "frozen.safetensors").exists()
    from modelbuilder_train.spec import JobSpec

    JobSpec.model_validate(spec)

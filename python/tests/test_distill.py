"""QAT building blocks, and ``modelbuilder job trunk-distill`` end to end on a fixture."""

import json
import sys
from pathlib import Path

import pytest
import torch
from test_e2e import BIN, mb

from modelbuilder_train.hf import attention, fakequant


def test_fake_quantizers_are_idempotent_and_straight_through():
    torch.manual_seed(0)
    x = torch.randn(4, 128, requires_grad=True)
    for f in (fakequant.q8_0, fakequant.q4_0, fakequant.nvfp4, fakequant.ternary):
        q = f(x.detach())
        assert torch.allclose(f(q), q, atol=1e-6), f.__name__
    y = fakequant.kv(x, "q4_0")
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_q4_0_matches_ggml_reference():
    # quantize_row_q4_0_ref on one block: d = max / -8, q = min(15, (int8)(x/d + 8.5)).
    x = torch.linspace(-1.0, 0.75, 32)
    d = torch.tensor(-1.0 / -8.0).to(torch.float16).float()
    want = (torch.clamp(torch.trunc(x / d + 8.5), max=15) - 8) * d
    assert torch.equal(fakequant.q4_0(x), want)


def test_rotated_weight_format_quantizes_in_the_stored_basis():
    torch.manual_seed(1)
    signs = torch.where(torch.rand(128) < 0.5, -1.0, 1.0)
    fmt = fakequant.WeightFormat("PQ2_0", True, 128, signs)
    w = torch.randn(8, 128)
    q = fmt(w)
    stored = fakequant.fwht(q * signs, 128)
    # In the stored basis every block is {-d, 0, +d}.
    for row in stored:
        assert len(torch.unique(row.abs().round(decimals=4))) <= 2
    assert torch.allclose(fmt(q), q, atol=1e-5)
    assert torch.allclose(fakequant.fwht(fakequant.fwht(w, 128), 128), w, atol=1e-5)


def test_share_groups():
    assert attention.share_groups([3, 7, 11, 15, 19], 2) == {3: 3, 7: 3, 11: 11, 15: 11, 19: 19}


def test_cache_quant_blocks_run_along_a_token_row():
    x = torch.randn(1, 2, 5, 16)  # 2 kv heads x 16 = one block of 32 per token
    got = attention._cache_quant(x, "q8_0")
    rows = x.transpose(1, 2).reshape(1, 5, 32)
    assert torch.allclose(got.transpose(1, 2).reshape(1, 5, 32), fakequant.q8_0(rows))


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_trunk_distill_job_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    target = Path(mb("fixture", "gguf-llama", str(tmp_path / "t")).strip())
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(48)]}) + "\n" for i in range(8))
    )
    run = tmp_path / "run"
    out = mb(
        "job", "trunk-distill", str(target), "--reference", str(target.parent), "--texts", str(texts),
        "-o", str(run), "--kv-format", "q4_0", "--kv-share-group", "2", "--steps", "6", "--seq-len", "32",
        "--batch-seqs", "2", "--lr", "1e-3", "--eval-every", "3", "--log-every", "3", "--device", "cpu",
        "--fp32", "--max-rel-error", "1e-4", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert events[-1]["event"] == "finished" and events[-1]["status"] == "ok"
    spec = json.loads((run / "job.json").read_text())
    assert spec["stages"][0]["trunk_distill"]["kv_share_group"] == 2
    new = run / f"{target.stem}-kv-quant-share-qat.gguf"
    assert new.exists()
    # Trained through the source format, so the write-back is exact.
    assert "re-quantization error 0.000% RMS" in out
    info = json.loads(mb("inspect", str(new), "--json"))
    assert info  # the new checkpoint opens

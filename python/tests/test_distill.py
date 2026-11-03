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


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_heal_a_pruned_model_against_the_original(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    target = Path(mb("fixture", "gguf-llama", str(tmp_path / "t")).strip())
    pruned = tmp_path / "pruned.gguf"
    mb("surgery", "prune", str(target), "--layers", "1..2", "-o", str(pruned))
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(48)]}) + "\n" for i in range(8))
    )
    common = ["--reference", str(target.parent), "--texts", str(texts), "--steps", "10", "--seq-len", "32",
              "--batch-seqs", "2", "--lr", "3e-3", "--eval-every", "10", "--log-every", "5", "--device", "cpu",
              "--fp32", "--raw-events", "--python", sys.executable, "--trainable", "mlp."]  # fmt: skip

    def evals(out: str) -> list[dict]:
        events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        assert events[-1]["event"] == "finished" and events[-1]["status"] == "ok"
        return [e for e in events if e["event"] == "eval"]

    run = tmp_path / "heal"
    out = mb("job", "trunk-distill", str(pruned), "--teacher", str(target), "-o", str(run), *common)
    e = evals(out)
    assert e[-1]["loss"] < e[0]["loss"], e
    assert json.loads((run / "job.json").read_text())["stages"][0]["trunk_distill"]["teacher"].endswith("teacher")
    assert (run / "pruned-heal.gguf").exists()

    # Cross-entropy on the texts alone (e.g. a long-context finetune).
    run = tmp_path / "ce"
    out = mb("job", "trunk-distill", str(target), "--loss", "ce", "-o", str(run), *common)
    e = evals(out)
    assert e[-1]["loss"] < e[0]["loss"], e
    assert json.loads((run / "job.json").read_text())["stages"][0]["trunk_distill"]["loss"] == "ce"
    bad = mb("job", "trunk-distill", str(target), "--loss", "ce", "--teacher", str(target), "-o", str(run), *common,
             check=False)  # fmt: skip
    assert "drop --teacher" in bad


def test_lora_switch_starts_at_the_original_and_trains_only_its_factors():
    from torch.nn.utils import parametrize

    from modelbuilder_train.hf.distill import Mode, Switch

    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8, bias=False)
    w0 = lin.weight.detach().clone()
    mode = Mode()
    parametrize.register_parametrization(lin, "weight", Switch(lin.weight, mode, None, lora_rank=2, lora_alpha=4.0))
    sw = lin.parametrizations.weight[0]
    assert sw.lora and sw.scale == 2.0 and not hasattr(sw, "original")
    assert torch.equal(lin.weight, w0)  # B = 0
    with torch.no_grad():
        sw.lora_b.fill_(0.1)
    assert not torch.allclose(lin.weight, w0)
    mode.student = False
    assert torch.equal(lin.weight, w0)  # the teacher sees the frozen base


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_lora_trunk_distill_writes_back_exactly(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    target = Path(mb("fixture", "gguf-llama", str(tmp_path / "t")).strip())
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(48)]}) + "\n" for i in range(8))
    )
    # Heal a pruned model (a real gap to close) with LoRA on the K/V projections.
    pruned = tmp_path / "pruned.gguf"
    mb("surgery", "prune", str(target), "--layers", "1..2", "-o", str(pruned))
    run = tmp_path / "run"
    out = mb(
        "job", "trunk-distill", str(pruned), "--reference", str(target.parent), "--texts", str(texts),
        "--teacher", str(target), "-o", str(run), "--kv-format", "q4_0", "--lora-rank", "4", "--steps", "10",
        "--seq-len", "32", "--batch-seqs", "2", "--lr", "1e-2", "--eval-every", "10", "--log-every", "5",
        "--device", "cpu", "--fp32", "--max-rel-error", "1e-4", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert events[-1]["event"] == "finished" and events[-1]["status"] == "ok"
    started = next(e for e in events if e["event"] == "started")
    # LoRA factors only: 1 layer × (K, V) × 4 × (64 + 32).
    assert started["trainable_params"] == 2 * 4 * (64 + 32)
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"], evals
    assert json.loads((run / "job.json").read_text())["stages"][0]["trunk_distill"]["lora_rank"] == 4
    assert "re-quantization error 0.000% RMS" in out


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_device_map_path_with_a_teacher(tmp_path: Path) -> None:
    """``device_map`` loads through accelerate; on a CPU-only machine everything lands on the CPU."""
    pytest.importorskip("accelerate")
    target = Path(mb("fixture", "gguf-llama", str(tmp_path / "t")).strip())
    pruned = tmp_path / "pruned.gguf"
    mb("surgery", "prune", str(target), "--layers", "1..2", "-o", str(pruned))
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(48)]}) + "\n" for i in range(8))
    )
    run = tmp_path / "run"
    out = mb(
        "job", "trunk-distill", str(pruned), "--reference", str(target.parent), "--texts", str(texts),
        "--teacher", str(target), "--device-map", "auto", "--trainable", "mlp.", "-o", str(run), "--steps", "6",
        "--seq-len", "32", "--batch-seqs", "2", "--lr", "3e-3", "--eval-every", "6", "--log-every", "3",
        "--fp32", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert events[-1]["event"] == "finished" and events[-1]["status"] == "ok"
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"], evals
    assert json.loads((run / "job.json").read_text())["stages"][0]["trunk_distill"]["device_map"] == "auto"
    bad = mb("job", "trunk-distill", str(pruned), "--reference", str(target.parent), "--texts", str(texts),
             "--device-map", "everywhere", "-o", str(tmp_path / "bad"), "--emit-only", check=False)  # fmt: skip
    assert "device_map" in bad


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_fsdp_over_two_ranks(tmp_path: Path) -> None:
    """Layers sharded over two CPU ranks (gloo): one event stream, it learns, and whole tensors write back."""
    pytest.importorskip("transformers")
    target = Path(mb("fixture", "gguf-llama", str(tmp_path / "t")).strip())
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(48)]}) + "\n" for i in range(8))
    )
    run = tmp_path / "run"
    out = mb(
        "job", "trunk-distill", str(target), "--reference", str(target.parent), "--texts", str(texts),
        "-o", str(run), "--loss", "ce", "--trainable", "mlp.", "--no-weight-fakequant", "--shard", "fsdp",
        "--nproc-per-node", "2", "--steps", "8", "--seq-len", "32", "--batch-seqs", "2", "--lr", "3e-3",
        "--eval-every", "8", "--log-every", "4", "--device", "cpu", "--fp32", "--max-rel-error", "0.05",
        "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    kinds = [e["event"] for e in events]
    assert kinds.count("started") == 1 and kinds.count("finished") == 1, kinds
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"], evals
    assert json.loads((run / "job.json").read_text())["stages"][0]["trunk_distill"]["shard"] == "fsdp"
    from safetensors import safe_open

    with safe_open(str(run / "updates" / "model.safetensors"), "pt") as f:
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}  # noqa: SIM118
    assert shapes["model.layers.0.mlp.gate_proj.weight"] == (128, 64)  # whole, not a shard
    assert (run / "t-finetune.gguf").exists() or (run / "tiny-llama-finetune.gguf").exists()
    bad = mb("job", "trunk-distill", str(target), "--reference", str(target.parent), "--texts", str(texts),
             "-o", str(tmp_path / "b"), "--shard", "fsdp", "--emit-only", check=False)  # fmt: skip
    assert "fsdp" in bad

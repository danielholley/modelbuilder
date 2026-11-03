"""``surgery upcycle`` checked in a real MoE implementation: transformers' Mixtral and Qwen3-MoE."""

import json
import sys
from pathlib import Path

import pytest
import torch
from test_e2e import BIN, mb

pytestmark = pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")


def _logits(hf: Path) -> tuple[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    with torch.no_grad():
        return type(m).__name__, m(torch.arange(5, 45)[None] % 96).logits


@pytest.mark.parametrize(
    ("fixture", "cls"), [("gguf-llama", "MixtralForCausalLM"), ("gguf-qwen3", "Qwen3MoeForCausalLM")]
)
def test_upcycled_model_computes_the_dense_function(tmp_path: Path, fixture: str, cls: str) -> None:
    pytest.importorskip("transformers")
    dense = Path(mb("fixture", fixture, str(tmp_path / "d")).strip())
    moe = tmp_path / "moe.gguf"
    mb("surgery", "upcycle", str(dense), "--experts", "4", "--top-k", "2", "--router-std", "0.5", "-o", str(moe))
    for src, out in [(dense, "dense-hf"), (moe, "moe-hf")]:
        mb("export-hf", str(src), "--reference", str(dense.parent), "--dtype", "f32", "-o", str(tmp_path / out))
    cfg = json.loads((tmp_path / "moe-hf" / "config.json").read_text())
    assert cfg["num_experts_per_tok"] == 2
    name_d, d = _logits(tmp_path / "dense-hf")
    name_m, m = _logits(tmp_path / "moe-hf")
    assert name_m == cls and name_d != cls
    # Identical experts and renormalized top-k weights: the router can't change the output.
    assert torch.allclose(d, m, atol=1e-5), (d - m).abs().max()


def test_trunk_distill_writes_back_into_an_upcycled_model(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    dense = Path(mb("fixture", "gguf-qwen3", str(tmp_path / "d")).strip())
    moe = tmp_path / "moe.gguf"
    mb("surgery", "upcycle", str(dense), "--experts", "2", "-o", str(moe))
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(40)]}) + "\n" for i in range(6))
    )
    run = tmp_path / "run"
    out = mb(
        "job", "trunk-distill", str(moe), "--reference", str(dense.parent), "--texts", str(texts), "-o", str(run),
        "--kv-format", "q4_0", "--steps", "3", "--seq-len", "32", "--batch-seqs", "1", "--device", "cpu", "--fp32",
        "--max-rel-error", "1e-4", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    assert "re-quantization error 0.000% RMS" in out
    info = json.loads(mb("inspect", str(run / "moe-kv-quant-qat.gguf"), "--json"))
    assert info

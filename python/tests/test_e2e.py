"""End to end with the Rust binary: fixture target → `modelbuilder job mtp-align` → GGUF sidecar.

Needs a built ``modelbuilder`` binary: ``$MODELBUILDER_BIN``, or ``target/debug/modelbuilder``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from modelbuilder_train.features import FeatureWriter

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("MODELBUILDER_BIN", ROOT / "target" / "debug" / "modelbuilder"))

pytestmark = pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")


def mb(*args: str) -> str:
    r = subprocess.run([str(BIN), *args], capture_output=True, text=True)
    assert r.returncode == 0, f"modelbuilder {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_mtp_align_job_end_to_end(tmp_path: Path) -> None:
    target = Path(mb("fixture", "gguf-bonsai-like", str(tmp_path / "t")).strip())
    ref = tmp_path / "r"
    mb("fixture", "qwen-hybrid-matching-bonsai-like", str(ref))

    # Synthetic trunk features: the target's own (primal) embeddings of a learnable sequence.
    mb("export-tensors", str(target), "--names", "token_embd.weight", "--dtype", "f32", "-o", str(tmp_path / "e.st"))
    from safetensors.torch import load_file

    emb = load_file(str(tmp_path / "e.st"))["token_embd.weight"]
    torch.manual_seed(0)
    w = FeatureWriter(tmp_path / "feat", emb.shape[1], {"runtime": "synthetic"})
    for s in range(16):
        toks = [(7 * s + 3 * i) % emb.shape[0] for i in range(64)]
        w.add(toks, emb[torch.tensor(toks)] + 0.05 * torch.randn(64, emb.shape[1]))
    w.close()

    run = tmp_path / "run"
    out = mb(
        "job", "mtp-align", str(target), "--from", str(ref), "--features", str(tmp_path / "feat"),
        "-o", str(run), "--steps", "60", "--seq-len", "32", "--lr", "3e-3",
        "--eval-every", "30", "--log-every", "30", "--device", "cpu", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started" and kinds[-1] == "finished" and events[-1]["status"] == "ok"
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"]

    spec = json.loads((run / "job.json").read_text())
    assert spec["stages"][0]["mtp_align"]["frozen_tensors"] == "frozen.safetensors"
    sidecar = run / f"{target.stem}-mtp-aligned.gguf"
    assert sidecar.exists()
    assert "aligned to this target" in out

    report = json.loads(mb("inspect", str(sidecar), "--json"))
    assert report["architecture"]["mtp_modules"] == 1

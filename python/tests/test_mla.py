"""``modelbuilder surgery mla`` and the MLA student path: exact at full rank, trainable when truncated."""

import json
import sys
from pathlib import Path

import pytest
import torch
from test_e2e import BIN, mb

pytestmark = pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")

COMMON = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 64,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
}


def _make(kind: str, out: Path) -> int:
    """Writes a tiny random HF checkpoint; returns its rotary dims per head."""
    import transformers as t

    torch.manual_seed(0)
    if kind == "llama":
        cfg = t.LlamaConfig(**COMMON)
    elif kind == "qwen2":  # Q/K/V biases
        cfg = t.Qwen2Config(**COMMON)
    elif kind == "qwen3":  # Q/K RMSNorm
        cfg = t.Qwen3Config(**COMMON)
    else:  # hybrid: DeltaNet + gated attention with partial RoPE
        cfg = t.Qwen3_5TextConfig(
            **COMMON | {"num_hidden_layers": 4},
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        )
    model = t.AutoModelForCausalLM.from_config(cfg).float()
    with torch.no_grad():  # random biases and norms, so they matter
        for n, p in model.named_parameters():
            if n.endswith("bias") or "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    model.save_pretrained(out)
    return 4 if kind == "qwen3_5" else 8


class Mode:
    student = True


def _logits(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    from transformers import AutoModelForCausalLM

    from modelbuilder_train.hf import mla

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    cfg = mla.load_config(model_dir)
    mode = Mode()
    mla.attach(model, model_dir, cfg, mode)
    ids = torch.arange(3, 23)[None] % 64
    with torch.no_grad():
        mode.student = False
        teacher = model(ids).logits
        mode.student = True
        student = model(ids).logits
    return teacher, student, cfg


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3", "qwen3_5"])
def test_full_rank_conversion_is_exact(tmp_path: Path, kind: str) -> None:
    pytest.importorskip("transformers")
    src = tmp_path / "src"
    rot = _make(kind, src)
    full = 2 * (8 - rot) + 2 * 8  # n_kv · (hd - r) + n_kv · hd, at most hidden = 32
    out = tmp_path / "mla"
    report = json.loads(
        mb("surgery", "mla", str(src), "--kv-rank", str(full), "--rope-dim", str(rot), "-o", str(out), "--json")
    )
    assert report["cache_after"] == full + 2 * rot
    assert all(abs(x["energy_kept"] - 1) < 1e-9 for x in report["layers"])
    assert (out / "model.safetensors").is_symlink()
    teacher, student, cfg = _logits(out)
    assert cfg["layers"] == ([3] if kind == "qwen3_5" else [0, 1])
    assert torch.allclose(teacher, student, atol=1e-4), (teacher - student).abs().max()

    # Truncated rank and partial RoPE: a different (lossy) student.
    low = tmp_path / "low"
    mb("surgery", "mla", str(src), "--kv-rank", "4", "--rope-dim", "2", "-o", str(low))
    teacher2, student2, _ = _logits(low)
    assert torch.allclose(teacher, teacher2, atol=1e-5)  # the teacher is the original model
    assert not torch.allclose(teacher2, student2, atol=1e-3)


def test_refusals(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    src = tmp_path / "src"
    _make("llama", src)
    out = tmp_path / "o"
    for args, why in [
        (["--kv-rank", "4", "--rope-dim", "3"], "even"),
        (["--kv-rank", "4", "--rope-dim", "10"], "rope_dim"),
        (["--kv-rank", "999", "--rope-dim", "2"], "kv_rank"),
    ]:
        r = mb("surgery", "mla", str(src), *args, "-o", str(out), check=False)
        assert why in r, r
    mb("surgery", "mla", str(src), "--kv-rank", "4", "--rope-dim", "2", "-o", str(out))
    again = mb("surgery", "mla", str(out), "--kv-rank", "4", "--rope-dim", "2", "-o", str(tmp_path / "o2"), check=False)
    assert "already" in again


def test_mla_recover_job_trains_the_latent(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    target = Path(mb("fixture", "gguf-llama", str(tmp_path / "t")).strip())
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(5 * i + 3 * j) % 96 for j in range(48)]}) + "\n" for i in range(8))
    )
    run = tmp_path / "run"
    out = mb(
        "job", "trunk-distill", str(target), "--reference", str(target.parent), "--texts", str(texts),
        "-o", str(run), "--mla-rank", "24", "--mla-rope-dim", "8", "--steps", "8", "--seq-len", "32",
        "--batch-seqs", "2", "--lr", "3e-3", "--eval-every", "8", "--log-every", "4", "--device", "cpu",
        "--fp32", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert events[-1]["event"] == "finished" and events[-1]["status"] == "ok"
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"], evals
    spec = json.loads((run / "job.json").read_text())
    assert spec["stages"][0]["name"] == "mla-recover"
    assert spec["stages"][0]["trunk_distill"]["trainable"] == ["self_attn.mla."]
    from safetensors import safe_open

    with safe_open(str(run / "updates" / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert keys and all(".self_attn.mla." in k for k in keys)

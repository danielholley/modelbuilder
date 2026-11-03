"""fetch-tensors: only the matching tensors, by byte range, stored as they were."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from modelbuilder_train.hub import fetch_tensors


def test_fetches_only_matching_tensors_by_range(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    a = {"model.layers.0.w": torch.randn(4, 3), "model.layers.1.w": torch.randn(2, 2)}
    b = {"model.layers.2.w": torch.randn(5).to(torch.bfloat16), "lm_head.weight": torch.randn(8, 3)}
    save_file(a, str(repo / "s1.safetensors"))
    save_file(b, str(repo / "s2.safetensors"))
    index = {"weight_map": {**{k: "s1.safetensors" for k in a}, **{k: "s2.safetensors" for k in b}}}
    (repo / "model.safetensors.index.json").write_text(json.dumps(index))
    (repo / "config.json").write_text('{"model_type": "x"}')
    calls = []

    def get(url: str, start=None, end=None) -> bytes:
        p = repo / url.rsplit("/", 1)[-1]
        if not p.exists():
            raise OSError("404")
        data = p.read_bytes()
        calls.append((p.name, start, end))
        return data if start is None else data[start : end + 1]

    got = fetch_tensors("org/m", r"layers\.(0|2)\.", tmp_path / "out", get=get)
    assert got == {"model.layers.0.w": [4, 3], "model.layers.2.w": [5]}
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.equal(out["model.layers.0.w"], a["model.layers.0.w"])
    assert out["model.layers.2.w"].dtype == torch.bfloat16
    assert torch.equal(out["model.layers.2.w"], b["model.layers.2.w"])
    assert (tmp_path / "out" / "config.json").exists()
    # Range requests only for the shards' headers and the wanted tensors.
    ranged = [c for c in calls if c[1] is not None]
    assert len(ranged) == 2 * 2 + 2

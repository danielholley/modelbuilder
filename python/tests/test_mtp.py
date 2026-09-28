import json
import subprocess
import sys

import pytest
import torch
from conftest import job

from modelbuilder_train.features import FeatureSet, FeatureWriter
from modelbuilder_train.mtp.model import MtpHead, load_mtp_state, save_mtp_state
from modelbuilder_train.mtp.train import lr_at, split
from modelbuilder_train.spec import Hyper

# The HF names that `modelbuilder surgery mtp` maps (crates/mb-surgery/src/mtp.rs TENSOR_MAP).
SURGERY_NAMES = {
    "fc.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
}


def test_parameter_names_match_the_surgery_mapping(tiny):
    assert set(MtpHead(tiny["cfg"]).state_dict()) == SURGERY_NAMES


def test_shapes_follow_the_hf_layout(tiny):
    c = tiny["cfg"]
    sd = MtpHead(c).state_dict()
    assert sd["fc.weight"].shape == (c.hidden_size, 2 * c.hidden_size)
    assert sd["layers.0.self_attn.q_proj.weight"].shape == (2 * c.num_heads * c.head_dim, c.hidden_size)
    assert sd["layers.0.self_attn.q_norm.weight"].shape == (c.head_dim,)
    assert c.rotary_dim == 4  # head_dim 8 × partial_rotary_factor 0.5


def test_save_load_round_trip(tiny, tmp_path):
    head = MtpHead(tiny["cfg"])
    save_mtp_state(head, tmp_path / "h.safetensors", dtype=torch.float32)
    from safetensors import safe_open

    with safe_open(str(tmp_path / "h.safetensors"), "pt") as f:
        assert all(k.startswith("mtp.") for k in f.keys())  # noqa: SIM118
    state = load_mtp_state([tmp_path / "h.safetensors"])
    other = MtpHead(tiny["cfg"])
    other.load_state_dict(state, strict=True)
    for (k, a), (_, b) in zip(head.state_dict().items(), other.state_dict().items(), strict=True):
        assert torch.equal(a, b), k


def test_fresh_norms_are_identity_scale_either_way(tiny):
    # Zero-centred norms start at weight 0 (scale 1), plain ones at weight 1.
    from modelbuilder_train.mtp.model import RMSNorm

    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    for offset in (1.0, 0.0):
        n = RMSNorm(4, 1e-6, offset)
        assert torch.allclose(n(x), x)
        assert n.weight.eq(0.0 if offset else 1.0).all()


def test_head_follows_the_config_family():
    from modelbuilder_train.mtp.model import MtpConfig, MtpHead

    llama = {"model_type": "llama", "hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
             "intermediate_size": 64, "rms_norm_eps": 1e-5, "rope_theta": 500000.0}  # fmt: skip
    c = MtpConfig.from_hf_config(llama)
    assert (c.norm_offset, c.attn_gate, c.qk_norm, c.head_dim, c.rotary_dim) == (0.0, False, False, 8, 8)
    names = set(MtpHead(c).state_dict())
    assert "layers.0.self_attn.q_norm.weight" not in names
    assert MtpHead(c).layers[0].self_attn.q_proj.out_features == 32  # no gate half
    out = MtpHead(c)(torch.randn(1, 5, 32), torch.randn(1, 5, 32), torch.arange(5))
    assert out.shape == (1, 5, 32)
    # A config key overrides the family default.
    assert MtpConfig.from_hf_config(llama | {"attn_output_gate": True}).attn_gate
    with pytest.raises(ValueError, match="no MTP-head family"):
        MtpConfig.from_hf_config(llama | {"model_type": "gpt2"})


def test_causal(tiny):
    """Changing a later position must not change earlier outputs."""
    head = MtpHead(tiny["cfg"]).eval()
    h, e = torch.randn(1, 6, 32), torch.randn(1, 6, 32)
    pos = torch.arange(6)
    out1 = head(h, e, pos)
    h2 = h.clone()
    h2[0, 5] += 1.0
    out2 = head(h2, e, pos)
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-6)
    assert not torch.allclose(out1[0, 5], out2[0, 5])


def test_feature_store_round_trip(tmp_path):
    w = FeatureWriter(tmp_path / "f", 4, {"runtime": "test"}, shard_tokens=10)
    for n in (6, 7, 3):
        w.add(list(range(n)), torch.full((n, 4), float(n)))
    m = w.close()
    assert (m["total_tokens"], m["total_sequences"], len(m["shards"])) == (16, 3, 2)
    fs = FeatureSet(tmp_path / "f")
    assert [s.length for s in fs.sequences] == [6, 7, 3]
    tokens, hidden = fs.get(fs.sequences[1], 2, 5)
    assert tokens.tolist() == [2, 3, 4] and hidden.shape == (3, 4) and float(hidden[0, 0]) == 7.0
    with pytest.raises(FileExistsError):
        FeatureWriter(tmp_path / "f", 4, {})
    with pytest.raises(ValueError):
        FeatureWriter(tmp_path / "g", 4, {}).add([1, 2], torch.zeros(3, 4))


def test_split_holds_out_the_tail(tiny):
    fs = FeatureSet(tiny["dir"] / "features")
    sp = split(fs, 0.2)
    assert sp.eval and sp.train and sp.train[-1] != sp.eval[0]
    assert sum(s.length for s in sp.eval) >= 0.2 * sum(s.length for s in fs.sequences)


def test_lr_schedule():
    h = Hyper(lr=1.0, steps=100, seq_len=8, batch_seqs=1, warmup_steps=10)
    assert lr_at(0, h) == pytest.approx(0.1)
    assert lr_at(9, h) == pytest.approx(1.0)
    assert lr_at(99, h) == pytest.approx(0.1, abs=1e-3)


def run_cli(tiny, spec: dict) -> list[dict]:
    path = tiny["dir"] / "job.json"
    path.write_text(json.dumps(spec))
    r = subprocess.run(
        [sys.executable, "-m", "modelbuilder_train", "run", str(path)], capture_output=True, text=True, timeout=600
    )
    events = [json.loads(line) for line in r.stdout.splitlines()]
    return r.returncode, events, r.stderr


def test_training_learns_and_reports_events(tiny):
    code, events, err = run_cli(tiny, job(tiny, steps=60))
    assert code == 0, err
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started" and kinds[-1] == "finished" and "checkpoint" in kinds
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[0]["step"] == 0 and evals[-1]["step"] == 60
    assert evals[-1]["loss"] < evals[0]["loss"] * 0.5, (evals[0], evals[-1])
    out = events[-1]["outputs"]["mtp_head"]
    # The saved head reloads into the same shapes, with the reference config beside it.
    state = load_mtp_state([tiny["dir"] / out / "model.safetensors"])
    MtpHead(tiny["cfg"]).load_state_dict(state, strict=True)
    assert (tiny["dir"] / out / "config.json").exists()


def test_failures_become_error_events(tiny):
    spec = job(tiny)
    spec["stages"][0]["mtp_align"]["features"] = "missing"
    code, events, _ = run_cli(tiny, spec)
    assert code == 1
    assert [e["event"] for e in events] == ["error", "finished"]
    assert events[-1]["status"] == "failed"


def test_data_parallel_under_torchrun(tiny):
    """Two CPU ranks (gloo): one event stream from rank 0, one saved head, and it still learns."""
    path = tiny["dir"] / "job.json"
    path.write_text(json.dumps(job(tiny, steps=40)))
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            "-m",
            "modelbuilder_train",
            "run",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    events = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    kinds = [e["event"] for e in events]
    assert kinds.count("started") == 1 and kinds.count("finished") == 1, kinds
    assert events[-1]["status"] == "ok"
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"]
    progress = [e for e in events if e["event"] == "progress"]
    # Tokens count both ranks' windows.
    assert progress[-1]["tokens"] == 2 * 40 * 2 * 16

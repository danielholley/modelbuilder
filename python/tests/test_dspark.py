"""DSpark drafters (DeepSpec semantics): masking, anchors, checkpoint names, and a job that learns."""

import json
import sys
from pathlib import Path

import pytest
import torch
from test_e2e import BIN, mb

from modelbuilder_train.draft.dspark import (
    DSpark,
    DSparkConfig,
    attention_mask,
    default_target_layers,
    forward_blocks,
    sample_anchors,
)

TARGET = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


def cfg(**kw) -> DSparkConfig:
    args = {
        "num_layers": 2,
        "block_size": 4,
        "mask_token_id": 63,
        "num_anchors": 6,
        "markov_rank": 8,
        "confidence": True,
    }
    return DSparkConfig.from_target(TARGET, **(args | kw))


def test_default_taps_reproduce_the_published_configs():
    assert default_target_layers(36) == [1, 9, 17, 25, 33]  # deepseek-ai/dspark_qwen3_4b_block7
    assert default_target_layers(64) == [1, 16, 31, 46, 61]  # satgeze/Qwen3.6-27B-DSpark
    assert default_target_layers(6) == [1, 2, 3, 4, 5]


def test_anchors_need_a_supervised_anchor_and_next_token():
    mask = torch.tensor([0, 1, 1, 1, 0, 1, 1])
    anchors, keep = sample_anchors(mask, 6)
    valid = anchors[keep].tolist()
    assert valid == [1, 2, 5]  # (1,2), (2,3), (5,6) are supervised pairs
    assert keep.tolist() == [True, True, True, False, False, False]


def test_blocks_see_earlier_context_and_their_own_block_only():
    anchors = torch.tensor([2, 5, 0])
    keep = torch.tensor([True, True, False])
    m = attention_mask(anchors, keep, seq_len=8, block=2)
    # Block 0 (anchor 2): context 0..1, its own two rows, not block 1's.
    assert m[0, :8].tolist() == [True, True] + [False] * 6
    assert m[0, 8:].tolist() == [True, True, False, False, False, False]
    assert m[2, :8].sum() == 5 and m[2, 10:12].all() and not m[2, 8:10].any()
    # A dummy block sees no context, only itself (so no empty softmax rows).
    assert not m[4, :8].any() and m[4, 12:].all()


def test_slot_i_predicts_the_token_after_anchor_plus_i():
    torch.manual_seed(0)
    c = cfg()
    model = DSpark(c).eval()
    ids = torch.arange(10, 30)
    taps = torch.randn(20, len(c.target_layer_ids) * 32)
    last = torch.randn(20, 32)
    anchors, keep = torch.tensor([3, 10, 17, 0, 0, 0]), torch.tensor([True, True, True, False, False, False])
    logits, targets, supervised, conf, aligned = forward_blocks(model, ids, taps, last, torch.ones(20), anchors, keep)
    assert logits.shape == (6, 4, 64) and conf.shape == (6, 4)
    assert targets[0].tolist() == [14, 15, 16, 17]  # x[anchor + 1 ..]
    # Block 2 (anchor 17) runs off the end after two targets: a supervised prefix.
    assert supervised[2].tolist() == [True, True, False, False]
    assert not supervised[3:].any()
    # The target's aligned distribution comes from the hidden state one before each label.
    assert torch.allclose(aligned[0, 0], model.lm_head(last[3]))


def test_state_dict_names_match_the_published_checkpoint():
    names = set(DSpark(cfg()).state_dict())
    # deepseek-ai/dspark_qwen3_4b_block7's top-level and per-layer names.
    expected = {
        "embed_tokens.weight", "lm_head.weight", "fc.weight", "hidden_norm.weight", "norm.weight",
        "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
        "confidence_head.proj.weight", "confidence_head.proj.bias",
    } | {
        f"layers.{i}.{n}"
        for i in range(2)
        for n in [
            "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        ]
    }  # fmt: skip
    assert names == expected


@pytest.mark.skipif(not BIN.exists(), reason=f"no modelbuilder binary at {BIN}")
def test_dspark_job_learns_and_writes_the_qwen3dspark_format(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    target = Path(mb("fixture", "gguf-qwen3", str(tmp_path / "t")).strip())
    texts = tmp_path / "t.jsonl"
    texts.write_text(
        "".join(json.dumps({"tokens": [(7 * i + 3 * j) % 90 for j in range(40)]}) + "\n" for i in range(10))
    )
    run = tmp_path / "run"
    out = mb(
        "job", "draft-align", str(target), "--kind", "dspark", "--reference", str(target.parent),
        "--texts", str(texts), "--blocks", "2", "--block-size", "4", "--target-layers", "0,1",
        "--mask-token-id", "95", "--markov-rank", "8", "--num-anchors", "16", "-o", str(run),
        "--steps", "30", "--seq-len", "32", "--batch-seqs", "2", "--lr", "3e-3", "--eval-every", "30",
        "--log-every", "10", "--device", "cpu", "--fp32", "--raw-events", "--python", sys.executable,
    )  # fmt: skip
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert events[-1]["event"] == "finished" and events[-1]["status"] == "ok", out[-2000:]
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"], evals
    head = run / "dspark-head"
    c = json.loads((head / "config.json").read_text())
    assert c["architectures"] == ["Qwen3DSparkModel"]
    assert (c["block_size"], c["target_layer_ids"], c["mask_token_id"], c["num_hidden_layers"]) == (4, [0, 1], 95, 2)
    assert c["num_target_layers"] == 2 and c["markov_head_type"] == "vanilla" and c["enable_confidence_head"]
    from safetensors import safe_open

    with safe_open(str(head / "model.safetensors"), "pt") as f:
        keys = set(f.keys())
        assert f.get_slice("fc.weight").get_shape() == [64, 128]  # 2 taps × hidden 64
    assert {"markov_head.markov_w1.weight", "confidence_head.proj.weight", "layers.1.self_attn.k_norm.weight"} <= keys
    spec = json.loads((run / "job.json").read_text())["stages"][0]["draft_align"]
    assert spec["kind"] == "dspark" and spec["target"] == "target-hf"

import json
from pathlib import Path

import pytest
import torch

from modelbuilder_train.features import FeatureWriter
from modelbuilder_train.mtp.model import MtpConfig, MtpHead, save_mtp_state

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schema"

TINY_HF_CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "mtp_num_hidden_layers": 1,
        "vocab_size": 50,
    },
}


@pytest.fixture
def tiny(tmp_path: Path) -> dict:
    """A tiny, fully self-consistent MTP-align setup on disk."""
    torch.manual_seed(0)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(TINY_HF_CONFIG))
    cfg = MtpConfig.from_hf_config(TINY_HF_CONFIG)
    vocab, h = 50, cfg.hidden_size

    head_dir = tmp_path / "init"
    head_dir.mkdir()
    save_mtp_state(MtpHead(cfg), head_dir / "model.safetensors", dtype=torch.float32)

    from safetensors.torch import save_file

    emb = torch.randn(vocab, h)
    lm = torch.randn(vocab, h)
    save_file({"token_embd.weight": emb, "output.weight": lm}, str(tmp_path / "frozen.safetensors"))

    # A learnable pattern: the token after next is (token + 2) mod vocab.
    w = FeatureWriter(tmp_path / "features", h, {"runtime": "test"})
    for s in range(12):
        tokens = [(s + i) % vocab for i in range(40)]
        hidden = emb[torch.tensor(tokens)] + 0.1 * torch.randn(40, h)
        w.add(tokens, hidden)
    w.close()
    return {"dir": tmp_path, "cfg": cfg, "config_path": cfg_path, "init": head_dir}


def job(tiny: dict, steps: int = 40, **hyper) -> dict:
    return {
        "schema_version": 1,
        "job_id": "tiny",
        "backend": "torch",
        "device": "cpu",
        "output_dir": "out",
        "stages": [
            {
                "name": "mtp-align",
                "kind": "mtp_align",
                "mtp_align": {
                    "reference_config": "config.json",
                    "init_head": "init",
                    "frozen_tensors": "frozen.safetensors",
                    "embedding_tensor": "token_embd.weight",
                    "lm_head_tensor": "output.weight",
                    "features": "features",
                    "eval_fraction": 0.2,
                },
                "hyper": {"lr": 3e-3, "steps": steps, "seq_len": 16, "batch_seqs": 2, "log_every": 5, "eval_every": 20}
                | hyper,
            }
        ],
    }

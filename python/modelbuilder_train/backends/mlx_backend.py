"""The ``mlx`` backend: frozen-trunk stages on Apple Silicon with MLX.

Runs ``mtp_align`` exactly as the torch backend does (same features, same
windows, same loss, same events and the same ``mtp.*`` output), with the head
in MLX (``mtp/mlx_model.py``). Feature shards are read with the shared
``FeatureSet`` and handed to MLX one window at a time. Other stage kinds need
the torch backend (which also runs on Apple GPUs through MPS).
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from modelbuilder_train.events import EventWriter
from modelbuilder_train.features import FeatureSet
from modelbuilder_train.mtp.mlx_model import MtpHead
from modelbuilder_train.mtp.model import MTP_PREFIX, MtpConfig
from modelbuilder_train.mtp.train import LOGIT_CHUNK, lr_at, split
from modelbuilder_train.spec import Hyper, JobSpec, MtpAlign, resolve


def _mx(t) -> mx.array:
    """A torch tensor as an MLX array (bf16 goes through float32: numpy has no bf16)."""
    import torch

    if t.dtype == torch.bfloat16:
        t = t.float()
    return mx.array(t.numpy())


def load_head_state(paths: list[Path]) -> dict[str, mx.array]:
    state: dict[str, mx.array] = {}
    for p in paths:
        for k, v in mx.load(str(p)).items():
            for prefix in (MTP_PREFIX, "model." + MTP_PREFIX):
                if k.startswith(prefix):
                    state[k[len(prefix) :]] = v.astype(mx.float32)
    return state


def head_loss(head: MtpHead, emb: mx.array, lm_head: mx.array, tokens: mx.array, hidden: mx.array):
    """Summed cross-entropy and correct top-1 count over one window (as ``mtp.train.head_loss``)."""
    n = tokens.shape[0]
    h = hidden[: n - 2].astype(mx.float32)[None]
    e = emb[tokens[1 : n - 1]].astype(mx.float32)[None]
    out = head(h, e, mx.arange(1, n - 1))[0]
    targets = tokens[2:]
    total, correct = mx.array(0.0), mx.array(0)
    for a in range(0, out.shape[0], LOGIT_CHUNK):
        logits = (out[a : a + LOGIT_CHUNK].astype(lm_head.dtype) @ lm_head.T).astype(mx.float32)
        t = targets[a : a + LOGIT_CHUNK]
        total = total + nn.losses.cross_entropy(logits, t, reduction="sum")
        correct = correct + (mx.argmax(logits, axis=-1) == t).sum()
    return total, correct


def run_mtp_align(stage: MtpAlign, hyper: Hyper, res, out_dir: Path, ev: EventWriter, job_id: str) -> dict:
    mx.random.seed(hyper.seed)
    rng = random.Random(hyper.seed)
    ref_config_path = res(stage.reference_config)
    cfg = MtpConfig.from_hf_config(json.loads(ref_config_path.read_text()))
    head = MtpHead(cfg)
    if stage.init_head:
        src = res(stage.init_head)
        files = sorted(src.glob("*.safetensors")) if src.is_dir() else [src]
        state = load_head_state(files)
        ours = dict(tree_flatten(head.parameters()))
        if set(state) != set(ours):
            raise ValueError(f"head tensors differ: missing {set(ours) - set(state)}, extra {set(state) - set(ours)}")
        head.update(tree_unflatten(list(state.items())))
    dtype = mx.bfloat16 if hyper.dtype == "bfloat16" else mx.float32
    frozen = mx.load(str(res(stage.frozen_tensors)))
    emb = frozen[stage.embedding_tensor].astype(dtype)
    lm_head = frozen[stage.lm_head_tensor].astype(dtype)
    features = FeatureSet(res(stage.features))
    if features.hidden_size != cfg.hidden_size or emb.shape[1] != cfg.hidden_size:
        raise ValueError(f"features/frozen width {features.hidden_size}/{emb.shape[1]} vs the head's {cfg.hidden_size}")
    sp = split(features, stage.eval_fraction)

    def window(ref, a, b):
        tokens, hidden = features.get(ref, a, b)
        return mx.array(tokens.numpy().astype(np.int32)), _mx(hidden)

    def evaluate() -> dict:
        loss_sum, correct, count = 0.0, 0, 0
        for ref in sp.eval:
            for a in range(0, ref.length - 2, hyper.seq_len):
                b = min(ref.length, a + hyper.seq_len + 2)
                if b - a < 4:
                    continue
                tokens, hidden = window(ref, a, b)
                loss, c = head_loss(head, emb, lm_head, tokens, hidden)
                loss_sum, correct, count = loss_sum + loss.item(), correct + c.item(), count + b - a - 2
        return {"loss": loss_sum / max(count, 1), "accuracy": correct / max(count, 1), "tokens": count}

    ev.emit(
        "started",
        job_id=job_id,
        device=f"mlx:{mx.default_device()}",
        trainable_params=sum(v.size for _, v in tree_flatten(head.parameters())),
        train_tokens=sum(s.length for s in sp.train),
        eval_tokens=sum(s.length for s in sp.eval),
    )
    opt = optim.AdamW(learning_rate=hyper.lr, weight_decay=hyper.weight_decay)

    def batch_loss(model, batch, n_targets):
        total, correct = mx.array(0.0), mx.array(0)
        for tokens, hidden in batch:
            loss, c = head_loss(model, emb, lm_head, tokens, hidden)
            total, correct = total + loss, correct + c
        return total / n_targets, (total, correct)

    step_fn = nn.value_and_grad(head, batch_loss)
    if sp.eval:
        ev.emit("eval", step=0, **evaluate())
    win = hyper.seq_len + 2
    tokens_seen, t0 = 0, time.time()
    for step in range(1, hyper.steps + 1):
        lr = lr_at(step - 1, hyper)
        opt.learning_rate = lr
        batch, n_targets = [], 0
        for _ in range(hyper.batch_seqs):
            ref = rng.choice(sp.train)
            a = rng.randrange(0, max(1, ref.length - win + 1))
            b = min(ref.length, a + win)
            batch.append(window(ref, a, b))
            n_targets += b - a - 2
        (_, (total, correct)), grads = step_fn(head, batch, n_targets)
        if hyper.grad_clip:
            grads, _ = optim.clip_grad_norm(grads, hyper.grad_clip)
        opt.update(head, grads)
        mx.eval(head.parameters(), opt.state, total, correct)
        tokens_seen += n_targets
        if step % hyper.log_every == 0 or step == hyper.steps:
            ev.emit(
                "progress",
                step=step,
                steps=hyper.steps,
                tokens=tokens_seen,
                loss=total.item() / n_targets,
                accuracy=correct.item() / n_targets,
                lr=lr,
                tokens_per_s=tokens_seen / max(time.time() - t0, 1e-9),
            )
        if sp.eval and (step % hyper.eval_every == 0 or step == hyper.steps):
            ev.emit("eval", step=step, **evaluate())

    out_head = out_dir / "mtp-head"
    out_head.mkdir(parents=True, exist_ok=True)
    tensors = {MTP_PREFIX + k: v.astype(mx.bfloat16) for k, v in tree_flatten(head.parameters())}
    meta = {"format": "pt", "modelbuilder": "mtp-head"}
    mx.save_safetensors(str(out_head / "model.safetensors"), tensors, metadata=meta)
    shutil.copyfile(ref_config_path, out_head / "config.json")
    ev.emit("checkpoint", step=hyper.steps, path=str(out_head / "model.safetensors"))
    return {"mtp_head": str(out_head)}


def run_mlx(spec: JobSpec, spec_dir: Path, ev: EventWriter) -> dict[str, str]:
    out_dir = resolve(spec_dir, spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for stage in spec.stages:
        ev.stage = stage.name
        if stage.kind != "mtp_align":
            raise ValueError(
                f"stage kind {stage.kind} is not supported by the mlx backend; use backend torch (MPS on Apple GPUs)"
            )
        outputs.update(
            run_mtp_align(stage.mtp_align, stage.hyper, lambda p: resolve(spec_dir, p), out_dir, ev, spec.job_id)
        )
    ev.stage = None
    return outputs

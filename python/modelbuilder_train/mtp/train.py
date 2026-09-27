"""The ``mtp_align`` stage: train an MTP head against a frozen trunk.

Inputs are precomputed trunk features (``modelbuilder_train.features``), so the
trunk itself never runs here. The frozen token embedding and output head come
from ``modelbuilder export-tensors`` (decoded, primal basis). Each training
window of tokens ``x[0..L]`` with hidden states ``h[0..L]`` gives the pairs
``(h[t], x[t+1]) -> x[t+2]``, exactly what the runtime feeds the draft head.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modelbuilder_train.events import EventWriter
from modelbuilder_train.features import FeatureSet, SequenceRef
from modelbuilder_train.mtp.model import MtpConfig, MtpHead, load_mtp_state, save_mtp_state
from modelbuilder_train.spec import Hyper, MtpAlign

#: Rows of the (tokens × vocab) logits computed at once.
LOGIT_CHUNK = 256


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _chunk_ce(x: torch.Tensor, lm_head: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (x.to(lm_head.dtype) @ lm_head.T).float()
    loss = F.cross_entropy(logits, targets, reduction="sum")
    return loss, (logits.argmax(-1) == targets).sum()


def head_loss(
    head: MtpHead, emb: torch.Tensor, lm_head: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    """Summed cross-entropy, correct top-1 count and number of targets for one window."""
    n = tokens.shape[0]
    h = hidden[: n - 2].float()[None]
    e = emb[tokens[1 : n - 1]].float()[None]
    positions = torch.arange(1, n - 1, device=tokens.device)
    out = head(h, e, positions)[0]
    targets = tokens[2:]
    total = out.new_zeros(())
    correct = 0
    for a in range(0, out.shape[0], LOGIT_CHUNK):
        sl = slice(a, a + LOGIT_CHUNK)
        if out.requires_grad:
            loss, c = checkpoint(_chunk_ce, out[sl], lm_head, targets[sl], use_reentrant=False)
        else:
            loss, c = _chunk_ce(out[sl], lm_head, targets[sl])
        total = total + loss
        correct += int(c)
    return total, correct, targets.shape[0]


@dataclass
class Split:
    train: list[SequenceRef]
    eval: list[SequenceRef]


def split(features: FeatureSet, eval_fraction: float) -> Split:
    """Deterministic split: the last sequences (by token count) are held out."""
    seqs = [s for s in features.sequences if s.length >= 4]
    if not seqs:
        raise ValueError("no feature sequence has at least 4 tokens")
    total = sum(s.length for s in seqs)
    held, cut = 0, len(seqs)
    while eval_fraction > 0 and cut > 1 and held < eval_fraction * total:
        cut -= 1
        held += seqs[cut].length
    return Split(seqs[:cut], seqs[cut:])


def evaluate(head: MtpHead, emb, lm_head, features: FeatureSet, seqs: list[SequenceRef], max_len: int, device) -> dict:
    head.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for ref in seqs:
            for a in range(0, ref.length - 2, max_len):
                b = min(ref.length, a + max_len + 2)
                if b - a < 4:
                    continue
                tokens, hidden = features.get(ref, a, b)
                loss, c, k = head_loss(head, emb, lm_head, tokens.to(device), hidden.to(device))
                loss_sum, correct, count = loss_sum + float(loss), correct + c, count + k
    head.train()
    return {"loss": loss_sum / max(count, 1), "accuracy": correct / max(count, 1), "tokens": count}


def lr_at(step: int, hyper: Hyper) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < hyper.warmup_steps:
        return hyper.lr * (step + 1) / hyper.warmup_steps
    span = max(1, hyper.steps - hyper.warmup_steps)
    progress = min(1.0, (step - hyper.warmup_steps) / span)
    return hyper.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _files(path: Path) -> list[Path]:
    return sorted(path.glob("*.safetensors")) if path.is_dir() else [path]


def run_mtp_align(
    stage: MtpAlign, hyper: Hyper, resolve, out_dir: Path, device: torch.device, ev: EventWriter, job_id: str
) -> dict:
    torch.manual_seed(hyper.seed)
    rng = random.Random(hyper.seed)

    ref_config_path = resolve(stage.reference_config)
    cfg = MtpConfig.from_hf_config(json.loads(ref_config_path.read_text()))
    head = MtpHead(cfg)
    if stage.init_head:
        state = load_mtp_state(_files(resolve(stage.init_head)))
        head.load_state_dict(state, strict=True)
    head.to(device).train()

    from safetensors import safe_open

    frozen_dtype = torch.bfloat16 if hyper.dtype == "bfloat16" else torch.float32
    with safe_open(str(resolve(stage.frozen_tensors)), framework="pt") as f:
        emb = f.get_tensor(stage.embedding_tensor).to(device=device, dtype=frozen_dtype)
        lm_head = f.get_tensor(stage.lm_head_tensor).to(device=device, dtype=frozen_dtype)
    if emb.shape[1] != cfg.hidden_size or lm_head.shape[1] != cfg.hidden_size:
        raise ValueError(
            f"frozen tensors have width {emb.shape[1]}/{lm_head.shape[1]}, the head expects {cfg.hidden_size}"
        )

    features = FeatureSet(resolve(stage.features))
    if features.hidden_size != cfg.hidden_size:
        raise ValueError(f"features have hidden size {features.hidden_size}, the head expects {cfg.hidden_size}")
    sp = split(features, stage.eval_fraction)

    ev.emit(
        "started",
        job_id=job_id,
        device=str(device),
        trainable_params=sum(p.numel() for p in head.parameters()),
        train_tokens=sum(s.length for s in sp.train),
        eval_tokens=sum(s.length for s in sp.eval),
    )

    opt = torch.optim.AdamW(head.parameters(), lr=hyper.lr, weight_decay=hyper.weight_decay)
    autocast = hyper.dtype == "bfloat16" and device.type in ("cuda", "cpu")
    window = hyper.seq_len + 2
    tokens_seen, t0 = 0, time.time()
    out_head = out_dir / "mtp-head"

    def save(step: int) -> None:
        out_head.mkdir(parents=True, exist_ok=True)
        save_mtp_state(head, out_head / "model.safetensors")
        shutil.copyfile(ref_config_path, out_head / "config.json")
        ev.emit("checkpoint", step=step, path=str(out_head / "model.safetensors"))

    if sp.eval:
        # Baseline: how well the initial (e.g. ported) head already drafts.
        ev.emit("eval", step=0, **evaluate(head, emb, lm_head, features, sp.eval, hyper.seq_len, device))

    for step in range(1, hyper.steps + 1):
        lr = lr_at(step - 1, hyper)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        batch_loss, batch_correct, batch_count = 0.0, 0, 0
        windows = []
        for _ in range(hyper.batch_seqs):
            ref = rng.choice(sp.train)
            a = rng.randrange(0, max(1, ref.length - window + 1))
            windows.append((ref, a, min(ref.length, a + window)))
        n_targets = sum(b - a - 2 for _, a, b in windows)
        for ref, a, b in windows:
            tokens, hidden = features.get(ref, a, b)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                loss, correct, count = head_loss(head, emb, lm_head, tokens.to(device), hidden.to(device))
            (loss / n_targets).backward()
            batch_loss, batch_correct, batch_count = (
                batch_loss + float(loss.detach()),
                batch_correct + correct,
                batch_count + count,
            )
        if hyper.grad_clip:
            torch.nn.utils.clip_grad_norm_(head.parameters(), hyper.grad_clip)
        opt.step()
        tokens_seen += batch_count

        if step % hyper.log_every == 0 or step == hyper.steps:
            ev.emit(
                "progress",
                step=step,
                steps=hyper.steps,
                tokens=tokens_seen,
                loss=batch_loss / batch_count,
                accuracy=batch_correct / batch_count,
                lr=lr,
                tokens_per_s=tokens_seen / max(time.time() - t0, 1e-9),
            )
        if sp.eval and (step % hyper.eval_every == 0 or step == hyper.steps):
            ev.emit("eval", step=step, **evaluate(head, emb, lm_head, features, sp.eval, hyper.seq_len, device))

    save(hyper.steps)
    return {"mtp_head": str(out_head)}

"""The ``draft_align`` stage: train a Medusa or EAGLE drafter against a frozen target.

Like ``mtp_align``, it reads precomputed trunk features (post-norm hidden
states and their tokens), so the target never runs here; the frozen token
embedding and LM head come from ``modelbuilder export-tensors``. For a window
of tokens ``x[0..n]`` with features ``h[0..n]``:

- Medusa head ``k`` (1-based) learns ``h_t → x[t+k+1]``, weighted ``0.8^k``.
- EAGLE learns ``(h_t, x[t+1]) → h_{t+1}`` (SmoothL1) and, through the frozen
  LM head, ``→ x[t+2]`` (cross-entropy × 0.1).

The reported ``accuracy`` is top-1 on ``x[t+2]`` for both, the same target
``mtp_align`` reports, so the three drafters compare directly.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modelbuilder_train import dist
from modelbuilder_train.draft.model import EAGLE_CLS_WEIGHT, MEDUSA_DECAY, Eagle, Medusa
from modelbuilder_train.events import EventWriter
from modelbuilder_train.features import FeatureSet
from modelbuilder_train.mtp.model import MtpConfig
from modelbuilder_train.mtp.train import LOGIT_CHUNK, lr_at, split, supports_bf16
from modelbuilder_train.spec import DraftAlign, Hyper


def _chunk_ce(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (x.to(w.dtype) @ w.T).float()
    return F.cross_entropy(logits, targets, reduction="sum"), (logits.argmax(-1) == targets).sum()


def chunked_ce(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Summed cross-entropy and top-1 hits, a chunk of rows at a time (the vocabulary may be large)."""
    total, hits = x.new_zeros((), dtype=torch.float32), 0
    for a in range(0, x.shape[0], LOGIT_CHUNK):
        sl = slice(a, a + LOGIT_CHUNK)
        if x.requires_grad or w.requires_grad:
            loss, c = checkpoint(_chunk_ce, x[sl], w, targets[sl], use_reentrant=False)
        else:
            loss, c = _chunk_ce(x[sl], w, targets[sl])
        total, hits = total + loss, hits + int(c)
    return total, hits


def window_loss(model, kind: str, emb, lm_head, tokens, hidden) -> tuple[torch.Tensor, int, int]:
    """Weighted summed loss, top-1 hits on x[t+2], and the number of x[t+2] targets."""
    n = tokens.shape[0]
    h = hidden.float()
    if kind == "medusa":
        core = model.module if hasattr(model, "module") else model
        outs = model(h[None])
        total, hits1, count1 = h.new_zeros(()), 0, 0
        for k in range(core.heads):
            m = n - (k + 2)  # positions t with a target x[t+k+2]
            if m <= 0:
                break
            loss, hits = chunked_ce(outs[k][0, :m], core.head_weight(k, lm_head), tokens[k + 2 : k + 2 + m])
            total = total + MEDUSA_DECAY ** (k + 1) * loss
            if k == 0:
                hits1, count1 = hits, m
        return total, hits1, count1
    # EAGLE: inputs at t = 0..n-3, feature targets h[t+1], token targets x[t+2].
    m = n - 2
    e = emb[tokens[1 : 1 + m]].float()[None]
    positions = torch.arange(0, m, device=tokens.device)
    pred = model(h[None, :m], e, positions)[0]
    reg = F.smooth_l1_loss(pred, h[1 : 1 + m], reduction="sum") / h.shape[-1]
    ce, hits = chunked_ce(pred, lm_head, tokens[2 : 2 + m])
    return reg + EAGLE_CLS_WEIGHT * ce, hits, m


def run_draft_align(
    stage: DraftAlign, hyper: Hyper, resolve, out_dir: Path, device: torch.device, ev: EventWriter, job_id: str
) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.manual_seed(hyper.seed)
    rng = random.Random(hyper.seed + 1_000_003 * dist.rank())
    use_bf16 = hyper.dtype == "bfloat16" and supports_bf16(device)
    frozen_dtype = torch.bfloat16 if use_bf16 else torch.float32
    with safe_open(str(resolve(stage.frozen_tensors)), framework="pt") as f:
        emb = f.get_tensor(stage.embedding_tensor).to(device=device, dtype=frozen_dtype)
        lm_head = f.get_tensor(stage.lm_head_tensor).to(device=device, dtype=frozen_dtype)
    features = FeatureSet(resolve(stage.features))
    hidden = features.hidden_size
    if emb.shape[1] != hidden or lm_head.shape[1] != hidden:
        raise ValueError(f"frozen tensors have width {emb.shape[1]}/{lm_head.shape[1]}, the features {hidden}")

    config: dict = {"kind": stage.kind, "hidden_size": hidden, "vocab_size": int(lm_head.shape[0])}
    if stage.kind == "medusa":
        head = Medusa(hidden, stage.heads, None if stage.share_lm_head else lm_head.float().cpu())
        config |= {"architectures": ["MedusaModel"], "num_heads": stage.heads, "num_hidden_layers": 1}
        config["share_lm_head"] = stage.share_lm_head
    else:
        if stage.reference_config is None:
            raise ValueError("EAGLE needs reference_config (the target's HF config) for the decoder layer's shape")
        ref = json.loads(resolve(stage.reference_config).read_text())
        cfg = MtpConfig.from_hf_config(ref)
        if cfg.hidden_size != hidden:
            raise ValueError(f"reference_config has hidden size {cfg.hidden_size}, the features {hidden}")
        head = Eagle(cfg, stage.blocks)
        config |= {"architectures": ["EagleModel"], "num_hidden_layers": stage.blocks, "base_config": ref}
    head.to(device).train()
    sp = split(features, stage.eval_fraction)

    def evaluate() -> dict:
        head.eval()
        loss_sum, hits, count = 0.0, 0, 0
        with torch.no_grad():
            for r in sp.eval:
                for a in range(0, r.length - 2, hyper.seq_len):
                    b = min(r.length, a + hyper.seq_len + 2)
                    if b - a < 4:
                        continue
                    tokens, h = features.get(r, a, b)
                    loss, c, k = window_loss(head, stage.kind, emb, lm_head, tokens.to(device), h.to(device))
                    loss_sum, hits, count = loss_sum + float(loss), hits + c, count + k
        head.train()
        return {"loss": loss_sum / max(count, 1), "accuracy": hits / max(count, 1), "tokens": count}

    ev.emit(
        "started",
        job_id=job_id,
        device=str(device),
        trainable_params=sum(p.numel() for p in head.parameters()),
        train_tokens=sum(s.length for s in sp.train),
        eval_tokens=sum(s.length for s in sp.eval),
    )
    model = dist.wrap(head, device)
    opt = torch.optim.AdamW(head.parameters(), lr=hyper.lr, weight_decay=hyper.weight_decay)
    if sp.eval and dist.is_main():
        ev.emit("eval", step=0, **evaluate())
    window = hyper.seq_len + 2
    tokens_seen, t0 = 0, time.time()
    for step in range(1, hyper.steps + 1):
        lr = lr_at(step - 1, hyper)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        windows = []
        for _ in range(hyper.batch_seqs):
            r = rng.choice(sp.train)
            a = rng.randrange(0, max(1, r.length - window + 1))
            windows.append((r, a, min(r.length, a + window)))
        n_targets = sum(b - a - 2 for _, a, b in windows)
        loss_sum, hits, count = 0.0, 0, 0
        for w, (r, a, b) in enumerate(windows):
            tokens, h = features.get(r, a, b)
            sync = w == len(windows) - 1 or not hasattr(model, "no_sync")
            with (
                contextlib.nullcontext() if sync else model.no_sync(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16),
            ):
                loss, c, k = window_loss(model, stage.kind, emb, lm_head, tokens.to(device), h.to(device))
                (loss / n_targets).backward()
            loss_sum, hits, count = loss_sum + float(loss.detach()), hits + c, count + k
        if hyper.grad_clip:
            torch.nn.utils.clip_grad_norm_(head.parameters(), hyper.grad_clip)
        opt.step()
        tokens_seen += count * dist.world()
        if dist.is_main() and (step % hyper.log_every == 0 or step == hyper.steps):
            ev.emit(
                "progress",
                step=step,
                steps=hyper.steps,
                tokens=tokens_seen,
                loss=loss_sum / max(count, 1),
                accuracy=hits / max(count, 1),
                lr=lr,
                tokens_per_s=tokens_seen / max(time.time() - t0, 1e-9),
            )
        if sp.eval and dist.is_main() and (step % hyper.eval_every == 0 or step == hyper.steps):
            ev.emit("eval", step=step, **evaluate())

    out = out_dir / f"{stage.kind}-head"
    if dist.is_main():
        out.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()}
        save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "modelbuilder": stage.kind})
        (out / "config.json").write_text(json.dumps(config, indent=2))
        ev.emit("checkpoint", step=hyper.steps, path=str(out / "model.safetensors"))
    return {"draft_head": str(out)}

"""The ``draft_align`` stage with ``kind = "dspark"``: train a DSpark drafter against a frozen target.

DSpark reads hidden states tapped from several target layers, which the
precomputed features (final hidden state only) don't have, so the target runs
online here: a Hugging Face checkpoint (e.g. from ``modelbuilder export-hf``)
in transformers, frozen, with forward hooks on the tapped layers (as DeepSpec's
``prepare_target_cache.py``). Every token of the texts is supervised (DeepSpec
supervises assistant turns only; use the model's own generations as texts).

The output ``dspark-head/`` is an HF checkpoint in the ``Qwen3DSparkModel``
format: convert it with the PrismML fork's
``convert_hf_to_gguf.py dspark-head --target-model-dir <target hf>`` and run it
with ``--spec-type draft-dspark``.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

from modelbuilder_train.draft.dspark import (
    DEFAULTS,
    DSpark,
    DSparkConfig,
    dspark_loss,
    forward_blocks,
    sample_anchors,
    tap_target,
)
from modelbuilder_train.events import EventWriter
from modelbuilder_train.hf.distill import load_sequences
from modelbuilder_train.mtp.train import lr_at, supports_bf16
from modelbuilder_train.spec import DraftAlign, Hyper


def run_dspark(
    stage: DraftAlign, hyper: Hyper, resolve, out_dir: Path, device: torch.device, ev: EventWriter, job_id: str
) -> dict:
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    torch.manual_seed(hyper.seed)
    rng = random.Random(hyper.seed)
    target_dir = resolve(stage.target)
    target_cfg = json.loads((target_dir / "config.json").read_text())
    dtype = torch.bfloat16 if hyper.dtype == "bfloat16" and supports_bf16(device) else torch.float32
    target = AutoModelForCausalLM.from_pretrained(target_dir, dtype=dtype).to(device).eval()
    target.requires_grad_(False)
    vocab = target.get_output_embeddings().weight.shape[0]
    mask_id = stage.mask_token_id
    if mask_id is None:
        mask_id = vocab - 1
        print(
            f"dspark: mask_token_id defaults to the last embedding row ({mask_id}); "
            "pick an id the tokenizer never produces",
            file=sys.stderr,
        )
    cfg = DSparkConfig.from_target(
        target_cfg,
        target_layer_ids=stage.target_layer_ids,
        num_layers=stage.blocks,
        block_size=stage.block_size,
        mask_token_id=mask_id,
        num_anchors=stage.num_anchors,
        markov_rank=stage.markov_rank,
        confidence=stage.confidence_head,
    )
    model = DSpark(cfg).to(device)
    model.init_from_target(target.get_input_embeddings().weight, target.get_output_embeddings().weight)
    model.to(dtype).train()

    seqs = [s for s in load_sequences(resolve(stage.texts), target_dir) if len(s) >= 3]
    if not seqs:
        raise ValueError("no text with at least 3 tokens")
    cut = len(seqs) - max(1, round(stage.eval_fraction * len(seqs))) if stage.eval_fraction > 0 else len(seqs)
    train, held = seqs[: max(cut, 1)], seqs[max(cut, 1) :]
    params = [p for p in model.parameters() if p.requires_grad]
    w = {
        "gamma": DEFAULTS["loss_decay_gamma"],
        "ce_alpha": DEFAULTS["ce_loss_alpha"],
        "l1_alpha": DEFAULTS["l1_loss_alpha"],
        "conf_alpha": DEFAULTS["confidence_head_alpha"],
    }

    def window(seq: list[int]) -> torch.Tensor:
        n = min(len(seq), hyper.seq_len)
        a = rng.randrange(0, len(seq) - n + 1)
        return torch.tensor(seq[a : a + n], device=device)

    def step_on(ids: torch.Tensor, gen: torch.Generator | None = None):
        taps, last = tap_target(target, ids, cfg.target_layer_ids)
        loss_mask = torch.ones_like(ids)
        anchors, keep = sample_anchors(loss_mask, cfg.num_anchors, gen)
        out = forward_blocks(model, ids, taps, last, loss_mask, anchors, keep)
        return dspark_loss(*out, **w)

    def evaluate() -> dict:
        model.eval()
        gen = torch.Generator().manual_seed(1234)  # the same anchors every time
        loss_sum, hits, count, taus = 0.0, 0, 0, []
        with torch.no_grad():
            for seq in held:
                loss, m = step_on(torch.tensor(seq[: hyper.seq_len], device=device), gen)
                loss_sum, hits, count = loss_sum + float(loss) * m["count"], hits + m["hits"], count + m["count"]
                taus.append(m["tau"])
        model.train()
        return {"loss": loss_sum / max(count, 1), "accuracy": hits / max(count, 1), "tokens": count}

    ev.emit(
        "started",
        job_id=job_id,
        device=str(device),
        trainable_params=sum(p.numel() for p in params),
        train_tokens=sum(len(s) for s in train),
        eval_tokens=sum(len(s) for s in held),
    )
    opt = torch.optim.AdamW(params, lr=hyper.lr, weight_decay=hyper.weight_decay)
    if held:
        ev.emit("eval", step=0, **evaluate())
    tokens_seen, t0 = 0, time.time()
    for step in range(1, hyper.steps + 1):
        lr = lr_at(step - 1, hyper)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum, hits, count = 0.0, 0, 0
        for _ in range(hyper.batch_seqs):
            ids = window(rng.choice(train))
            loss, m = step_on(ids)
            (loss / hyper.batch_seqs).backward()
            loss_sum, hits, count = loss_sum + float(loss.detach()), hits + m["hits"], count + m["count"]
            tokens_seen += len(ids)
        if hyper.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, hyper.grad_clip)
        opt.step()
        if step % hyper.log_every == 0 or step == hyper.steps:
            ev.emit(
                "progress",
                step=step,
                steps=hyper.steps,
                tokens=tokens_seen,
                loss=loss_sum / hyper.batch_seqs,
                accuracy=hits / max(count, 1),
                lr=lr,
                tokens_per_s=tokens_seen / max(time.time() - t0, 1e-9),
            )
        if held and (step % hyper.eval_every == 0 or step == hyper.steps):
            ev.emit("eval", step=step, **evaluate())

    out = out_dir / "dspark-head"
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "modelbuilder": "dspark"})
    (out / "config.json").write_text(json.dumps(cfg.hf_config(target_cfg), indent=2) + "\n")
    ev.emit("checkpoint", step=hyper.steps, path=str(out / "model.safetensors"))
    return {"draft_head": str(out)}

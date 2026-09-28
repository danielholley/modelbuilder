"""The ``trunk_distill`` stage: retrain some trunk tensors so a modified model matches the original.

The model is an HF checkpoint (e.g. from ``modelbuilder export-hf``). The
student is the same model with its modifications on:

- a quantized KV cache (``kv_format``: ``q8_0``, ``q4_0``, ``nvfp4``);
- cross-layer KV sharing (``kv_share_group``: consecutive attention layers
  share the first one's cache);
- trainable weights fake-quantized to their source format
  (``weight_fakequant``, from ``modelbuilder_quantization.json``), so the
  result re-encodes losslessly with ``modelbuilder surgery replace``.

The teacher is the same weights with everything switched off: the original
values of the trainable tensors and a plain cache. So no second copy of the
model is held. The loss is KL(teacher ‖ student) on next-token distributions,
computed in chunks of positions so a large vocabulary never materializes all
at once. Only tensors whose names contain one of ``trainable`` are updated.
The output ``updates/`` directory has those tensors (HF names, float32) plus
the config and the source-quantization manifest.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from modelbuilder_train import dist
from modelbuilder_train.events import EventWriter
from modelbuilder_train.hf import attention, fakequant
from modelbuilder_train.mtp.train import lr_at, supports_bf16
from modelbuilder_train.spec import Hyper, TrunkDistill

MANIFEST = "modelbuilder_quantization.json"
LOGIT_CHUNK = 256


class Mode:
    """Shared switch: teacher (original weights, plain cache) or student."""

    def __init__(self) -> None:
        self.student = True


class Switch(nn.Module):
    """A parametrization: the original value for the teacher, the (fake-quantized) trained value for the student."""

    def __init__(self, original: torch.Tensor, mode: Mode, fmt: fakequant.WeightFormat | None) -> None:
        super().__init__()
        self.register_buffer("original", original.detach().clone(), persistent=False)
        self.mode = mode
        self.fmt = fmt

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.mode.student:
            return self.original
        return self.fmt(w) if self.fmt is not None else w


def load_sequences(path: Path, tokenizer_dir: Path) -> list[list[int]]:
    """JSONL lines with ``tokens`` (ids) or ``text`` (tokenized with the model's tokenizer)."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if all("tokens" in x for x in lines):
        return [x["tokens"] for x in lines]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return [tok(x["text"], add_special_tokens=True).input_ids for x in lines]


def kl_and_agreement(student_h, teacher_h, lm_head: nn.Module) -> tuple[torch.Tensor, int, int]:
    """Summed KL(teacher ‖ student) over positions, top-1 agreements, positions."""
    total = student_h.new_zeros((), dtype=torch.float32)
    agree = 0
    n = student_h.shape[0]
    for a in range(0, n, LOGIT_CHUNK):
        s = lm_head(student_h[a : a + LOGIT_CHUNK]).float()
        with torch.no_grad():
            t = lm_head(teacher_h[a : a + LOGIT_CHUNK]).float()
        tp = F.log_softmax(t, -1)
        total = total + F.kl_div(F.log_softmax(s, -1), tp, log_target=True, reduction="sum")
        agree += int((s.argmax(-1) == t.argmax(-1)).sum())
    return total, agree, n


def run_trunk_distill(
    stage: TrunkDistill, hyper: Hyper, resolve, out_dir: Path, device: torch.device, ev: EventWriter, job_id: str
) -> dict:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(hyper.seed)
    rng = random.Random(hyper.seed + 1_000_003 * dist.rank())
    model_dir = resolve(stage.model)
    use_bf16 = hyper.dtype == "bfloat16" and supports_bf16(device)
    if hyper.dtype == "bfloat16" and not use_bf16:
        print(f"{device} has no fast bf16: training in float32", file=sys.stderr)
    attention.register()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16 if use_bf16 else torch.float32, attn_implementation=attention.NAME
    ).to(device)
    model.train()

    # What changes in the student.
    behavior = attention.CacheBehavior(kv_format=stage.kv_format)
    layers = attention.attach(model, behavior)
    if stage.kv_share_group:
        behavior.share = attention.share_groups(layers, stage.kv_share_group)
    formats = {}
    if stage.weight_fakequant and (model_dir / MANIFEST).exists():
        formats = fakequant.load_manifest(model_dir / MANIFEST)

    mode = Mode()
    trainable: dict[str, nn.Parameter] = {}
    model.requires_grad_(False)
    for mod_name, mod in list(model.named_modules()):
        for pname, p in list(mod.named_parameters(recurse=False)):
            full = f"{mod_name}.{pname}" if mod_name else pname
            if not any(pat in full for pat in stage.trainable):
                continue
            fmt = formats.get(full)
            if fmt is not None:
                fmt = fakequant.resolve_signs(fmt, p.shape[-1], full)
            parametrize.register_parametrization(mod, pname, Switch(p, mode, fmt))
            orig = getattr(mod.parametrizations, pname).original
            orig.requires_grad_(True)
            trainable[full] = orig
    if not trainable:
        raise ValueError(f"no parameter name contains any of {stage.trainable}")

    seqs = [s for s in load_sequences(resolve(stage.texts), model_dir) if len(s) >= 4]
    if not seqs:
        raise ValueError("no training sequence with at least 4 tokens")
    cut = max(1, len(seqs) - max(1, round(stage.eval_fraction * len(seqs)))) if stage.eval_fraction > 0 else len(seqs)
    train, held = seqs[:cut], seqs[cut:]

    decoder = model.get_decoder()
    lm_head = model.get_output_embeddings()

    def pass_(ids: torch.Tensor, student: bool) -> torch.Tensor:
        mode.student = student
        behavior.enabled = student
        behavior.reset()
        return decoder(input_ids=ids[None]).last_hidden_state[0]

    def window(seq: list[int]) -> torch.Tensor:
        n = min(len(seq), hyper.seq_len)
        a = rng.randrange(0, len(seq) - n + 1)
        return torch.tensor(seq[a : a + n], device=device)

    def evaluate() -> dict:
        kl_sum, agree, count = 0.0, 0, 0
        with torch.no_grad():
            for seq in held:
                ids = torch.tensor(seq[: hyper.seq_len], device=device)
                t = pass_(ids, False)
                s = pass_(ids, True)
                k, a, n = kl_and_agreement(s, t, lm_head)
                kl_sum, agree, count = kl_sum + float(k), agree + a, count + n
        return {"loss": kl_sum / max(count, 1), "accuracy": agree / max(count, 1), "tokens": count}

    n_trainable = sum(p.numel() for p in trainable.values())
    ev.emit(
        "started",
        job_id=job_id,
        device=str(device),
        trainable_params=n_trainable,
        train_tokens=sum(len(s) for s in train),
        eval_tokens=sum(len(s) for s in held),
    )
    opt = torch.optim.AdamW(list(trainable.values()), lr=hyper.lr, weight_decay=hyper.weight_decay)
    if held and dist.is_main():
        ev.emit("eval", step=0, **evaluate())
    tokens_seen, t0 = 0, time.time()
    for step in range(1, hyper.steps + 1):
        lr = lr_at(step - 1, hyper)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        batch = [window(rng.choice(train)) for _ in range(hyper.batch_seqs)]
        n_targets = sum(len(b) for b in batch)
        kl_total, agree_total = 0.0, 0
        for ids in batch:
            with torch.no_grad():
                t = pass_(ids, False)
            s = pass_(ids, True)
            k, a, _ = kl_and_agreement(s, t, lm_head)
            (k / n_targets).backward()
            kl_total, agree_total = kl_total + float(k.detach()), agree_total + a
        if dist.world() > 1:
            for p in trainable.values():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad)
                    p.grad /= dist.world()
        if hyper.grad_clip:
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), hyper.grad_clip)
        opt.step()
        tokens_seen += n_targets * dist.world()
        if dist.is_main() and (step % hyper.log_every == 0 or step == hyper.steps):
            ev.emit(
                "progress",
                step=step,
                steps=hyper.steps,
                tokens=tokens_seen,
                loss=kl_total / n_targets,
                accuracy=agree_total / n_targets,
                lr=lr,
                tokens_per_s=tokens_seen / max(time.time() - t0, 1e-9),
            )
        if held and dist.is_main() and (step % hyper.eval_every == 0 or step == hyper.steps):
            ev.emit("eval", step=step, **evaluate())

    out = out_dir / "updates"
    if dist.is_main():
        from safetensors.torch import save_file

        out.mkdir(parents=True, exist_ok=True)
        mode.student = True
        tensors = {}
        for full in trainable:
            mod_name, _, pname = full.rpartition(".")
            mod = model.get_submodule(mod_name) if mod_name else model
            tensors[full] = getattr(mod, pname).detach().float().contiguous()  # the student's (fake-quantized) value
        save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "modelbuilder": "trunk-distill"})
        for f in ("config.json", MANIFEST):
            if (model_dir / f).exists():
                shutil.copyfile(model_dir / f, out / f)
        ev.emit("checkpoint", step=hyper.steps, path=str(out / "model.safetensors"))
    if not math.isfinite(kl_total):
        raise RuntimeError("the loss is not finite")
    return {"updates": str(out)}

"""The ``trunk_distill`` stage: retrain some trunk tensors so a modified model matches the original.

The model is an HF checkpoint (e.g. from ``modelbuilder export-hf``). The
student is the same model with its modifications on:

- a quantized KV cache (``kv_format``: ``q8_0``, ``q4_0``, ``nvfp4``);
- cross-layer KV sharing (``kv_share_group``: consecutive attention layers
  share the first one's cache);
- trainable weights fake-quantized to their source format
  (``weight_fakequant``, from ``modelbuilder_quantization.json``), so the
  result re-encodes losslessly with ``modelbuilder surgery replace``;
- multi-head latent attention, when the checkpoint comes from
  ``modelbuilder surgery mla`` (``modelbuilder_mla`` in its config): the
  student rebuilds K/V from the latent with partial RoPE (``hf/mla.py``).

The teacher is the same weights with everything switched off: the original
values of the trainable tensors and a plain cache. So no second copy of the
model is held. A separate ``teacher`` checkpoint (e.g. the unpruned original
when healing a pruned model) replaces that, at the cost of a second model in
memory. With ``loss = "ce"`` there is no teacher: the trainable tensors learn
next-token cross-entropy on the texts (e.g. a long-context finetune).

Otherwise the loss is KL(teacher ‖ student) on next-token distributions,
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
from modelbuilder_train.hf import attention, fakequant, mla
from modelbuilder_train.mtp.train import lr_at, supports_bf16
from modelbuilder_train.spec import Hyper, TrunkDistill

MANIFEST = "modelbuilder_quantization.json"
LOGIT_CHUNK = 256


class Mode:
    """Shared switch: teacher (original weights, plain cache) or student."""

    def __init__(self) -> None:
        self.student = True


class Switch(nn.Module):
    """A parametrization: the original value for the teacher, the (fake-quantized) trained value for the student.

    With ``lora_rank`` the stored weight stays frozen (so it is also the
    teacher's value and no copy is kept) and the student sees
    ``fmt(w + scale · B·A)`` with trainable ``A`` [r, in] and ``B`` [out, r]
    (``B = 0`` at the start, as in LoRA, Hu et al., 2021). Merged and
    fake-quantized, that is exactly the tensor the write-back stores.
    """

    def __init__(
        self,
        original: torch.Tensor,
        mode: Mode,
        fmt: fakequant.WeightFormat | None,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fmt = fmt
        self.lora = lora_rank is not None and original.ndim == 2
        if self.lora:
            out_f, in_f = original.shape
            a = torch.empty(lora_rank, in_f, dtype=original.dtype, device=original.device)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_a = nn.Parameter(a)
            self.lora_b = nn.Parameter(torch.zeros(out_f, lora_rank, dtype=original.dtype, device=original.device))
            self.scale = (lora_alpha if lora_alpha is not None else lora_rank) / lora_rank
        else:
            self.register_buffer("original", original.detach().clone(), persistent=False)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.lora:
            if not self.mode.student:
                return w
            w = w + self.scale * (self.lora_b @ self.lora_a)
        elif not self.mode.student:
            return self.original
        return self.fmt(w) if self.fmt is not None else w


def checkpoint_names(model: nn.Module, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Parameters under their checkpoint names: transformers ≥ 5 fuses some on load
    (e.g. MoE experts into ``experts.gate_up_proj``) and this splits them back."""
    try:
        from transformers.core_model_loading import revert_weight_conversion
    except ImportError:  # transformers 4: parameters are stored as they are named
        return tensors
    return revert_weight_conversion(model, dict(tensors))


def load_sequences(path: Path, tokenizer_dir: Path) -> list[list[int]]:
    """JSONL lines with ``tokens`` (ids) or ``text`` (tokenized with the model's tokenizer)."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if all("tokens" in x for x in lines):
        return [x["tokens"] for x in lines]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return [tok(x["text"], add_special_tokens=True).input_ids for x in lines]


def ce_and_accuracy(student_h, targets: torch.Tensor, lm_head: nn.Module) -> tuple[torch.Tensor, int, int]:
    """Summed next-token cross-entropy, top-1 hits, positions (``student_h[i]`` predicts ``targets[i]``)."""
    total = student_h.new_zeros((), dtype=torch.float32)
    hits = 0
    n = student_h.shape[0]
    for a in range(0, n, LOGIT_CHUNK):
        s = lm_head(student_h[a : a + LOGIT_CHUNK]).float()
        t = targets[a : a + LOGIT_CHUNK]
        total = total + F.cross_entropy(s, t, reduction="sum")
        hits += int((s.argmax(-1) == t).sum())
    return total, hits, n


def kl_and_agreement(
    student_h, teacher_h, lm_head: nn.Module, teacher_head: nn.Module | None = None
) -> tuple[torch.Tensor, int, int]:
    """Summed KL(teacher ‖ student) over positions, top-1 agreements, positions."""
    teacher_head = teacher_head or lm_head
    total = student_h.new_zeros((), dtype=torch.float32)
    agree = 0
    n = student_h.shape[0]
    for a in range(0, n, LOGIT_CHUNK):
        s = lm_head(student_h[a : a + LOGIT_CHUNK]).float()
        with torch.no_grad():
            t = teacher_head(teacher_h[a : a + LOGIT_CHUNK]).float().to(s.device)
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
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    def load(path: Path, **kw):
        """On ``device``, or with ``device_map`` spread layer by layer over this process's GPUs."""
        if stage.device_map is None:
            return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kw).to(device)
        if dist.world() > 1:
            raise ValueError("device_map splits one model over a process's GPUs; run one process, not torchrun")
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=stage.device_map, **kw)

    model = load(model_dir, attn_implementation=attention.NAME)
    if stage.device_map is not None:
        device = model.get_input_embeddings().weight.device  # where the token ids go
    model.train()
    teacher = None
    if stage.teacher and stage.loss == "kl":
        teacher = load(resolve(stage.teacher))
        teacher.eval().requires_grad_(False)
        tv, sv = teacher.get_output_embeddings().weight.shape[0], model.get_output_embeddings().weight.shape[0]
        if tv != sv:
            raise ValueError(f"the teacher's vocabulary ({tv}) differs from the model's ({sv})")

    # What changes in the student.
    behavior = attention.CacheBehavior(kv_format=stage.kv_format)
    layers = attention.attach(model, behavior)
    if stage.kv_share_group:
        behavior.share = attention.share_groups(layers, stage.kv_share_group)
    formats = {}
    if stage.weight_fakequant and (model_dir / MANIFEST).exists():
        formats = fakequant.load_manifest(model_dir / MANIFEST)

    mode = Mode()
    mla_cfg = mla.load_config(model_dir)
    if mla_cfg is not None:
        mla.attach(model, model_dir, mla_cfg, mode)
    trainable: dict[str, nn.Parameter] = {}
    lora_params: list[nn.Parameter] = []
    model.requires_grad_(False)
    for mod_name, mod in list(model.named_modules()):
        for pname, p in list(mod.named_parameters(recurse=False)):
            full = f"{mod_name}.{pname}" if mod_name else pname
            if not any(pat in full for pat in stage.trainable):
                continue
            fmt = formats.get(full)
            if fmt is None and formats:
                # A fused parameter (e.g. stacked experts): the format of the tensors it was built from.
                names = list(checkpoint_names(model, {full: p.detach()}))
                found = {(formats[n].dtype, formats[n].rotated) for n in names if n in formats}
                if names != [full] and len(found) == 1 and all(n in formats for n in names):
                    fmt = formats[names[0]]
            if fmt is not None:
                fmt = fakequant.resolve_signs(fmt, p.shape[-1], full)
            sw = Switch(p, mode, fmt, stage.lora_rank, stage.lora_alpha)
            parametrize.register_parametrization(mod, pname, sw)
            plist = getattr(mod.parametrizations, pname)
            if sw.lora:
                plist[0].lora_a.requires_grad_(True)
                plist[0].lora_b.requires_grad_(True)
                lora_params += [plist[0].lora_a, plist[0].lora_b]
            else:
                plist.original.requires_grad_(True)
            trainable[full] = plist.original
    if not trainable:
        raise ValueError(f"no parameter name contains any of {stage.trainable}")
    # What the optimizer updates: the tensors themselves, or their LoRA factors.
    params = lora_params + [p for p in trainable.values() if p.requires_grad]

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
        if not student and teacher is not None:
            tdev = teacher.get_input_embeddings().weight.device
            return teacher.get_decoder()(input_ids=ids[None].to(tdev)).last_hidden_state[0]
        return decoder(input_ids=ids[None]).last_hidden_state[0]

    teacher_head = teacher.get_output_embeddings() if teacher is not None else None

    def losses(ids: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Summed loss, top-1 agreements (or hits), positions."""
        if stage.loss == "ce":
            s = pass_(ids, True)
            return ce_and_accuracy(s[:-1], ids[1:].to(s.device), lm_head)
        with torch.no_grad():
            t = pass_(ids, False)
        return kl_and_agreement(pass_(ids, True), t, lm_head, teacher_head)

    def window(seq: list[int]) -> torch.Tensor:
        n = min(len(seq), hyper.seq_len)
        a = rng.randrange(0, len(seq) - n + 1)
        return torch.tensor(seq[a : a + n], device=device)

    def evaluate() -> dict:
        kl_sum, agree, count = 0.0, 0, 0
        with torch.no_grad():
            for seq in held:
                ids = torch.tensor(seq[: hyper.seq_len], device=device)
                k, a, n = losses(ids)
                kl_sum, agree, count = kl_sum + float(k), agree + a, count + n
        return {"loss": kl_sum / max(count, 1), "accuracy": agree / max(count, 1), "tokens": count}

    n_trainable = sum(p.numel() for p in params)
    ev.emit(
        "started",
        job_id=job_id,
        device=str(device),
        trainable_params=n_trainable,
        train_tokens=sum(len(s) for s in train),
        eval_tokens=sum(len(s) for s in held),
    )
    opt = torch.optim.AdamW(params, lr=hyper.lr, weight_decay=hyper.weight_decay)
    if held and dist.is_main():
        ev.emit("eval", step=0, **evaluate())
    tokens_seen, t0 = 0, time.time()
    for step in range(1, hyper.steps + 1):
        lr = lr_at(step - 1, hyper)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        batch = [window(rng.choice(train)) for _ in range(hyper.batch_seqs)]
        n_targets = sum(len(b) - (stage.loss == "ce") for b in batch)
        kl_total, agree_total = 0.0, 0
        for ids in batch:
            k, a, _ = losses(ids)
            (k / n_targets).backward()
            kl_total, agree_total = kl_total + float(k.detach()), agree_total + a
        if dist.world() > 1:
            for p in params:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad)
                    p.grad /= dist.world()
        if hyper.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, hyper.grad_clip)
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
        tensors = {k: v.contiguous() for k, v in checkpoint_names(model, tensors).items()}
        save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "modelbuilder": "trunk-distill"})
        for f in ("config.json", MANIFEST):
            if (model_dir / f).exists():
                shutil.copyfile(model_dir / f, out / f)
        ev.emit("checkpoint", step=hyper.steps, path=str(out / "model.safetensors"))
    if not math.isfinite(kl_total):
        raise RuntimeError("the loss is not finite")
    return {"updates": str(out)}

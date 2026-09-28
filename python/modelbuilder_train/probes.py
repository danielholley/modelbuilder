"""Behavioral probes: measure what a change costs in quality, for any model.

- :func:`perplexity_llamacpp`: ``llama-perplexity`` on a text file, optionally
  with a quantized KV cache (``-ctk/-ctv``). Works for any GGUF.
- :func:`needle`: long-context retrieval through ``llama-server``. A fact is
  hidden at several depths of a filler context, the model is asked for it, and
  the answer is checked. Also any GGUF, and any KV cache type.
- :func:`kv_cache_sweep`: both of the above for each KV cache type, against
  the unquantized cache. This is the measurement that decides whether a
  quantized KV cache needs QAT at all.
- :func:`perplexity_hf`: the same perplexity for an HF checkpoint in PyTorch
  (e.g. before re-encoding a QAT result), with a strided sliding window.

Every result records its settings, so reports can be compared later.
"""

from __future__ import annotations

import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

_PPL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


@dataclass
class PerplexityResult:
    model: str
    text: str
    ctx: int
    cache_type_k: str
    cache_type_v: str
    ppl: float
    ppl_err: float


def perplexity_llamacpp(
    llama_bin: Path,
    model: Path,
    text: Path,
    *,
    ctx: int = 2048,
    chunks: int | None = None,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    threads: int | None = None,
    gpu_layers: int | None = None,
    extra: list[str] | None = None,
) -> PerplexityResult:
    cmd = [str(Path(llama_bin) / "llama-perplexity"), "-m", str(model), "-f", str(text), "-c", str(ctx)]
    cmd += ["-ctk", cache_type_k, "-ctv", cache_type_v]
    if cache_type_v not in ("f16", "f32", "bf16"):
        cmd += ["-fa", "on"]  # llama.cpp needs flash attention for a quantized V cache
    if chunks:
        cmd += ["--chunks", str(chunks)]
    if threads:
        cmd += ["-t", str(threads)]
    if gpu_layers is not None:
        cmd += ["-ngl", str(gpu_layers)]
    cmd += extra or []
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = _PPL.search(r.stdout + r.stderr)
    if r.returncode != 0 or not m:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f"llama-perplexity failed (exit {r.returncode}):\n{tail}")
    return PerplexityResult(str(model), str(text), ctx, cache_type_k, cache_type_v, float(m[1]), float(m[2]))


FILLER = (
    "The committee reviewed the quarterly figures and noted that shipments had risen in the northern region "
    "while costs in the southern warehouses stayed flat. Several members asked for a breakdown by product line. "
    "The chair proposed revisiting the question at the next meeting, after the auditors had finished their review. "
)


@dataclass
class NeedleTrial:
    context_tokens: int
    depth: float
    found: bool
    answer: str


@dataclass
class NeedleResult:
    model: str
    cache_type_k: str
    cache_type_v: str
    trials: list[NeedleTrial] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(t.found for t in self.trials) / max(len(self.trials), 1)


def needle(
    server,
    *,
    lengths: list[int],
    depths: list[float],
    filler: str = FILLER,
    seed: int = 0,
    model: str = "",
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    progress=None,
) -> NeedleResult:
    """Hides ``The secret code is NNNNNN.`` at each depth of each context length and asks for it.

    ``server`` is a :class:`modelbuilder_train.server.Server`, started with a
    context of at least ``max(lengths)`` + 64 tokens per slot. Lengths are in
    tokens of the server's own tokenizer; the chat template is applied.
    """
    rng = random.Random(seed)
    unit = server.tokenize(filler)
    res = NeedleResult(model, cache_type_k, cache_type_v)
    for n in lengths:
        reps = max(1, n // max(len(unit), 1))
        paragraphs = [filler] * reps
        for d in depths:
            code = f"{rng.randrange(100000, 999999)}"
            doc = paragraphs[:]
            doc.insert(round(d * len(doc)), f"The secret code is {code}. ")
            question = "What is the secret code mentioned in the text? Answer with the number only."
            prompt = server.apply_template([{"role": "user", "content": "".join(doc) + "\n\n" + question}])
            answer = server.complete(prompt, n_predict=24, temperature=0.0, seed=seed)
            trial = NeedleTrial(n, d, code in answer, answer.strip()[:80])
            res.trials.append(trial)
            if progress:
                progress(trial)
    return res


def kv_cache_sweep(
    llama_bin: Path,
    model: Path,
    *,
    types: list[str],
    text: Path | None = None,
    ctx: int = 2048,
    chunks: int | None = 20,
    needle_lengths: list[int] | None = None,
    needle_depths: list[float] | None = None,
    threads: int | None = None,
    gpu_layers: int | None = None,
    progress=None,
) -> dict:
    """Perplexity and retrieval per KV cache type (K and V both set to it), against f16.

    Returns ``{"results": [...], "summary": [{"type", "ppl", "ppl_delta_pct", "needle_accuracy"}]}``.
    """
    from modelbuilder_train.server import Server

    rows = []
    base_ppl = None
    for t in ["f16", *[t for t in types if t != "f16"]]:
        row: dict = {"type": t}
        if text is not None:
            p = perplexity_llamacpp(
                llama_bin,
                model,
                text,
                ctx=ctx,
                chunks=chunks,
                cache_type_k=t,
                cache_type_v=t,
                threads=threads,
                gpu_layers=gpu_layers,
            )
            row["perplexity"] = asdict(p)
            row["ppl"] = p.ppl
            base_ppl = base_ppl or p.ppl
            row["ppl_delta_pct"] = 100.0 * (p.ppl / base_ppl - 1.0)
        if needle_lengths:
            extra = ["-ctk", t, "-ctv", t] + (["-fa", "on"] if t not in ("f16", "f32", "bf16") else [])
            with Server.launch(
                llama_bin,
                model,
                port=8093,
                parallel=1,
                ctx=max(needle_lengths) + 256,
                threads=threads,
                gpu_layers=gpu_layers,
                extra=extra,
            ) as s:
                nr = needle(
                    s,
                    lengths=needle_lengths,
                    depths=needle_depths or [0.1, 0.5, 0.9],
                    model=str(model),
                    cache_type_k=t,
                    cache_type_v=t,
                )
            row["needle"] = {"trials": [asdict(x) for x in nr.trials]}
            row["needle_accuracy"] = nr.accuracy
        rows.append(row)
        if progress:
            progress(row)
    summary = [{k: r.get(k) for k in ("type", "ppl", "ppl_delta_pct", "needle_accuracy")} for r in rows]
    return {"model": str(model), "results": rows, "summary": summary}


def perplexity_hf(
    model_dir: Path, text: str, *, ctx: int = 2048, stride: int | None = None, device: str = "auto", dtype=None
) -> dict:
    """Strided sliding-window perplexity of an HF causal LM on ``text``."""
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from modelbuilder_train.mtp.train import pick_device

    dev = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype or torch.float32).to(dev).eval()
    ids = tok(text, return_tensors="pt").input_ids[0]
    stride = stride or ctx // 2
    nll, count, prev_end = 0.0, 0, 0
    with torch.no_grad():
        for begin in range(0, len(ids), stride):
            end = min(begin + ctx, len(ids))
            window = ids[begin:end].to(dev)[None]
            target_len = end - prev_end  # only score tokens not scored before
            labels = window.clone()
            labels[:, :-target_len] = -100
            out = model(window, labels=labels)
            n = int((labels[:, 1:] != -100).sum())
            nll += float(out.loss) * n
            count += n
            prev_end = end
            if end == len(ids):
                break
    return {
        "model": str(model_dir),
        "ctx": ctx,
        "stride": stride,
        "tokens": count,
        "ppl": math.exp(nll / max(count, 1)),
    }


@dataclass
class HiddenMatch:
    tokens: int
    #: mean and worst per-token cosine similarity of the final (post-norm) hidden states
    cosine_mean: float
    cosine_min: float
    #: ‖hf − gguf‖ / ‖gguf‖ over all tokens
    rel_rms: float
    #: fraction of tokens whose next-token argmax (through the HF LM head) agrees
    top1_agreement: float


def hf_vs_gguf(server, hf_dir: Path, texts: list[str], max_tokens: int = 512, device: str = "auto") -> HiddenMatch:
    """Checks an HF export against the GGUF it came from, token for token.

    ``server`` runs the GGUF with ``--embeddings --pooling none`` (see
    ``Server.launch(embeddings=True)``), which returns the final post-norm
    hidden state per token. The same token ids go through the HF model; both
    hidden states are also pushed through the HF LM head to compare argmaxes.
    Use a pruned GGUF (``modelbuilder surgery prune``) to check a big model on
    a small machine.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from modelbuilder_train.mtp.train import pick_device

    dev = pick_device(device)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).to(dev).eval()
    head = model.get_output_embeddings()
    cos_all, err, norm, agree, n = [], 0.0, 0.0, 0, 0
    with torch.no_grad():
        for text in texts:
            ids = server.tokenize(text)[:max_tokens]
            if len(ids) < 2:
                continue
            ref = server.hidden_states(ids).to(dev)
            got = model.get_decoder()(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0].float()
            cos_all.append(torch.nn.functional.cosine_similarity(got, ref, dim=-1))
            err += float((got - ref).pow(2).sum())
            norm += float(ref.pow(2).sum())
            agree += int((head(got).argmax(-1) == head(ref).argmax(-1)).sum())
            n += len(ids)
    if not n:
        raise ValueError("no text with at least 2 tokens")
    cos = torch.cat(cos_all)
    return HiddenMatch(n, float(cos.mean()), float(cos.min()), (err / max(norm, 1e-30)) ** 0.5, agree / n)

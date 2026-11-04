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
- :func:`skills`: a short auto-graded suite of code, reasoning, tool-use and
  language items through ``llama-server``, per category; a smoke test of what
  a model can do and what a change breaks, not a benchmark.

Every result records its settings, so reports can be compared later.
"""

from __future__ import annotations

import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

_PPL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def strip_thinking(answer: str) -> str:
    """Drops a reasoning block (``<think>...</think>``, or an unfinished one) from a chat answer."""
    return _THINK.sub("", answer).strip()


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
            prompt = server.apply_template(
                [{"role": "user", "content": "".join(doc) + "\n\n" + question}], thinking=False
            )
            answer = strip_thinking(server.complete(prompt, n_predict=48, temperature=0.0, seed=seed))
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


def hf_vs_gguf(
    server,
    hf_dir: Path,
    texts: list[str],
    max_tokens: int = 512,
    device: str = "auto",
    dtype: str = "float32",
    kv_share_group: int | None = None,
) -> HiddenMatch:
    """Checks an HF export against the GGUF it came from, token for token.

    ``server`` runs the GGUF with ``--embeddings --pooling none`` (see
    ``Server.launch(embeddings=True)``), which returns the final post-norm
    hidden state per token. The same token ids go through the HF model; both
    hidden states are also pushed through the HF LM head to compare argmaxes.
    Use a pruned GGUF (``modelbuilder surgery prune``) to check a big model on
    a small machine, and ``dtype="bfloat16"`` to halve the HF model's memory
    (a large vocabulary's embedding and head dominate it); the comparison is
    still in float32, at bf16 precision. With ``kv_share_group`` the HF model
    shares K/V across groups of attention layers as ``trunk_distill`` trains it,
    to check a GGUF marked by ``modelbuilder surgery kv-share`` in a runtime
    that implements the sharing. An ``hf_dir`` from ``modelbuilder surgery mla``
    runs as its latent-attention student, to check ``surgery mla-gguf`` output.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from modelbuilder_train.mtp.train import pick_device

    dev = pick_device(device)
    behavior = None
    if kv_share_group:
        from modelbuilder_train.hf import attention

        attention.register()
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir, dtype=getattr(torch, dtype), attn_implementation=attention.NAME
        ).to(dev)
        behavior = attention.CacheBehavior()
        behavior.share = attention.share_groups(attention.attach(model, behavior), kv_share_group)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=getattr(torch, dtype)).to(dev)
    model.eval()
    from modelbuilder_train.hf import mla

    mla_cfg = mla.load_config(hf_dir)
    if mla_cfg is not None:  # an MLA conversion: compare the latent-attention model it describes

        class _Student:
            student = True

        mla.attach(model, hf_dir, mla_cfg, _Student())
    head = model.get_output_embeddings()
    cos_all, err, norm, agree, n = [], 0.0, 0.0, 0, 0
    with torch.no_grad():
        for text in texts:
            ids = server.tokenize(text)[:max_tokens]
            if len(ids) < 2:
                continue
            ref = server.hidden_states(ids).to(dev)
            if behavior is not None:
                behavior.reset()
            got = model.get_decoder()(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0].float()
            cos_all.append(torch.nn.functional.cosine_similarity(got, ref, dim=-1))
            err += float((got - ref).pow(2).sum())
            norm += float(ref.pow(2).sum())
            hd = head.weight.dtype
            agree += int((head(got.to(hd)).argmax(-1) == head(ref.to(hd)).argmax(-1)).sum())
            n += len(ids)
    if not n:
        raise ValueError("no text with at least 2 tokens")
    cos = torch.cat(cos_all)
    return HiddenMatch(n, float(cos.mean()), float(cos.min()), (err / max(norm, 1e-30)) ** 0.5, agree / n)


# ---------------------------------------------------------------------------
# Skills: a short, auto-graded suite per category (code, reasoning, tool use,
# languages). A smoke test of what a model can do and of what a change breaks,
# not a benchmark: compare the same suite before and after.

TOOLS_PROMPT = (
    "You can call these tools. To call one, reply with only a JSON object "
    '{"name": <tool name>, "arguments": {<argument>: <value>}} and nothing else.\n'
    "- get_weather(city: string): current weather in a city\n"
    "- convert_currency(amount: number, source: string, target: string): convert money, ISO codes\n"
    "- search(query: string): web search"
)

_PY = "What does this Python print? Answer with the output only.\n\n"
_JS = "What does this JavaScript log? Answer with the output only.\n\n"
_NUM = " Answer with the number only."
_TR = "Answer with the translation only."

#: (category, prompt, grader, expected). Graders: ``exact`` (the answer appears
#: as a whole token), ``any`` (any of the answers appears), ``call`` (a JSON tool
#: call with this name and at least these arguments).
SKILLS: list[tuple[str, str, str, object]] = [
    ("code", _PY + "print(sum(range(5)))", "exact", "10"),
    ("code", _PY + "print(len('hello big world'.split()))", "exact", "3"),
    ("code", _PY + "print('ab' * 3)", "exact", "ababab"),
    ("code", _PY + "def f(n):\n    return 1 if n < 2 else n * f(n - 1)\nprint(f(5))", "exact", "120"),
    ("code", _PY + "x = [3, 1, 2]\nx.sort()\nprint(x[-1])", "exact", "3"),
    ("code", _JS + "console.log([1, 2, 3].map(x => x * 2).join('-'))", "exact", "2-4-6"),
    ("reasoning", "What is 17 * 23?" + _NUM, "exact", "391"),
    ("reasoning", "How many days are in a leap year?" + _NUM, "exact", "366"),
    ("reasoning", "What comes next: 2, 4, 8, 16, ?" + _NUM, "exact", "32"),
    ("reasoning", "A shop sells pens at 3 for $2. How many dollars do 12 pens cost?" + _NUM, "exact", "8"),
    ("reasoning", "Tom is taller than Ann. Ann is taller than Bo. Who is the shortest? Answer with the name only.", "exact", "Bo"),  # noqa: E501
    ("tool_use", TOOLS_PROMPT + "\n\nWhat's the weather like in Paris right now?", "call", {"name": "get_weather", "arguments": {"city": "Paris"}}),  # noqa: E501
    ("tool_use", TOOLS_PROMPT + "\n\nHow much is 100 US dollars in euros?", "call", {"name": "convert_currency", "arguments": {"source": "USD", "target": "EUR"}}),  # noqa: E501
    ("tool_use", TOOLS_PROMPT + "\n\nLook up who won the 2018 football World Cup.", "call", {"name": "search", "arguments": {}}),  # noqa: E501
    ("languages", "Translate into French: 'thank you'. " + _TR, "any", ["merci"]),
    ("languages", "Translate into Spanish: 'good morning'. " + _TR, "any", ["buenos días", "buenos dias"]),
    ("languages", "Translate into German: 'the cat'. " + _TR, "any", ["die katze"]),
    ("languages", "Translate into Italian: 'water'. " + _TR, "any", ["acqua"]),
    ("languages", "Translate into Japanese: 'water'. " + _TR, "any", ["水", "みず"]),
    ("languages", "Translate into Chinese: 'hello'. " + _TR, "any", ["你好", "您好"]),
]  # fmt: skip


def _first_json(text: str) -> dict | None:
    import json

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return None


def grade(grader: str, expected, answer: str) -> bool:
    a = answer.strip().strip("`").strip()
    low = a.lower()
    if grader == "exact":
        return re.search(rf"(?<![\w-]){re.escape(str(expected).lower())}(?![\w-])", low) is not None
    if grader == "any":
        return any(str(x).lower() in low for x in expected)
    if grader == "call":
        call = _first_json(a)
        if not call or call.get("name") != expected["name"]:
            return False
        args = call.get("arguments") or call.get("parameters") or {}
        return all(str(args.get(k, "")).strip().lower() == str(v).lower() for k, v in expected["arguments"].items())
    raise ValueError(f"unknown grader {grader}")


@dataclass
class SkillTrial:
    category: str
    passed: bool
    answer: str


@dataclass
class SkillsResult:
    model: str
    trials: list[SkillTrial] = field(default_factory=list)

    @property
    def by_category(self) -> dict[str, float]:
        cats: dict[str, list[bool]] = {}
        for t in self.trials:
            cats.setdefault(t.category, []).append(t.passed)
        return {c: sum(v) / len(v) for c, v in cats.items()}


def load_items(path: Path) -> list[tuple[str, str, str, object]]:
    """A custom suite: JSONL with ``category``, ``prompt``, ``grader`` and ``expected`` per line."""
    import json

    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            items.append((x["category"], x["prompt"], x["grader"], x["expected"]))
    return items


def skills(server, items=None, *, model: str = "", seed: int = 0, n_predict: int = 96, progress=None) -> SkillsResult:
    """Asks every item through the chat template at temperature 0 and grades the answer."""
    res = SkillsResult(model)
    for category, prompt, grader, expected in items or SKILLS:
        text = server.apply_template([{"role": "user", "content": prompt}], thinking=False)
        answer = strip_thinking(server.complete(text, n_predict=n_predict, temperature=0.0, seed=seed))
        trial = SkillTrial(category, grade(grader, expected, answer), answer.strip()[:120])
        res.trials.append(trial)
        if progress:
            progress(trial)
    return res

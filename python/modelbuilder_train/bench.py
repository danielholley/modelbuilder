"""Draft acceptance and decode speed of MTP sidecars, measured in the PrismML fork.

Runs ``llama-speculative-simple`` once per (prompt, variant) with greedy
decoding and parses its summary (``n_drafted``, ``n_accept``, decode speed).
The ``plain`` baseline uses ``--spec-type ngram-simple``, a no-op on this
model family, so it is the same binary and settings without drafting (see
``docs/research/mtp-port.md``). This is the number that decides whether an
aligned head beats the ported one.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

_PAT = {
    "decoded": re.compile(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+) t/s"),
    "drafted": re.compile(r"n_drafted\s*=\s*(\d+)"),
    "accept": re.compile(r"n_accept\s*=\s*(\d+)"),
}


@dataclass
class Result:
    prompt: int
    variant: str
    tokens: int
    seconds: float
    tokens_per_s: float
    drafted: int
    accepted: int

    @property
    def acceptance(self) -> float | None:
        return self.accepted / self.drafted if self.drafted else None


def parse(output: str) -> dict:
    d = _PAT["decoded"].findall(output)
    if not d:
        raise ValueError("no 'decoded … tokens' line in llama-speculative-simple's output")
    n, secs, tps = d[-1]
    drafted = _PAT["drafted"].findall(output)
    accepted = _PAT["accept"].findall(output)
    return {
        "tokens": int(n),
        "seconds": float(secs),
        "tokens_per_s": float(tps),
        "drafted": int(drafted[-1]) if drafted else 0,
        "accepted": int(accepted[-1]) if accepted else 0,
    }


def run(
    llama_bin: Path,
    model: Path,
    sidecars: dict[str, Path],
    prompts: list[str],
    *,
    n_predict: int = 128,
    ctx: int = 4096,
    threads: int | None = None,
    gpu_layers: int | None = None,
    extra: list[str] | None = None,
    baseline: bool = True,
    progress=None,
) -> list[Result]:
    exe = Path(llama_bin) / "llama-speculative-simple"
    variants: list[tuple[str, list[str]]] = [("plain", ["--spec-type", "ngram-simple"])] if baseline else []
    variants += [(name, ["-md", str(p), "--spec-type", "draft-mtp"]) for name, p in sidecars.items()]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, prompt in enumerate(prompts):
            pf = Path(tmp) / f"{i}.txt"
            pf.write_text(prompt)
            for name, args in variants:
                cmd = [
                    str(exe),
                    "-m",
                    str(model),
                    *args,
                    "-f",
                    str(pf),
                    "--no-escape",
                    "-n",
                    str(n_predict),
                    "--temp",
                    "0",
                    "-c",
                    str(ctx),
                ]
                if threads:
                    cmd += ["-t", str(threads)]
                if gpu_layers is not None:
                    cmd += ["-ngl", str(gpu_layers), "-ngld", str(gpu_layers)]
                cmd += extra or []
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    tail = "\n".join(r.stderr.strip().splitlines()[-10:])
                    raise RuntimeError(f"{name}, prompt {i}: exit code {r.returncode}\n{tail}")
                results.append(Result(prompt=i, variant=name, **parse(r.stdout + r.stderr)))
                if progress:
                    progress(results[-1])
    return results


def summarize(results: list[Result]) -> list[dict]:
    """Per variant: total acceptance, and decode speed relative to ``plain`` on the same prompts."""
    plain = {r.prompt: r.tokens_per_s for r in results if r.variant == "plain"}
    rows = []
    for v in dict.fromkeys(r.variant for r in results):
        rs = [r for r in results if r.variant == v]
        drafted, accepted = sum(r.drafted for r in rs), sum(r.accepted for r in rs)
        speedups = [r.tokens_per_s / plain[r.prompt] for r in rs if plain.get(r.prompt)]
        rows.append(
            {
                "variant": v,
                "prompts": len(rs),
                "accepted": accepted,
                "drafted": drafted,
                "acceptance": accepted / drafted if drafted else None,
                "mean_tokens_per_s": sum(r.tokens_per_s for r in rs) / len(rs),
                "mean_speedup": sum(speedups) / len(speedups) if speedups else None,
            }
        )
    return rows


def to_json(results: list[Result]) -> dict:
    return {"results": [asdict(r) | {"acceptance": r.acceptance} for r in results], "summary": summarize(results)}

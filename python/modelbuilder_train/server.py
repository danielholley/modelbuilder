"""Corpus generation and feature extraction through a long-running ``llama-server``.

One server per machine (or per GPU) keeps the model loaded, so work is not
dominated by reloading 7 GB of weights per text as ``extract_llamacpp`` is.
Both jobs split their input with ``shard=(i, n)`` (item ``k`` goes to shard
``k % n``), so a cluster can run shard ``i`` on machine ``i`` and write to a
shared array, and both resume where they stopped.

- :func:`generate_corpus` writes the model's own answers to prompts
  (self-distillation data for aligning a draft head): the server applies the
  chat template, and each output line is the full templated sequence.
- :func:`extract_features` stores, for every token of every text, the trunk's
  hidden state after its final norm. The server must run with
  ``--embeddings --pooling none``, which returns unnormalized per-token
  states. Texts are tokenized by the same server, and the token ids are sent
  back as the prompt, so positions line up exactly.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from modelbuilder_train.features import MANIFEST, FeatureWriter

try:  # optional: several times faster on the large embedding responses
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover - depends on the environment
    _loads = json.loads

# Local servers must not go through an HTTP proxy from the environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Server:
    """A client for one ``llama-server``; optionally starts it and stops it on exit."""

    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 3600.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None

    @classmethod
    def launch(
        cls,
        llama_bin: Path,
        model: Path,
        port: int = 8080,
        *,
        embeddings: bool = False,
        parallel: int = 1,
        ctx: int = 8192,
        threads: int | None = None,
        gpu_layers: int | None = None,
        extra: list[str] | None = None,
        log: Path | None = None,
    ) -> Server:
        cmd = [str(Path(llama_bin) / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", str(port)]
        # -c is the total context, shared by the parallel slots.
        cmd += ["-np", str(parallel), "-c", str(ctx * parallel)]
        if embeddings:
            # Whole sequences in one batch: the recurrent (DeltaNet) state needs them unsplit.
            cmd += ["--embeddings", "--pooling", "none", "-b", str(ctx), "-ub", str(ctx)]
        if threads:
            cmd += ["-t", str(threads)]
        if gpu_layers is not None:
            cmd += ["-ngl", str(gpu_layers)]
        cmd += extra or []
        out = open(log, "a") if log else subprocess.DEVNULL  # noqa: SIM115
        s = cls(f"http://127.0.0.1:{port}")
        s.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        s.wait_ready()
        return s

    def wait_ready(self, timeout: float = 900.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode} (see its log)")
            try:
                with _OPENER.open(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"{self.url} did not become ready in {timeout:.0f}s")

    def post(self, path: str, body: dict):
        req = urllib.request.Request(
            f"{self.url}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with _OPENER.open(req, timeout=self.timeout) as r:
            return _loads(r.read())

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- endpoints

    def tokenize(self, text: str) -> list[int]:
        return self.post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]

    def apply_template(self, messages: list[dict]) -> str:
        return self.post("/apply-template", {"messages": messages})["prompt"]

    def complete(self, prompt: str, n_predict: int, temperature: float, seed: int) -> str:
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "seed": seed,
            "cache_prompt": False,
        }
        return self.post("/completion", body)["content"]

    def hidden_states(self, tokens: list[int]) -> torch.Tensor:
        """[n_tokens, hidden] float32: the trunk's post-norm hidden state per token."""
        r = self.post("/embedding", {"content": tokens})
        emb = r[0]["embedding"] if isinstance(r, list) else r["embedding"]
        return torch.tensor(emb, dtype=torch.float32)


def _sharded(items: Iterable, shard: tuple[int, int]) -> Iterator[tuple[int, object]]:
    i, n = shard
    for k, item in enumerate(items):
        if k % n == i:
            yield k, item


def _done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            continue  # a line cut off by an interrupted run
    return done


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def generate_corpus(
    servers: list[Server],
    prompts: list[dict],
    out: Path,
    *,
    shard: tuple[int, int] = (0, 1),
    max_tokens: int = 1024,
    temperature: float = 0.7,
    workers: int = 4,
    seed: int = 0,
    progress=None,
) -> int:
    """Answers each prompt with the model and appends ``{"id", "text", "response"}`` lines.

    A prompt is ``{"prompt": str}`` or ``{"messages": [...]}``. ``text`` is the
    chat-templated prompt followed by the model's answer: the sequence the
    model sees when it generates, which is what a draft head drafts against.
    Requests are spread over ``servers`` round-robin, ``workers`` at a time.
    Returns how many lines were written.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = [(k, p) for k, p in _sharded(prompts, shard) if k not in done]

    def one(job: tuple[int, tuple[int, dict]]) -> dict:
        n, (k, p) = job
        s = servers[n % len(servers)]
        messages = p.get("messages") or [{"role": "user", "content": p["prompt"]}]
        templated = s.apply_template(messages)
        answer = s.complete(templated, max_tokens, temperature, seed + k)
        return {"id": k, "text": templated + answer, "response": answer}

    written = 0
    with open(out, "a") as f, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(one, enumerate(todo)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if progress:
                progress(written, len(todo))
    return written


def extract_features(
    server: Server,
    texts: list[dict],
    out_dir: Path,
    *,
    shard: tuple[int, int] = (0, 1),
    max_tokens: int = 8192,
    min_tokens: int = 4,
    source: dict | None = None,
    progress=None,
) -> dict:
    """Stores (tokens, hidden states) for this shard's texts under ``out_dir/shard-i-of-n``.

    ``texts`` are ``{"text": str}`` lines (e.g. from :func:`generate_corpus`).
    Texts are truncated to ``max_tokens`` (the server's per-slot context).
    A shard directory that already has a manifest is complete, so it is skipped.
    """
    i, n = shard
    shard_dir = Path(out_dir) / f"shard-{i:04d}-of-{n:04d}"
    if (shard_dir / MANIFEST).exists():
        return json.loads((shard_dir / MANIFEST).read_text())
    writer: FeatureWriter | None = None
    items = list(_sharded(texts, shard))
    for count, (k, t) in enumerate(items, 1):
        ids = server.tokenize(t["text"])[:max_tokens]
        if len(ids) < min_tokens:
            continue
        hidden = server.hidden_states(ids)
        if hidden.shape[0] != len(ids):
            raise RuntimeError(f"text {k}: {hidden.shape[0]} hidden rows for {len(ids)} tokens")
        if writer is None:
            if shard_dir.exists():  # an interrupted run: start the shard over
                for p in shard_dir.glob("features-*.safetensors"):
                    p.unlink()
            writer = FeatureWriter(shard_dir, hidden.shape[1], {"runtime": "llama-server", **(source or {})})
        writer.add(ids, hidden)
        if progress:
            progress(count, len(items))
    if writer is None:
        raise ValueError(f"shard {i}/{n}: no text with at least {min_tokens} tokens")
    return writer.close()

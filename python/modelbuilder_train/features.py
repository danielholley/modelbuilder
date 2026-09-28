"""Trunk features: per-token (token id, trunk hidden state) sequences on disk.

A frozen-trunk stage such as MTP alignment only needs the trunk's outputs, so
they are computed once with the runtime that already runs the model (the
llama.cpp, or the fork that supports its quantization), not by loading the trunk in PyTorch.

Layout of a feature directory::

    manifest.json                  {"format": "modelbuilder.features", "version": 1, ...}
    features-00000.safetensors     hidden [N, H] bf16, tokens [N] int32, seq_starts [S+1] int64

``hidden[i]`` is the trunk's hidden state *after its final norm* at token i
(llama.cpp ``h_nextn`` / ``llama-embedding --pooling none``).
"""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

MANIFEST = "manifest.json"
FORMAT = "modelbuilder.features"
VERSION = 1


class FeatureWriter:
    def __init__(self, out_dir: Path, hidden_size: int, source: dict, shard_tokens: int = 1 << 20) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / MANIFEST).exists():
            raise FileExistsError(f"{self.out} already holds features")
        self.hidden_size = hidden_size
        self.source = source
        self.shard_tokens = shard_tokens
        self.shards: list[dict] = []
        self._reset()

    def _reset(self) -> None:
        self._hidden: list[torch.Tensor] = []
        self._tokens: list[torch.Tensor] = []
        self._starts = [0]

    def add(self, tokens: list[int], hidden: torch.Tensor) -> None:
        if hidden.shape != (len(tokens), self.hidden_size):
            raise ValueError(f"hidden {tuple(hidden.shape)} does not match {len(tokens)} tokens × {self.hidden_size}")
        self._hidden.append(hidden.to(torch.bfloat16))
        self._tokens.append(torch.tensor(tokens, dtype=torch.int32))
        self._starts.append(self._starts[-1] + len(tokens))
        if self._starts[-1] >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if len(self._starts) == 1:
            return
        from safetensors.torch import save_file

        name = f"features-{len(self.shards):05d}.safetensors"
        save_file(
            {
                "hidden": torch.cat(self._hidden).contiguous(),
                "tokens": torch.cat(self._tokens).contiguous(),
                "seq_starts": torch.tensor(self._starts, dtype=torch.int64),
            },
            str(self.out / name),
        )
        self.shards.append({"file": name, "tokens": self._starts[-1], "sequences": len(self._starts) - 1})
        self._reset()

    def close(self) -> dict:
        self._flush()
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "hidden_size": self.hidden_size,
            "hidden_dtype": "bfloat16",
            "hidden_is": "trunk output after the final norm (llama.cpp h_nextn)",
            "total_tokens": sum(s["tokens"] for s in self.shards),
            "total_sequences": sum(s["sequences"] for s in self.shards),
            "shards": self.shards,
            "source": self.source,
        }
        (self.out / MANIFEST).write_text(json.dumps(manifest, indent=2))
        return manifest


@dataclass(frozen=True)
class SequenceRef:
    shard: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class FeatureSet:
    """Read access to feature data; tensors are loaded shard file by shard file, lazily.

    ``root`` is one feature directory (with ``manifest.json``) or a directory
    of them, e.g. the ``shard-*`` directories that several machines wrote with
    ``extract-features --shard i/n``. All of them are read as one dataset.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        dirs = (
            [self.root]
            if (self.root / MANIFEST).exists()
            else sorted(p.parent for p in self.root.glob(f"*/{MANIFEST}"))
        )
        if not dirs:
            raise ValueError(f"{root}: no {MANIFEST} here or in its subdirectories")
        self.files: list[Path] = []
        self.sequences: list[SequenceRef] = []
        hidden = set()
        from safetensors import safe_open

        for d in dirs:
            manifest = json.loads((d / MANIFEST).read_text())
            if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
                raise ValueError(f"{d}: not a version-{VERSION} feature directory")
            hidden.add(manifest["hidden_size"])
            for s in manifest["shards"]:
                idx = len(self.files)
                self.files.append(d / s["file"])
                with safe_open(str(self.files[-1]), framework="pt") as f:
                    starts = f.get_tensor("seq_starts").tolist()
                self.sequences += [SequenceRef(idx, a, b) for a, b in zip(starts, starts[1:], strict=False)]
        if len(hidden) != 1:
            raise ValueError(f"{root}: feature directories disagree on hidden size ({sorted(hidden)})")
        self.hidden_size: int = hidden.pop()
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def _shard(self, i: int) -> dict[str, torch.Tensor]:
        if i not in self._cache:
            from safetensors.torch import load_file

            if len(self._cache) >= 2:  # keep memory bounded
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = load_file(str(self.files[i]))
        return self._cache[i]

    def get(self, ref: SequenceRef, a: int = 0, b: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens [n] (int64) and hidden [n, H] (bf16) for positions a..b of a sequence."""
        b = ref.length if b is None else b
        sh = self._shard(ref.shard)
        lo, hi = ref.start + a, ref.start + b
        return sh["tokens"][lo:hi].long(), sh["hidden"][lo:hi]


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed with exit code {r.returncode}:\n{tail}")
    return r.stdout


def extract_llamacpp(
    llama_bin: Path,
    model: Path,
    texts: Iterable[str],
    out_dir: Path,
    threads: int = 4,
    extra_args: list[str] | None = None,
) -> dict:
    """Runs each text through the target in llama.cpp and stores (tokens, hidden states).

    Uses ``llama-tokenize --ids`` for token ids and ``llama-embedding --pooling none
    --embd-normalize -1`` for the post-norm hidden state of every token. ``-np 2`` keeps
    the embedding tool from sizing its recurrent-state cache for 256 sequences, which
    is tens of GB on hybrid models (Qwen3.5-family DeltaNet layers).
    """
    llama_bin = Path(llama_bin)
    tok_bin, emb_bin = llama_bin / "llama-tokenize", llama_bin / "llama-embedding"
    for b in (tok_bin, emb_bin):
        if not b.exists():
            raise FileNotFoundError(f"{b} not found (build the llama-tokenize and llama-embedding targets)")
    writer: FeatureWriter | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(texts):
            path = Path(tmp) / f"{i}.txt"
            path.write_text(text)
            ids = ast.literal_eval(
                _run([str(tok_bin), "-m", str(model), "-f", str(path), "--ids", "--log-disable"])
                .strip()
                .splitlines()[-1]
            )
            if len(ids) < 4:
                continue
            ctx = str(max(512, len(ids) + 16))
            out = _run(
                [
                    str(emb_bin),
                    "-m",
                    str(model),
                    "-f",
                    str(path),
                    "--pooling",
                    "none",
                    "--embd-normalize",
                    "-1",
                    "--embd-output-format",
                    "array",
                    "--embd-separator",
                    "<#modelbuilder-never#>",
                    "-np",
                    "2",
                    "-c",
                    ctx,
                    "-b",
                    ctx,
                    "-ub",
                    ctx,
                    "-t",
                    str(threads),
                    *(extra_args or []),
                ]
            )
            hidden = torch.tensor(json.loads(out), dtype=torch.float32)
            if hidden.shape[0] != len(ids):
                raise RuntimeError(f"text {i}: {hidden.shape[0]} hidden rows for {len(ids)} tokens")
            if writer is None:
                source = {
                    "runtime": "llama.cpp",
                    "model": str(model),
                    "tool": "llama-embedding --pooling none --embd-normalize -1",
                }
                writer = FeatureWriter(out_dir, hidden.shape[1], source)
            writer.add(ids, hidden)
    if writer is None:
        raise ValueError("no text produced at least 4 tokens")
    return writer.close()

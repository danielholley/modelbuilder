"""Fetch selected tensors of a Hugging Face model without downloading the whole checkpoint.

A safetensors shard is an 8-byte header length, a JSON header with each
tensor's byte range, then the data. So the headers come from two small range
requests per shard, and each wanted tensor from one more. The result is one
safetensors file with those tensors, stored exactly as in the source (no
dtype conversion), plus the small JSON and tokenizer files next to it. That is
enough to compare tensors against a reference, or to use a model's config and
tokenizer, without its tens of gigabytes of weights.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import struct
import urllib.request
from collections.abc import Callable
from pathlib import Path

SMALL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if start is not None and len(data) != end - start + 1:
        raise OSError(f"{url}: asked for {end - start + 1} bytes from {start}, got {len(data)}")
    return data


def fetch_tensors(
    repo: str,
    pattern: str,
    out: Path,
    *,
    revision: str = "main",
    endpoint: str = "https://huggingface.co",
    get: Callable[..., bytes] = _get,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, list[int]]:
    """Writes the tensors whose names match ``pattern`` (a regex) to ``out/model.safetensors``.

    Returns ``{name: shape}``. ``get(url, start, end)`` fetches a byte range (or
    the whole file without a range); tests pass a local one.
    """
    base = f"{endpoint}/{repo}/resolve/{revision}/"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in SMALL_FILES:
        with contextlib.suppress(OSError):  # optional files
            (out / f).write_bytes(get(base + f))
    try:
        index = json.loads(get(base + "model.safetensors.index.json"))
        shards: dict[str, list[str]] = {}
        for name, shard in index["weight_map"].items():
            shards.setdefault(shard, []).append(name)
    except OSError:
        shards = {"model.safetensors": []}  # one shard: names come from its header
    rx = re.compile(pattern)
    header: dict[str, dict] = {}
    offset = 0
    tmp = out / ".tensors.tmp"  # data streams here, then goes behind the header
    data_file = open(tmp, "wb")  # noqa: SIM115 - closed below, before the copy
    for shard, names in shards.items():
        if names and not any(rx.search(n) for n in names):
            continue
        url = base + shard
        (n,) = struct.unpack("<Q", get(url, 0, 7))
        h = json.loads(get(url, 8, 8 + n - 1))
        for name, info in h.items():
            if name == "__metadata__" or not rx.search(name):
                continue
            a, b = info["data_offsets"]
            data = get(url, 8 + n + a, 8 + n + b - 1) if b > a else b""
            span = [offset, offset + len(data)]
            header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": span}
            data_file.write(data)
            offset += len(data)
            if progress:
                progress(name, len(data))
    data_file.close()
    if not header:
        tmp.unlink()
        raise ValueError(f"{repo}: no tensor name matches {pattern!r}")
    header["__metadata__"] = {"format": "pt", "modelbuilder.fetched_from": f"{repo}@{revision}"}
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out / "model.safetensors", "wb") as f, open(tmp, "rb") as d:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        shutil.copyfileobj(d, f, 16 << 20)
    tmp.unlink()
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}

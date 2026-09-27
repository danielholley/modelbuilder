"""Corpus generation and extraction against a stand-in for llama-server's HTTP API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from modelbuilder_train.features import FeatureSet
from modelbuilder_train.server import Server, extract_features, generate_corpus

HIDDEN = 8


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass

    def _send(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"status": "ok"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self._send({"tokens": [ord(c) % 97 for c in body["content"]]})
        elif self.path == "/embedding":
            # A deterministic "hidden state" per (position, token), so tests can check alignment.
            rows = [[float(t + p * 1000 + j) for j in range(HIDDEN)] for p, t in enumerate(body["content"])]
            self._send([{"index": 0, "embedding": rows}])
        elif self.path == "/apply-template":
            msgs = body["messages"]
            self._send({"prompt": "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"})
        elif self.path == "/completion":
            self._send({"content": f"answer({body['prompt'][-12:]})"})
        else:
            self.send_error(404)


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Server(f"http://127.0.0.1:{httpd.server_address[1]}")
    httpd.shutdown()


def test_corpus_is_sharded_and_resumes(tmp_path: Path, server: Server) -> None:
    prompts = [{"prompt": f"q{i}"} for i in range(5)] + [{"messages": [{"role": "user", "content": "m"}]}]
    out = tmp_path / "corpus.jsonl"
    assert generate_corpus([server], prompts, out, shard=(0, 2), workers=2) == 3
    ids = [json.loads(line)["id"] for line in out.read_text().splitlines()]
    assert sorted(ids) == [0, 2, 4]
    # Rerun: nothing new. The other shard appends its own items.
    assert generate_corpus([server], prompts, out, shard=(0, 2)) == 0
    assert generate_corpus([server], prompts, out, shard=(1, 2)) == 3
    recs = {json.loads(line)["id"]: json.loads(line) for line in out.read_text().splitlines()}
    assert recs[0]["text"] == "<user>q0<assistant>" + recs[0]["response"]
    assert recs[5]["text"].startswith("<user>m<assistant>")


def test_extraction_shards_line_up_and_read_as_one_set(tmp_path: Path, server: Server) -> None:
    texts = [{"text": "abcdefgh" * (i + 1)} for i in range(5)] + [{"text": "ab"}]  # the last is too short
    for i in range(2):
        m = extract_features(server, texts, tmp_path / "feat", shard=(i, 2), max_tokens=20)
        assert m["hidden_size"] == HIDDEN
    fs = FeatureSet(tmp_path / "feat")
    assert len(fs.sequences) == 5
    assert sorted(r.length for r in fs.sequences) == [8, 16, 20, 20, 20]  # truncated to max_tokens
    for ref in fs.sequences:
        tokens, hidden = fs.get(ref)
        expected = torch.tensor(
            [[float(t + p * 1000 + j) for j in range(HIDDEN)] for p, t in enumerate(tokens.tolist())]
        )
        assert torch.allclose(hidden.float(), expected, rtol=1e-2)
    # A finished shard is not redone.
    before = (tmp_path / "feat" / "shard-0000-of-0002" / "manifest.json").stat().st_mtime_ns
    extract_features(server, texts, tmp_path / "feat", shard=(0, 2))
    assert (tmp_path / "feat" / "shard-0000-of-0002" / "manifest.json").stat().st_mtime_ns == before

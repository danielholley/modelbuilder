import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelbuilder_train import probes
from modelbuilder_train.server import Server


def test_perplexity_parses_llama_perplexity(tmp_path: Path):
    fake = tmp_path / "llama-perplexity"
    # Echo the cache flags so the test can check they were passed.
    fake.write_text('#!/bin/sh\necho "args: $*" >&2\necho "Final estimate: PPL = 7.1234 +/- 0.05678" >&2\n')
    fake.chmod(0o755)
    text = tmp_path / "t.txt"
    text.write_text("hello")
    r = probes.perplexity_llamacpp(tmp_path, Path("m.gguf"), text, ctx=512, cache_type_k="q4_0", cache_type_v="q4_0")
    assert (r.ppl, r.ppl_err, r.cache_type_v) == (7.1234, 0.05678, "q4_0")


def test_perplexity_reports_failures(tmp_path: Path):
    fake = tmp_path / "llama-perplexity"
    fake.write_text('#!/bin/sh\necho "error: model not found" >&2\nexit 1\n')
    fake.chmod(0o755)
    (tmp_path / "t.txt").write_text("x")
    with pytest.raises(RuntimeError, match="model not found"):
        probes.perplexity_llamacpp(tmp_path, Path("m.gguf"), tmp_path / "t.txt")


class Oracle(BaseHTTPRequestHandler):
    """Answers with the code it finds in the prompt, unless the context is 'too long'."""

    limit = 3_000  # characters: 2 filler paragraphs pass, 20 do not

    def log_message(self, *a) -> None:
        pass

    def _send(self, obj) -> None:
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        self._send({"status": "ok"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self._send({"tokens": list(range(len(body["content"].split())))})
        elif self.path == "/apply-template":
            self._send({"prompt": body["messages"][0]["content"]})
        else:
            m = re.search(r"secret code is (\d+)", body["prompt"])
            forgot = len(body["prompt"]) > self.limit
            self._send({"content": "I don't know" if forgot or not m else m[1]})


def test_needle_finds_short_and_misses_long():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Oracle)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        s = Server(f"http://127.0.0.1:{httpd.server_address[1]}")
        # The filler is 50 words; 100 "tokens" is 2 paragraphs, 1000 is 20 (past the oracle's limit).
        r = probes.needle(s, lengths=[100, 1000], depths=[0.0, 0.5, 1.0])
    finally:
        httpd.shutdown()
    assert [t.found for t in r.trials] == [True, True, True, False, False, False]
    assert r.accuracy == 0.5


def test_hf_vs_gguf_matches_a_model_against_itself(tmp_path: Path):
    """A fake server that answers with the HF model's own hidden states: a perfect match."""
    transformers = pytest.importorskip("transformers")
    import torch

    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )  # fmt: skip
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg)
    model.save_pretrained(tmp_path / "hf")

    class Fake:
        def tokenize(self, text: str) -> list[int]:
            return [ord(c) % 64 for c in text]

        def hidden_states(self, ids: list[int]) -> torch.Tensor:
            with torch.no_grad():
                return model.model(input_ids=torch.tensor([ids])).last_hidden_state[0]

    r = probes.hf_vs_gguf(Fake(), tmp_path / "hf", ["hello world", "x"], device="cpu")
    assert r.tokens == 11 and r.top1_agreement == 1.0
    assert r.cosine_min > 0.99999 and r.rel_rms < 1e-5


def test_skill_grading():
    assert probes.grade("exact", "10", "The output is `10`.")
    assert not probes.grade("exact", "10", "100")
    assert probes.grade("exact", "2-4-6", "2-4-6")
    assert probes.grade("any", ["buenos días", "buenos dias"], "Buenos días")
    call = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert probes.grade("call", call, 'Sure: {"name": "get_weather", "arguments": {"city": "paris"}}')
    assert not probes.grade("call", call, '{"name": "search", "arguments": {"query": "Paris"}}')
    assert not probes.grade("call", call, "It is sunny in Paris.")


class Student(Oracle):
    """Gets code and tool calls right, reasoning and languages wrong."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            self._send({"prompt": body["messages"][0]["content"]})
            return
        p = body["prompt"]
        answers = {"sum(range(5))": "10", "hello big world": "3", "'ab' * 3": "ababab", "f(5)": "120",
                   "x[-1]": "3", "join('-')": "2-4-6"}  # fmt: skip
        content = next((a for k, a in answers.items() if k in p), "no idea")
        p = p.rsplit("\n\n", 1)[-1]  # the question, after the tool list
        if "weather" in p:
            content = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        elif "euros" in p:
            content = '{"name": "convert_currency", "arguments": {"amount": 100, "source": "USD", "target": "EUR"}}'
        elif "World Cup" in p:
            content = '{"name": "search", "arguments": {"query": "2018 World Cup winner"}}'
        self._send({"content": content})


def test_skills_reports_per_category():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Student)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        r = probes.skills(Server(f"http://127.0.0.1:{httpd.server_address[1]}"))
    finally:
        httpd.shutdown()
    assert r.by_category == {"code": 1.0, "reasoning": 0.0, "tool_use": 1.0, "languages": 0.0}

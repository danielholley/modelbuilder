import json
import os
import subprocess
import sys
from pathlib import Path

from modelbuilder_train.bench import parse

OUT = """main: prompt
encoded   12 tokens in    0.500 seconds, speed:   24.000 t/s
decoded   64 tokens in   {secs} seconds, speed:   {tps} t/s

n_draft   = 3
n_predict = 64
n_drafted = {drafted}
n_accept  = {accepted}
accept    = 50.000%
"""


def test_parses_speculative_simple_output():
    r = parse(OUT.format(secs="20.000", tps="3.200", drafted=40, accepted=30))
    assert r == {"tokens": 64, "seconds": 20.0, "tokens_per_s": 3.2, "drafted": 40, "accepted": 30}


def test_bench_cli_compares_variants(tmp_path: Path):
    # A stand-in llama-speculative-simple: drafting variants are faster and report acceptance.
    fake = tmp_path / "llama-speculative-simple"
    fake.write_text(
        '#!/bin/sh\ncase "$*" in *draft-mtp*) printf "%s" "$MTP_OUT" ;; *) printf "%s" "$PLAIN_OUT" ;; esac\n'
    )
    fake.chmod(0o755)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"prompt": "a"}\n{"prompt": "b"}\n')
    env = os.environ | {
        "PLAIN_OUT": OUT.format(secs="40.000", tps="1.600", drafted=0, accepted=0),
        "MTP_OUT": OUT.format(secs="20.000", tps="3.200", drafted=40, accepted=30),
    }
    out = tmp_path / "bench.json"
    r = subprocess.run(
        [sys.executable, "-m", "modelbuilder_train", "bench-draft", "--llama-bin", str(tmp_path), "--model", "m.gguf",
         "--sidecar", "ported=p.gguf", "--prompts", str(prompts), "--out", str(out)],
        capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    summary = {row["variant"]: row for row in json.loads(out.read_text())["summary"]}
    assert summary["ported"]["acceptance"] == 0.75
    assert summary["ported"]["mean_speedup"] == 2.0
    assert summary["plain"]["drafted"] == 0
    assert "ported" in r.stdout and "75.0%" in r.stdout

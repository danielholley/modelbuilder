"""Command line: ``python -m modelbuilder_train <command>`` (or ``modelbuilder-train``)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from modelbuilder_train.events import EventWriter
from modelbuilder_train.spec import JobSpec


def cmd_run(args: argparse.Namespace) -> int:
    stream = sys.stdout if args.events == "-" else open(args.events, "a")  # noqa: SIM115
    ev = EventWriter(stream)
    try:
        spec = JobSpec.load(args.spec)
        from modelbuilder_train.backends import BACKENDS

        outputs = BACKENDS[spec.backend](spec, Path(args.spec).resolve().parent, ev)
        ev.emit("finished", status="ok", outputs=outputs)
        return 0
    except Exception as e:  # report every failure as an event, then exit non-zero
        ev.emit("error", message=f"{type(e).__name__}: {e}")
        ev.stage = None
        ev.emit("finished", status="failed", outputs={})
        traceback.print_exc(file=sys.stderr)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    spec = JobSpec.load(args.spec)
    print(f"ok: {spec.job_id}, {len(spec.stages)} stage(s)")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from modelbuilder_train.features import extract_llamacpp

    texts = [json.loads(line)["text"] for line in Path(args.texts).read_text().splitlines() if line.strip()]
    manifest = extract_llamacpp(Path(args.llama_bin), Path(args.model), texts, Path(args.out), threads=args.threads)
    print(f"wrote {manifest['total_tokens']} tokens in {manifest['total_sequences']} sequences to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from safetensors import safe_open

    from modelbuilder_train.features import FeatureSet
    from modelbuilder_train.mtp.model import MtpConfig, MtpHead, load_mtp_state
    from modelbuilder_train.mtp.train import _files, evaluate, pick_device

    device = pick_device(args.device)
    cfg = MtpConfig.from_hf_config(json.loads(Path(args.reference_config).read_text()))
    head = MtpHead(cfg)
    head.load_state_dict(load_mtp_state(_files(Path(args.head))), strict=True)
    head.to(device)
    with safe_open(args.frozen, framework="pt") as f:
        emb = f.get_tensor(args.embedding_tensor).to(device)
        lm = f.get_tensor(args.lm_head_tensor).to(device)
    feats = FeatureSet(Path(args.features))
    r = evaluate(head, emb, lm, feats, [s for s in feats.sequences if s.length >= 4], args.seq_len, device)
    print(json.dumps(r))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="modelbuilder-train")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a job spec, writing JSONL events")
    r.add_argument("spec")
    r.add_argument("--events", default="-", help="events file ('-' for stdout)")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="check a job spec against the schema")
    v.add_argument("spec")
    v.set_defaults(fn=cmd_validate)

    x = sub.add_parser("extract-features", help="compute trunk features with llama.cpp")
    x.add_argument("--llama-bin", required=True, help="directory with llama-tokenize and llama-embedding")
    x.add_argument("--model", required=True, help="target GGUF")
    x.add_argument("--texts", required=True, help='JSONL with {"text": ...} per line')
    x.add_argument("--out", required=True)
    x.add_argument("--threads", type=int, default=4)
    x.set_defaults(fn=cmd_extract)

    e = sub.add_parser("evaluate-mtp", help="top-1 accuracy of an MTP head on features")
    e.add_argument("--reference-config", required=True)
    e.add_argument("--head", required=True, help="file or directory with mtp.* tensors")
    e.add_argument("--frozen", required=True)
    e.add_argument("--embedding-tensor", default="token_embd.weight")
    e.add_argument("--lm-head-tensor", default="output.weight")
    e.add_argument("--features", required=True)
    e.add_argument("--seq-len", type=int, default=1024)
    e.add_argument("--device", default="auto")
    e.set_defaults(fn=cmd_evaluate)

    args = p.parse_args(argv)
    return args.fn(args)

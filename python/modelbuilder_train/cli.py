"""Command line: ``python -m modelbuilder_train <command>`` (or ``modelbuilder-train``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from modelbuilder_train.events import EventWriter
from modelbuilder_train.spec import JobSpec


def cmd_run(args: argparse.Namespace) -> int:
    from modelbuilder_train import dist

    # Under torchrun only rank 0 reports; the others' events would be duplicates.
    if not dist.is_main():
        stream = open(os.devnull, "w")  # noqa: SIM115
    else:
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


def _shard(s: str) -> tuple[int, int]:
    i, _, n = s.partition("/")
    shard = (int(i), int(n or 1))
    if not 0 <= shard[0] < shard[1]:
        raise argparse.ArgumentTypeError(f"shard {s}: use i/n with 0 <= i < n")
    return shard


def _servers(args: argparse.Namespace, embeddings: bool):
    """The servers to use: --server URLs, or one launched here from --llama-bin/--model."""
    from modelbuilder_train.server import Server

    if args.server:
        return [Server(u) for u in args.server], []
    if not (args.llama_bin and args.model):
        raise SystemExit("give --server URL(s), or --llama-bin and --model to start one")
    s = Server.launch(
        Path(args.llama_bin),
        Path(args.model),
        port=args.port,
        embeddings=embeddings,
        parallel=args.parallel,
        ctx=args.ctx,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        log=Path(args.server_log) if args.server_log else None,
    )
    return [s], [s]


def _progress(label: str):
    t0 = time.time()

    def show(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"{label}: {done}/{total} ({rate:.2f}/s)", file=sys.stderr, flush=True)

    return show


def cmd_corpus(args: argparse.Namespace) -> int:
    from modelbuilder_train.server import generate_corpus, read_jsonl

    servers, owned = _servers(args, embeddings=False)
    try:
        n = generate_corpus(
            servers,
            read_jsonl(Path(args.prompts)),
            Path(args.out),
            shard=args.shard,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            workers=args.workers or args.parallel * len(servers),
            seed=args.seed,
            progress=_progress("generated"),
        )
    finally:
        for s in owned:
            s.close()
    print(f"wrote {n} new samples to {args.out}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    texts = [json.loads(line) for line in Path(args.texts).read_text().splitlines() if line.strip()]
    if args.no_server:
        from modelbuilder_train.features import extract_llamacpp

        manifest = extract_llamacpp(
            Path(args.llama_bin),
            Path(args.model),
            [t["text"] for t in texts],
            Path(args.out),
            threads=args.threads or 4,
        )
    else:
        from modelbuilder_train.server import extract_features

        servers, owned = _servers(args, embeddings=True)
        try:
            manifest = extract_features(
                servers[0],
                texts,
                Path(args.out),
                shard=args.shard,
                max_tokens=args.ctx,
                source={"model": str(args.model or args.server[0])},
                progress=_progress("extracted"),
            )
        finally:
            for s in owned:
                s.close()
    print(f"wrote {manifest['total_tokens']} tokens in {manifest['total_sequences']} sequences under {args.out}")
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

    def server_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("llama-server (use running ones, or start one here)")
        g.add_argument("--server", action="append", help="URL of a running llama-server (repeatable)")
        g.add_argument("--llama-bin", help="directory with llama-server (the PrismML fork's build/bin)")
        g.add_argument("--model", help="target GGUF")
        g.add_argument("--port", type=int, default=8080)
        g.add_argument("--parallel", type=int, default=4, help="server slots")
        g.add_argument("--ctx", type=int, default=8192, help="context per slot (texts are cut to this)")
        g.add_argument("--threads", type=int, help="CPU threads")
        g.add_argument("--gpu-layers", type=int, help="layers to offload (-ngl); 99 for all")
        g.add_argument("--server-log", help="file for the server's output")
        p.add_argument("--shard", type=_shard, default=(0, 1), help="i/n: this machine's share of the input")

    c = sub.add_parser("generate-corpus", help="the model's own answers to prompts, via llama-server")
    c.add_argument("--prompts", required=True, help='JSONL: {"prompt": ...} or {"messages": [...]} per line')
    c.add_argument("--out", required=True, help="JSONL to append to (resumes: finished ids are skipped)")
    c.add_argument("--max-tokens", type=int, default=1024)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--workers", type=int, help="concurrent requests (default: slots × servers)")
    c.add_argument("--seed", type=int, default=0)
    server_args(c)
    c.set_defaults(fn=cmd_corpus)

    x = sub.add_parser("extract-features", help="trunk hidden states per token, via llama-server")
    x.add_argument("--texts", required=True, help='JSONL with {"text": ...} per line')
    x.add_argument("--out", required=True, help="writes out/shard-i-of-n; training reads the whole directory")
    x.add_argument("--no-server", action="store_true", help="old mode: llama-tokenize + llama-embedding per text")
    server_args(x)
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

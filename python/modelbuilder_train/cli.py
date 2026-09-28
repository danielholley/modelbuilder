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


def cmd_bench(args: argparse.Namespace) -> int:
    from modelbuilder_train import bench

    sidecars = {}
    for item in args.sidecar:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = Path(item).stem, item
        sidecars[name] = Path(path)
    prompts = [json.loads(line)["prompt"] for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    results = bench.run(
        Path(args.llama_bin),
        Path(args.model),
        sidecars,
        prompts,
        n_predict=args.n_predict,
        ctx=args.ctx,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        baseline=not args.no_baseline,
        progress=lambda r: print(
            f"prompt {r.prompt} {r.variant:<12} {r.tokens_per_s:7.2f} t/s  accepted {r.accepted}/{r.drafted}",
            file=sys.stderr,
            flush=True,
        ),
    )
    report = bench.to_json(results)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'variant':<14}{'accepted':>16}{'rate':>9}{'tok/s':>9}{'speedup':>9}")
    for row in report["summary"]:
        rate = f"{100 * row['acceptance']:.1f}%" if row["acceptance"] is not None else "–"
        speed = f"{row['mean_speedup']:.2f}×" if row["mean_speedup"] is not None else "–"
        acc = f"{row['accepted']}/{row['drafted']}"
        print(f"{row['variant']:<14}{acc:>16}{rate:>9}{row['mean_tokens_per_s']:>9.2f}{speed:>9}")
    return 0


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _needle_progress(t) -> None:
    status = "found" if t.found else "MISSED"
    print(f"{t.context_tokens:>7} tokens, depth {t.depth:.2f}: {status} ({t.answer!r})", file=sys.stderr)


def _sweep_progress(row: dict) -> None:
    print(f"{row['type']}: ppl {row.get('ppl')}, retrieval {row.get('needle_accuracy')}", file=sys.stderr)


def cmd_probe(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from modelbuilder_train import probes

    if args.probe == "perplexity":
        if args.hf:
            report = probes.perplexity_hf(Path(args.hf), Path(args.text).read_text(), ctx=args.ctx, device=args.device)
        else:
            report = asdict(
                probes.perplexity_llamacpp(
                    Path(args.llama_bin), Path(args.model), Path(args.text), ctx=args.ctx, chunks=args.chunks,
                    cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v,
                    threads=args.threads, gpu_layers=args.gpu_layers,
                )
            )  # fmt: skip
        print(f"perplexity {report['ppl']:.4f}")
    elif args.probe == "needle":
        from modelbuilder_train.server import Server

        extra = ["-ctk", args.cache_type_k, "-ctv", args.cache_type_v]
        if args.cache_type_v not in ("f16", "f32", "bf16"):
            extra += ["-fa", "on"]
        lengths = _ints(args.lengths)
        with Server.launch(
            Path(args.llama_bin), Path(args.model), port=args.port, parallel=1, ctx=max(lengths) + 256,
            threads=args.threads, gpu_layers=args.gpu_layers, extra=extra,
        ) as s:  # fmt: skip
            r = probes.needle(s, lengths=lengths, depths=_floats(args.depths), model=args.model,
                              cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v,
                              progress=_needle_progress)  # fmt: skip
        report = {"model": r.model, "accuracy": r.accuracy, "trials": [asdict(t) for t in r.trials]}
        print(f"retrieval accuracy {100 * r.accuracy:.1f}% over {len(r.trials)} trials")
    else:  # kv-cache
        report = probes.kv_cache_sweep(
            Path(args.llama_bin), Path(args.model), types=args.types.split(","),
            text=Path(args.text) if args.text else None, ctx=args.ctx, chunks=args.chunks,
            needle_lengths=_ints(args.lengths) if args.lengths else None, needle_depths=_floats(args.depths),
            threads=args.threads, gpu_layers=args.gpu_layers,
            progress=_sweep_progress,
        )  # fmt: skip
        print(f"{'cache':<8}{'ppl':>10}{'Δ ppl':>9}{'needle':>9}")
        for r in report["summary"]:
            ppl = f"{r['ppl']:.3f}" if r["ppl"] is not None else "–"
            delta = f"{r['ppl_delta_pct']:+.2f}%" if r["ppl_delta_pct"] is not None else "–"
            nd = f"{100 * r['needle_accuracy']:.0f}%" if r["needle_accuracy"] is not None else "–"
            print(f"{r['type']:<8}{ppl:>10}{delta:>9}{nd:>9}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


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

    b = sub.add_parser("bench-draft", help="draft acceptance and speedup of MTP sidecars in llama.cpp")
    b.add_argument("--llama-bin", required=True, help="directory with llama-speculative-simple")
    b.add_argument("--model", required=True, help="target GGUF")
    b.add_argument(
        "--sidecar", action="append", required=True, help="name=path.gguf (repeatable), e.g. ported=... aligned=..."
    )
    b.add_argument(
        "--prompts",
        required=True,
        help='JSONL with {"prompt": ...} per line (raw text; apply the chat template yourself)',
    )
    b.add_argument("--n-predict", type=int, default=128)
    b.add_argument(
        "--ctx", type=int, default=4096, help="context size (the default 262K would allocate a 16 GB KV cache)"
    )
    b.add_argument("--threads", type=int)
    b.add_argument("--gpu-layers", type=int, help="-ngl and -ngld; 99 for all")
    b.add_argument("--no-baseline", action="store_true", help="skip the plain (no drafting) runs")
    b.add_argument("--out", help="write all results as JSON")
    b.set_defaults(fn=cmd_bench)

    pr = sub.add_parser("probe", help="measure quality: perplexity, long-context retrieval, KV cache types")
    psub = pr.add_subparsers(dest="probe", required=True)

    def probe_common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--llama-bin", help="directory with llama-perplexity / llama-server")
        q.add_argument("--model", help="GGUF to probe")
        q.add_argument("--threads", type=int)
        q.add_argument("--gpu-layers", type=int)
        q.add_argument("--ctx", type=int, default=2048)
        q.add_argument("--chunks", type=int, default=20, help="perplexity: chunks of --ctx tokens to score")
        q.add_argument("--out", help="write the report as JSON")
        q.set_defaults(fn=cmd_probe)

    q = psub.add_parser("perplexity", help="perplexity on a text file (GGUF via llama.cpp, or --hf in PyTorch)")
    q.add_argument("--text", required=True, help="plain-text file")
    q.add_argument("--hf", help="an HF model directory instead of --model")
    q.add_argument("--device", default="auto")
    q.add_argument("--cache-type-k", default="f16")
    q.add_argument("--cache-type-v", default="f16")
    probe_common(q)
    q = psub.add_parser("needle", help="long-context retrieval of a hidden fact at several depths")
    q.add_argument("--lengths", default="1024,4096,16384", help="context lengths in tokens")
    q.add_argument("--depths", default="0.1,0.5,0.9")
    q.add_argument("--cache-type-k", default="f16")
    q.add_argument("--cache-type-v", default="f16")
    q.add_argument("--port", type=int, default=8093)
    probe_common(q)
    q = psub.add_parser("kv-cache", help="perplexity and retrieval per KV cache type, vs f16")
    q.add_argument("--types", default="q8_0,q4_0", help="llama.cpp cache types, e.g. q8_0,q5_0,q4_0")
    q.add_argument("--text", help="plain-text file for perplexity")
    q.add_argument("--lengths", help="needle context lengths (omit to skip retrieval)")
    q.add_argument("--depths", default="0.1,0.5,0.9")
    probe_common(q)

    args = p.parse_args(argv)
    return args.fn(args)

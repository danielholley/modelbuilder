# Runbook: align an MTP head to Ternary-Bonsai-2-27B on a local cluster

This runbook trains a speculative-decoding (MTP) head against the frozen
Bonsai 2 trunk and measures whether it beats the ported head. It is written
for this cluster:

- 2 GPU servers with 4 × P40 each;
- 9 CPU servers with 256 GB and 32 threads;
- 1 CPU server with 768 GB;
- dual 40 GbE, and a shared 50 TB array.

The design and the checks behind it are in
[`../research/mtp-align.md`](../research/mtp-align.md).

**Nothing below has been run on this hardware yet.** Each step was checked on
the real Bonsai 2 file on a 16 GB CPU machine (corpus generation, server
extraction and `bench-draft`), or in tests (`torchrun`/DDP). GPU throughput is
unmeasured: step 1 checks it before anything big runs.

## The pipeline

```
prompts.jsonl ──generate-corpus──► corpus/*.jsonl ──extract-features──► feat/shard-*/   (10 KB/token on the array)
                   (llama-server)                      (llama-server --embeddings)
feat/ + Qwen3.8 MTP head ──job mtp-align --emit-only──► job.json ──torchrun ×8 P40──► mtp-head/
mtp-head/ ──surgery mtp──► Bonsai-2-mtp-aligned.gguf ──bench-draft──► acceptance vs the ported head
```

The trunk never runs in PyTorch. It runs in the PrismML llama.cpp fork, the
only runtime for its ternary weights. Only the 425 M-parameter head trains, in
fp32: P40s have no bf16, and the code falls back automatically.

Sizes to plan for:

| Tokens | Features on disk | Training at an estimated 4,000 tokens/s on 8 P40s |
|---|---|---|
| 5 M (smoke run) | 50 GB | about 20 min |
| 50 M | 500 GB | about 3.5 h |
| 300 M | 3 TB | about 21 h |

The training throughput is an estimate from FLOPs (about 10 GFLOP/token,
dominated by the 248K-vocab LM head), not a measurement. The planner's
`4xP40` profile charges a full trunk forward pass per token, which this
precomputed-feature setup doesn't do, so its hours are an upper bound.

## 0. Common setup (every machine)

The paths assume the array is mounted at `/array` on every machine.

```sh
export MB=/array/mb            # everything below lives here
mkdir -p $MB/{models,runs,corpus,feat}

# modelbuilder (Rust; any stable toolchain ≥ 1.85)
git clone https://github.com/danielholley/modelbuilder $MB/modelbuilder
cd $MB/modelbuilder && cargo build --release
export PATH=$MB/modelbuilder/target/release:$PATH

# The Python side. On GPU servers, install a PyTorch build that still has sm_61
# (see the P40 note below) BEFORE this.
python3 -m venv $MB/venv && $MB/venv/bin/pip install -e "$MB/modelbuilder/python[dev]"
```

**PyTorch on P40 (sm_61).** Recent CUDA 12.8+ wheels may not include Pascal
kernels. Check:

```sh
$MB/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.get_device_capability(0))"
```

`sm_61` (or `sm_60`) must be in the list. If it isn't, install an older CUDA
build of PyTorch, e.g. the `cu126` or `cu118` wheels from
https://download.pytorch.org/whl/.

**The PrismML llama.cpp fork**, built per machine type:

```sh
git clone https://github.com/PrismML-Eng/llama.cpp $MB/src/prism-llama.cpp
cd $MB/src/prism-llama.cpp
# GPU servers (P40 = compute capability 6.1)
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build-cuda -j --target llama-server llama-speculative-simple
# CPU servers (Xeon E5 v2: AVX, no AVX2; build on the machine itself)
cmake -B build-cpu -DGGML_NATIVE=ON && cmake --build build-cpu -j --target llama-server llama-speculative-simple
```

## 1. Models, and a check that the GPUs work

```sh
cd $MB/models
# Target: the 7.2 GB PQ2_0 file.
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir .
# Reference: only Qwen3.8-27B's config, tokenizer, and the shards that hold mtp.* (not all 54 GB).
hf download Qwen/Qwen3.8-27B config.json tokenizer.json tokenizer_config.json model.safetensors.index.json --local-dir Qwen3.8-27B
python3 - <<'EOF'
import json; m = json.load(open("Qwen3.8-27B/model.safetensors.index.json"))["weight_map"]
print(" ".join(sorted({f for k, f in m.items() if k.startswith("mtp.")})))
EOF
hf download Qwen/Qwen3.8-27B <the shard names printed above> --local-dir Qwen3.8-27B
```

The ported head is the baseline, and the initial weights for alignment:

```sh
modelbuilder surgery mtp models/Ternary-Bonsai-2-27B-PQ2_0.gguf --from models/Qwen3.8-27B \
    -o models/Ternary-Bonsai-2-27B-mtp-ported.gguf
```

**GPU smoke test (do this first).** Run it on one P40 with a few prompts:

```sh
CUDA_VISIBLE_DEVICES=0 $MB/venv/bin/python -m modelbuilder_train bench-draft \
    --llama-bin $MB/src/prism-llama.cpp/build-cuda/bin --model $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --sidecar ported=$MB/models/Ternary-Bonsai-2-27B-mtp-ported.gguf \
    --prompts bench-prompts.jsonl --gpu-layers 99 --n-predict 128 --out $MB/runs/bench-ported.json
```

It shows three things:
- whether PQ2_0 runs on the P40s;
- the decode speed, which sets how long corpus generation takes;
- the ported head's acceptance, which is the number to beat.

On CPU the port accepted 64–87% of drafts (`mtp-port.md`). The fork has CUDA
PQ2_0 kernels on the `dp4a` path, which compute capability 6.1 supports, but
they are untested on Pascal here. If the GPU run fails, use the CPU servers
for steps 2–3.

`bench-prompts.jsonl` holds `{"prompt": ...}` lines of raw text. For chat
prompts, apply the Qwen chat template yourself; `corpus/*.jsonl` `text`
fields are already templated.

## 2. Corpus: the model's own answers

Alignment should train on what the model actually generates. The draft head
has to predict *this* model's next tokens, not a dataset's. You supply the
prompts, as `{"prompt": ...}` or `{"messages": [...]}` lines. Pick
instruction, chat and code sets whose licenses fit your use, and match the
mix to what you'll serve.

On each P40 server, run one server per GPU, then one generator over all
four. Use `--shard k/N`, with a different `k` on each machine:

```sh
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup $MB/src/prism-llama.cpp/build-cuda/bin/llama-server \
      -m $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 --port $((8080+g)) -np 4 -c 16384 \
      > $MB/runs/server-gen-$g.log 2>&1 &
done
$MB/venv/bin/python -m modelbuilder_train generate-corpus --prompts $MB/prompts.jsonl \
    --out $MB/corpus/$(hostname).jsonl --shard 0/2 --max-tokens 1024 --temperature 0.7 \
    --server http://127.0.0.1:8080 --server http://127.0.0.1:8081 --server http://127.0.0.1:8082 --server http://127.0.0.1:8083
```

- `-c 16384 -np 4` gives each of the 4 slots 4096 tokens.
- The CPU servers can generate too (one `llama-server` with
  `-t 32 -np 2` each; add them to `--shard`), but decode on Ivy Bridge will
  be a few tokens per second.
- The run resumes: rerunning skips finished ids.

Merge the per-machine files when all shards are done:

```sh
cat $MB/corpus/*.jsonl > $MB/corpus.jsonl
```

## 3. Features: trunk hidden states per token

Prefill is much cheaper than decode, and all 11 machines can help. On each
machine `i` of `N`:

```sh
# GPU server, per GPU g (use 4 shards per GPU server)
CUDA_VISIBLE_DEVICES=$g $MB/venv/bin/python -m modelbuilder_train extract-features \
    --texts $MB/corpus.jsonl --out $MB/feat --shard $i/$N \
    --llama-bin $MB/src/prism-llama.cpp/build-cuda/bin --model $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --gpu-layers 99 --ctx 4096 --port $((9090+g))
# CPU server
$MB/venv/bin/python -m modelbuilder_train extract-features \
    --texts $MB/corpus.jsonl --out $MB/feat --shard $i/$N \
    --llama-bin $MB/src/prism-llama.cpp/build-cpu/bin --model $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --threads 32 --ctx 4096
```

- Each shard writes `feat/shard-iiii-of-NNNN/`. A finished shard is skipped
  on rerun; an unfinished one starts over.
- Training reads the whole `feat/` directory.
- Features are 10 KB per token (5120 × bf16).

## 4. Train the head on the 8 P40s

Write the job spec once. This also exports the frozen embedding and LM head
from the GGUF, decoded and un-rotated (5 GB):

```sh
modelbuilder job mtp-align $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --from $MB/models/Qwen3.8-27B \
    --features $MB/feat -o $MB/runs/mtp-align-1 --emit-only \
    --steps 1500 --seq-len 1024 --batch-seqs 4 --lr 1e-4 --fp32
```

Each step trains `ranks × batch_seqs × seq_len` tokens: 8 × 4 × 1024 = 32K,
so 1500 steps is about 50 M tokens. Then launch, with one node or two:

```sh
cd $MB/runs/mtp-align-1
# one server, 4 GPUs
$MB/venv/bin/torchrun --standalone --nproc-per-node 4 -m modelbuilder_train run job.json --events events.jsonl
# both servers (run on each, NODE=0 on the first)
$MB/venv/bin/torchrun --nnodes 2 --node-rank $NODE --master-addr <gpu-server-1> --master-port 29500 \
    --nproc-per-node 4 -m modelbuilder_train run job.json --events events.jsonl
```

Watch it from any machine:

```sh
modelbuilder tui --events $MB/runs/mtp-align-1/events.jsonl
```

Or open the web dashboard's Jobs page and follow the same events file.

- **What to expect:** eval top-1 at step 0 is the ported head's accuracy on
  your corpus. It should rise from there.
- **Memory:** about 17–20 GB per P40 in fp32 (head, gradients, AdamW, and the
  fp32 embedding and LM head). If it doesn't fit, lower `--batch-seqs` or
  `--seq-len`.

## 5. Sidecar and benchmark

```sh
modelbuilder surgery mtp $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --from $MB/runs/mtp-align-1/mtp-head \
    -o $MB/models/Ternary-Bonsai-2-27B-mtp-aligned.gguf
CUDA_VISIBLE_DEVICES=0 $MB/venv/bin/python -m modelbuilder_train bench-draft \
    --llama-bin $MB/src/prism-llama.cpp/build-cuda/bin --model $MB/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --sidecar ported=$MB/models/Ternary-Bonsai-2-27B-mtp-ported.gguf \
    --sidecar aligned=$MB/models/Ternary-Bonsai-2-27B-mtp-aligned.gguf \
    --prompts bench-prompts.jsonl --gpu-layers 99 --n-predict 256 --out $MB/runs/bench-aligned.json
```

**Use prompts held out from the corpus.** Record the result in
`docs/research/mtp-align.md`: acceptance and speedup, ported against aligned,
and on which hardware.

## Troubleshooting

- **`llama-server` dies at start with a huge allocation:** the context
  defaults to the model's 262K. Always pass `-c`. `extract-features`,
  `generate-corpus` and `bench-draft` do this for you.
- **Features don't line up with text:** use the server path (the default).
  The old `--no-server` path runs text through llama.cpp's escape
  processing, which turns `\t` in text into a tab.
- **Training runs in fp32 on the P40s:** that is expected. bf16 needs compute
  capability 8.0, and the backend falls back to fp32 and says so on stderr.

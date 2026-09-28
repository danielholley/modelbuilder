# Verify the HF export and run KV-cache QAT on the first target

Notes for the first target (`prism-ml/Ternary-Bonsai-2-27B-gguf`, 64 layers
in a period-4 pattern of 3 DeltaNet + 1 gated attention, 248K vocabulary).
The commands are generic; only the paths and numbers here are specific to it.

## 1. Check `export-hf` against llama.cpp on a pruned copy

A full fp32 export is about 108 GB. A 4-layer copy is enough to check every
tensor type and transform (DeltaNet, attention, norms, rotation), and it
keeps the embedding and LM head.

```sh
M=models/Ternary-Bonsai-2-27B-PQ2_0.gguf
REF=models/Qwen3.8-27B-config     # config.json + tokenizer files only; no weights needed
mkdir -p runs/verify
# Keep layers 0..3; 60 is a whole number of periods, so the pattern survives.
modelbuilder surgery prune $M --layers 4..64 -o runs/verify/target-4l.gguf
modelbuilder export-hf runs/verify/target-4l.gguf --reference $REF --dtype f32 -o runs/verify/hf-4l
python -m modelbuilder_train probe hf-vs-gguf --llama-bin <fork>/build/bin \
  --model runs/verify/target-4l.gguf --hf runs/verify/hf-4l \
  --texts texts.jsonl --ctx 512 --device cpu --out runs/verify/match.json
```

Disk: about 17 GB for the f32 export, most of it the embedding and LM head
(248K × 5120 each). With `--dtype bf16` it is about 8.5 GB, but the check
itself runs in float32.

What to expect: llama.cpp's CPU path quantizes activations (Q8) for its dot
products, so the match is close but not exact. A cosine near 1 and a top-1
agreement near 100% mean the export is right. A systematic error shows up
as a much lower value. For example, a missing norm offset (+1) gave 0% draft
acceptance in `docs/research/mtp-port.md`.

## 2. Decide whether KV-cache QAT is needed

Measure first, with no training:

```sh
python -m modelbuilder_train probe kv-cache --llama-bin <fork>/build/bin --model $M \
  --types q8_0,q4_0 --text wikitext2-test.txt --ctx 2048 --chunks 40 \
  --lengths 4096,16384,65536 --depths 0.1,0.5,0.9 --out runs/kv-sweep.json
```

If q4_0 costs little perplexity and no retrieval, run it without training.

## 3. KV-cache QAT (if needed)

```sh
modelbuilder job trunk-distill $M --reference $REF --texts corpus/distill.jsonl \
  -o runs/kv-q4 --kv-format q4_0 \
  --trainable self_attn.k_proj.weight,self_attn.v_proj.weight,self_attn.k_norm.weight \
  --emit-only
modelbuilder job run runs/kv-q4/job.json      # on the GPU machine
modelbuilder surgery replace $M --updates runs/kv-q4/updates -o runs/kv-q4/target-kv-q4.gguf --max-rel-error 1e-3
```

The stage loads the whole trunk on one device: about 108 GB in fp32, or about
54 GB plus activations in bf16 on Ampere or newer. That rules out the P40
servers (no bf16, 24 GB each) until the stage gets FSDP. Trained tensors go
through PQ2_0 in the rotated basis during training, so the write-back
re-encodes them exactly (check the reported re-quantization error).

#!/usr/bin/env bash
# Wait for the reasoning sweep to finish, release the GPU, then run the six-distribution
# KLD sweep — the last *pending* block on the release card.
#
# These two cannot overlap: vLLM holds ~89 GB serving the sweep, and KLD needs to hold a
# bf16 reference AND a quantized model. --two-pass keeps only one on the card at a time
# (the reference's logits are cached in CPU RAM as fp16, which is lossless against bf16 —
# fp16 has 10 mantissa bits to bf16's 7), but even one model plus a live vLLM will not fit.
#
# The quantized side reads `checkpoint`, NOT `checkpoint-vllm`: transformers needs the
# native model.language_model.* naming. checkpoint-vllm is the renamed serving artifact.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
REF=$REPO/out/exp-060/model_extracted
CORPORA=$REPO/out/exp-060-32k/corpora
cd $REPO

echo "waiting for the four sweep CSVs …"
for _ in $(seq 1 720); do            # up to 6h
  n=0
  for lvl in xhigh medium low off; do
    [ -f "$OUT/results/toolcall_w4a16_${lvl}.csv" ] && n=$((n+1))
  done
  [ "$n" -eq 4 ] && { echo "sweep complete ($n/4)"; break; }
  sleep 30
done

# Release the GPU, by recorded PID. Never pkill -f: it would match this script too.
if [ -f "$OUT/vllm.pid" ]; then
  pid=$(cat "$OUT/vllm.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping vllm pid $pid"
    kill "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$OUT/vllm.pid"
fi
sleep 20
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== KLD: 6 distributions vs the bf16 reference ==="
PYTHONPATH=src .venv/bin/python scripts/run_hf_kld.py \
  --ref "$REF" \
  --quant "$OUT/checkpoint" \
  --corpora-dir "$CORPORA" \
  --out "$OUT/results/kld_results.csv" \
  --ctx 8192 \
  --two-pass \
  --model-class Qwen3_5ForCausalLM \
  2>&1 | tee "$OUT/logs/kld.log" | grep -vE "^  \["

#!/usr/bin/env bash
# CPU-only end-to-end smoke test for the Jacobian-lens stack on a tiny GGUF.
#
# Exercises: build the native server -> fit a small lens -> capture two runs
# -> diff them -> serve + curl the API. Everything is CPU-viable on a ~0.6B
# model; this is the per-phase acceptance gate.
#
#   QT_TINY_GGUF=/path/to/tiny-Q8_0.gguf bash scripts/lens_smoke.sh
#
# If QT_TINY_GGUF is unset, prints how to fetch one and exits 0 (skip).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ -z "${QT_TINY_GGUF:-}" ]; then
    echo "QT_TINY_GGUF unset — skipping the model-dependent smoke."
    echo "To run it, point QT_TINY_GGUF at a small GGUF llama.cpp can load, e.g.:"
    echo "  huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir /tmp/tiny"
    echo "  QT_TINY_GGUF=/tmp/tiny/Qwen3-0.6B-Q8_0.gguf bash scripts/lens_smoke.sh"
    exit 0
fi

MODEL="$QT_TINY_GGUF"
WORK="${QT_SMOKE_WORK:-$REPO/out/lens_smoke}"
mkdir -p "$WORK"
RUN="uv run"

echo "== 1. build jlens-server =="
bash native/jlens_server/build.sh

echo "== 2. fit a tiny lens (2-layer band, small corpus) =="
$RUN quant-tuner lens fit --model "$MODEL" --corpus "wikitext:20" \
    --layers 0-3 --band-size 2 --n-prompts 20 --max-seq-len 64 --ngl 0 \
    -o "$WORK/tiny.lens.gguf"
$RUN quant-tuner lens inspect "$WORK/tiny.lens.gguf"

echo "== 3. capture two runs (clean + tail) =="
A=$($RUN quant-tuner lens capture --model "$MODEL" --runs-dir "$WORK/runs" \
    --prompt "The capital of France is" --tail 8 --ngl 0 | awk '{print $2}')
B=$($RUN quant-tuner lens capture --model "$MODEL" --runs-dir "$WORK/runs" \
    --prompt "The capital of France is" --tail 8 --ngl 0 | awk '{print $2}')
echo "captured runs: $A $B"

echo "== 4. diff (identical prompts should fully agree) =="
$RUN quant-tuner lens diff "$A" "$B" --runs-dir "$WORK/runs" \
    --lens "$WORK/tiny.lens.gguf" --out "$WORK/diff"

echo "== 5. serve + curl the API =="
$RUN quant-tuner lens serve --model "$MODEL" --lens "$WORK/tiny.lens.gguf" \
    --runs-dir "$WORK/runs" --ngl 0 --port 8099 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8099/api/backends >/dev/null 2>&1; then break; fi
    sleep 1
done
curl -sf "http://127.0.0.1:8099/api/runs" | head -c 400
echo
echo "== smoke OK =="

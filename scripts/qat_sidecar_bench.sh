#!/usr/bin/env bash
# CPU sidecar bench of a mid-run coder1 checkpoint — GPU untouched (trainer owns it).
# export Q2_0 -> CPU llama-server -> GGUF stop probe -> graded mimic episode (T=0.7).
set -uo pipefail
cd /workspace/Quant-Tuner
STEP="${1:?usage: qat_sidecar_bench.sh STEP}"
RUN="${RUN:-out/exp-059/kd32b-full-coder1}"
PREFIX="${PREFIX:-coder1}"
CK=${RUN}/trained_latents.step${STEP}.pt
[ -f "$CK" ] || { echo "[sidecar] no checkpoint $CK"; exit 1; }
TAG=${PREFIX}-s${STEP}
UTAG=$(echo "$TAG" | tr "[:lower:]" "[:upper:]")
export CUDA_VISIBLE_DEVICES=""
export LLAMA_CPP_DIR=vendor/llama.cpp-prism

echo "[sidecar] 1/3 export $CK -> $TAG"
PYTHONPATH=src nice -n 10 .venv/bin/python scripts/exp057_qat_export.py \
    --latents "$CK" --tag "$TAG" 2>&1 | grep -vE "it/s\]|%\|" | tail -3
GGUF=out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf
[ -f "$GGUF" ] || { echo "[sidecar] export produced no $GGUF"; exit 1; }

BIN=vendor/llama.cpp-prism/build/bin/llama-server
PORT=18095
nice -n 10 "$BIN" --model "$GGUF" --ctx-size 32768 --n-gpu-layers 0 --threads 48 \
    --jinja --host 127.0.0.1 --port $PORT > out/exp-059/sidecar_server_${TAG}.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[sidecar] server died (see server log)"; exit 1; }
  sleep 5
done
[ "$ok" = "1" ] || { echo "[sidecar] server never healthy"; exit 1; }

echo "[sidecar] 2/3 GGUF stop probe"
PYTHONPATH=src .venv/bin/python scripts/probe_stop_prob.py \
    --base-url "http://127.0.0.1:$PORT" --label "${TAG}-Q2_0-cpu" 2>&1 | tail -8

echo "[sidecar] 3/3 mimic episode (dask, T=0.7)"
cd /workspace/swe-mimic
timeout 10800 .venv/bin/python run_agent.py \
    --instance instance.json \
    --base-url "http://127.0.0.1:$PORT/v1" --label "${UTAG}-CPU" \
    --temperature 0.7 --out "sidecar_${PREFIX}.csv" 2>&1 | tail -12
echo "[sidecar] done rc=$?"

#!/usr/bin/env bash
# Swap the serving GPU from the bf16 reference to the W4A16 checkpoint and run
# the reasoning-level sweep on it.
#
# The candidate template is passed with --chat-template on purpose: the stock
# Qwen3.8 template RAISES on reasoning_effort="high" (the OpenAI-standard
# value), so a client sending it gets HTTP 400. qwen3_8_safe_v2.jinja accepts
# it as an xhigh alias. Serving with the candidate is what lets the sweep cover
# all four levels plus "high".
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=$OUT/checkpoint-vllm
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
PORT=18080
cd $REPO

# Stop whatever holds the port, by recorded PID — never pkill -f, which would
# also match this script and any neighbouring job's vllm.
for pidfile in "$OUT/vllm.pid" "$OUT/vllm_bf16.pid"; do
  [ -f "$pidfile" ] || continue
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping pid $pid ($pidfile)"
    kill "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$pidfile"
done
sleep 15

echo "=== serving W4A16 ($CKPT) with the candidate template ==="
nohup $REPO/.venv-vllm/bin/vllm serve "$CKPT" \
  --served-model-name local \
  --max-model-len 32768 \
  --port $PORT \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --chat-template "$TEMPLATE" \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  > "$OUT/logs/vllm_w4a16_reasoning.log" 2>&1 &
pid=$!
echo "$pid" > "$OUT/vllm.pid"

for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "healthy"; break; }
  kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$OUT/logs/vllm_w4a16_reasoning.log"; exit 1; }
  sleep 5
done

echo "=== reasoning sweep ==="
bash "$REPO/scripts/reasoning_sweep.sh" w4a16

#!/usr/bin/env bash
# Re-run the SWE instance on the graft with thinking OFF.
#
# --reasoning-budget is a llama.cpp server extension that vLLM ignores, so the first
# attempt reasoned unbounded: 8096 output tokens in ONE turn, zero tool calls, zero steps.
# --chat-template-kwargs is the portable lever, and thinking-off is the configuration this
# model card already recommends for tool-calling under a token cap.
set -uo pipefail
REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=$OUT/checkpoint-mm-graft
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
PORT=18080
cd $REPO

for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 20
done

echo "=== serving graft ==="
nohup $REPO/.venv-vllm/bin/vllm serve "$CKPT" \
  --served-model-name local --max-model-len 32768 --port $PORT \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --chat-template "$TEMPLATE" --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 > "$OUT/logs/vllm_swe_rerun.log" 2>&1 &
pid=$!; echo "$pid" > "$OUT/vllm.pid"
for _ in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "healthy"; break; }
  kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED"; exit 1; }
  sleep 5
done

echo "=== SWE, thinking off ==="
cd /workspace/swe-mimic
.venv/bin/python run_agent.py \
  --base-url "http://127.0.0.1:$PORT/v1" --model-name local \
  --label GRAFT-NOTHINK \
  --chat-template-kwargs '{"enable_thinking":false}' \
  2>&1 | tee "$OUT/logs/swe_graft_nothink.log" | tail -25

cd $REPO
if [ -f "$OUT/vllm.pid" ]; then p=$(cat "$OUT/vllm.pid"); kill "$p" 2>/dev/null; rm -f "$OUT/vllm.pid"; fi

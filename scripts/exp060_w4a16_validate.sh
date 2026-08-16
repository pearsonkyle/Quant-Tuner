#!/usr/bin/env bash
# End-to-end validation of the exp-060 W4A16 checkpoint, reproducing the metric
# set published on the GGUF ladder card (out/exp-060-32k/release/README.md) so
# the two releases can be read against each other.
#
# Stages are separately runnable:  bash scripts/exp060_w4a16_validate.sh <stage>
#   kld     — 6 eval distributions vs the bf16 reference (PPL / median KLD / top_p)
#   serve   — start vLLM (must be running for the stages below)
#   tools   — 25-session / 174-turn tool-call replay (tool-selection + param acc)
#   swe     — SWE-rebench mimic, one instance (resolved / steps / malformed)
#   needle  — long-context retrieval at ~30k tokens
#   stop    — kill the server
#
# GPU exclusivity: `kld` holds both HF models (~65 GB) and `serve` holds vLLM.
# Never run them at the same time.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=${CKPT:-$OUT/checkpoint-mtp}
REF=$REPO/out/exp-060/model_extracted
CORPORA=$REPO/out/exp-060-32k/corpora
PORT=${PORT:-18080}          # NOT 8080: Jupyter binds 0.0.0.0:8080 on this image
BASE_URL=http://127.0.0.1:$PORT/v1
mkdir -p "$OUT/logs" "$OUT/results"

stage=${1:-all}

kld() {
  echo "=== KLD: 6 distributions vs bf16 reference ==="
  cd $REPO
  PYTHONPATH=src .venv/bin/python scripts/run_hf_kld.py \
    --ref "$REF" --quant "$CKPT" --corpora-dir "$CORPORA" \
    --out "$OUT/results/kld_results.csv" \
    --ctx 8192 --model-class Qwen3_5ForConditionalGeneration \
    2>&1 | tee "$OUT/logs/kld.log" | grep -vE "^  \["
}

serve() {
  echo "=== serving $CKPT on :$PORT ==="
  # --served-model-name local: eval/toolcall.py requests model id "local".
  # --tool-call-parser qwen3_xml: Qwen3.8 emits XML tool calls
  #   (<tool_call><function=NAME><parameter=KEY>), NOT the JSON form. Without
  #   this + --enable-auto-tool-choice, message.tool_calls is always empty and
  #   the tool-call eval scores 0.0 — which looks exactly like a destroyed
  #   quantization but is a serving flag.
  cd $REPO
  nohup .venv-vllm/bin/vllm serve "$CKPT" \
    --served-model-name local \
    --max-model-len 32768 \
    --port $PORT \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --gpu-memory-utilization 0.90 \
    > "$OUT/logs/vllm_server.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/vllm.pid"
  echo "pid $pid — waiting for health"
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "healthy after ${SECONDS}s"; curl -s "http://127.0.0.1:$PORT/v1/models"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "SERVER DIED — tail of log:"; tail -30 "$OUT/logs/vllm_server.log"; return 1
    fi
    sleep 5
  done
  echo "TIMEOUT waiting for health"; tail -30 "$OUT/logs/vllm_server.log"; return 1
}

tools() {
  echo "=== tool-call replay (174 turns) ==="
  cd $REPO
  # Matches the GGUF card: greedy, ctx 32768, --no-stop-on-fail so every model
  # is scored on the identical turn set (stop-on-fail scores a weak model on
  # fewer, easier turns and makes models incomparable).
  PYTHONPATH=src .venv/bin/python scripts/eval_toolcall.py \
    --base-url "$BASE_URL" \
    --holdout "$REPO/out/exp-060-32k/eval/toolcall_holdout.jsonl" \
    --out "$OUT/results/toolcall.csv" \
    --log-dir "$OUT/results/toolcall_logs" \
    --temperature 0 --ctx 32768 --no-stop-on-fail \
    2>&1 | tee "$OUT/logs/toolcall.log" | tail -30
}

swe() {
  echo "=== SWE-rebench mimic (1 instance) ==="
  cd /workspace/swe-mimic
  .venv/bin/python run_agent.py \
    --base-url "$BASE_URL" \
    --model-name local \
    --label W4A16 --reasoning-budget 2048 \
    2>&1 | tee "$OUT/logs/swe.log" | tail -40
}

needle() {
  echo "=== long-context retrieval ~30k ==="
  cd $REPO
  .venv-vllm/bin/python scripts/longctx_needle.py \
    --base-url "$BASE_URL" --model local \
    --haystack "$CORPORA/corpus.eval.broad.txt" \
    --target-tokens 30000 \
    --out "$OUT/results/needle.json" \
    2>&1 | tee "$OUT/logs/needle.log"
}

stop() {
  # Kill by recorded PID, never `pkill -f "vllm serve"` — a pattern that also
  # matches the shell issuing it (bit the GGUF session twice; handoff §8).
  if [ -f "$OUT/vllm.pid" ]; then
    local pid; pid=$(cat "$OUT/vllm.pid")
    kill "$pid" 2>/dev/null && echo "stopped $pid" || echo "pid $pid not running"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null
    rm -f "$OUT/vllm.pid"
  else
    echo "no pid file at $OUT/vllm.pid"
  fi
}

case "$stage" in
  kld) kld ;;
  serve) serve ;;
  tools) tools ;;
  swe) swe ;;
  needle) needle ;;
  stop) stop ;;
  all) kld && serve && tools && swe && needle && stop ;;
  *) echo "unknown stage: $stage"; exit 2 ;;
esac

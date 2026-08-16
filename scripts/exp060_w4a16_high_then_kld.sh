#!/usr/bin/env bash
# Measure the new distinct `high` reasoning rung, then hand the GPU to KLD.
#
# Ordering matters. The running sweep's server loaded the PREVIOUS candidate template,
# in which `high` was an alias of `xhigh` — so `high` cannot be measured without a
# restart. We therefore: wait for the four in-flight levels → restart vLLM on the UPDATED
# template → score `high` on the same 174 turns → only then release the GPU to KLD.
#
# `high` is written to its own CSV, so reasoning_sweep.sh's existence check leaves the
# four completed levels untouched and this stays re-runnable.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=$OUT/checkpoint-vllm
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
REF=$REPO/out/exp-060/model_extracted
CORPORA=$REPO/out/exp-060-32k/corpora
HOLDOUT=$REPO/out/exp-060-32k/eval/toolcall_holdout.jsonl
PORT=18080
cd $REPO

echo "waiting for the four in-flight sweep levels …"
for _ in $(seq 1 720); do
  n=0
  for lvl in xhigh medium low off; do
    [ -f "$OUT/results/toolcall_w4a16_${lvl}.csv" ] && n=$((n+1))
  done
  [ "$n" -eq 4 ] && { echo "sweep complete ($n/4)"; break; }
  sleep 30
done

stop_server() {
  [ -f "$OUT/vllm.pid" ] || return 0
  local pid; pid=$(cat "$OUT/vllm.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping vllm pid $pid"
    kill "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$OUT/vllm.pid"
}

# ---- restart on the UPDATED template so `high` renders as its own rung ----------------
stop_server
sleep 15
echo "=== restarting W4A16 on the updated template (distinct high) ==="
nohup $REPO/.venv-vllm/bin/vllm serve "$CKPT" \
  --served-model-name local \
  --max-model-len 32768 \
  --port $PORT \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --chat-template "$TEMPLATE" \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  > "$OUT/logs/vllm_w4a16_high.log" 2>&1 &
pid=$!
echo "$pid" > "$OUT/vllm.pid"
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "healthy"; break; }
  kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$OUT/logs/vllm_w4a16_high.log"; exit 1; }
  sleep 5
done

# Guard: prove the served template really does treat high as its own rung before
# spending 174 turns on it. If this comes back 400, the wrong template got loaded.
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":4,
       "chat_template_kwargs":{"reasoning_effort":"high"}}')
echo "reasoning_effort=high probe -> HTTP $code"
[ "$code" = "200" ] || { echo "ABORT: served template still rejects high"; exit 1; }

echo "=== scoring reasoning_effort=high (174 turns) ==="
PYTHONPATH=src $REPO/.venv/bin/python $REPO/scripts/eval_toolcall.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --holdout "$HOLDOUT" \
  --out "$OUT/results/toolcall_w4a16_high.csv" \
  --log-dir "$OUT/results/toolcall_w4a16_high_logs" \
  --temperature 0 --ctx 32768 --no-stop-on-fail \
  --chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
  > "$OUT/logs/toolcall_w4a16_high.log" 2>&1
grep -E "Tool selection accuracy|Param accuracy|Schema-valid" "$OUT/logs/toolcall_w4a16_high.log" \
  || echo "   (no result — check log)"

# ---- release the GPU, then KLD -------------------------------------------------------
stop_server
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

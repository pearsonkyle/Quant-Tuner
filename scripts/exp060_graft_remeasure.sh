#!/usr/bin/env bash
# Re-measure every card number on the GRAFTED checkpoint — the artifact we actually ship.
#
# The reasoning ladder, SWE and needle results were collected on the renamed text-only
# serving variant. It carries the same trunk weights, but reaches them through a different
# vLLM code path (text-only Qwen3_5 vs the multimodal wrapper). Rather than footnote that
# on a public card, re-run them here so every published number comes from the exact
# checkpoint in the repo.
#
# KLD is deliberately NOT re-run: the graft HARDLINKS the same model.safetensors that KLD
# measured (verified identical inode), so those numbers are the same bytes by construction.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=$OUT/checkpoint-mm-graft
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
HOLDOUT=$REPO/out/exp-060-32k/eval/toolcall_holdout.jsonl
CORPORA=$REPO/out/exp-060-32k/corpora
PORT=18080
BASE_URL=http://127.0.0.1:$PORT/v1
cd $REPO

stop_server() {
  [ -f "$OUT/vllm.pid" ] || return 0
  local pid; pid=$(cat "$OUT/vllm.pid")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$OUT/vllm.pid"
}

echo "waiting for the graft gates to finish …"
for _ in $(seq 1 240); do
  [ -f "$OUT/results/graft_validation.json" ] && { echo "gates done"; break; }
  sleep 30
done
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 20
done

# Archive the text-only-variant results so the fresh run cannot be confused with them and
# reasoning_sweep.sh's existence check does not skip every level.
ARCHIVE=$OUT/results/text-only-variant
mkdir -p "$ARCHIVE"
for f in "$OUT"/results/toolcall_w4a16_*.csv; do
  [ -e "$f" ] && mv "$f" "$ARCHIVE/" && echo "archived $(basename "$f")"
done
[ -e "$OUT/results/needle.json" ] && mv "$OUT/results/needle.json" "$ARCHIVE/"

echo "=== serving the graft ==="
stop_server; sleep 10
nohup $REPO/.venv-vllm/bin/vllm serve "$CKPT" \
  --served-model-name local --max-model-len 32768 --port $PORT \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --chat-template "$TEMPLATE" --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 > "$OUT/logs/vllm_remeasure.log" 2>&1 &
pid=$!; echo "$pid" > "$OUT/vllm.pid"
for _ in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "healthy"; break; }
  kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED"; tail -30 "$OUT/logs/vllm_remeasure.log"; exit 1; }
  sleep 5
done

echo
echo "=== needle @ ~30k ==="
$REPO/.venv-vllm/bin/python scripts/longctx_needle.py \
  --base-url "$BASE_URL" --model local \
  --haystack "$CORPORA/corpus.eval.broad.txt" \
  --target-tokens 30000 \
  --out "$OUT/results/needle.json" 2>&1 | tee "$OUT/logs/needle.log" | tail -6

echo
echo "=== reasoning ladder (5 levels incl. high) ==="
run_level() {
  local name=$1 kwargs=$2
  local csv="$OUT/results/toolcall_w4a16_${name}.csv"
  [ -f "$csv" ] && { echo "== $name already done"; return; }
  echo "== $name"
  PYTHONPATH=src "$REPO/.venv/bin/python" "$REPO/scripts/eval_toolcall.py" \
    --base-url "$BASE_URL" --holdout "$HOLDOUT" --out "$csv" \
    --log-dir "$OUT/results/toolcall_w4a16_${name}_logs" \
    --temperature 0 --ctx 32768 --no-stop-on-fail \
    --chat-template-kwargs "$kwargs" \
    > "$OUT/logs/toolcall_w4a16_${name}.log" 2>&1
  grep -E "Tool selection accuracy|Param accuracy|Schema-valid" \
    "$OUT/logs/toolcall_w4a16_${name}.log" || echo "   (no result — check log)"
}
run_level "xhigh"  '{"enable_thinking":true,"reasoning_effort":"xhigh"}'
run_level "high"   '{"enable_thinking":true,"reasoning_effort":"high"}'
run_level "medium" '{"enable_thinking":true,"reasoning_effort":"medium"}'
run_level "low"    '{"enable_thinking":true,"reasoning_effort":"low"}'
# `off` is Gate 3's run on this same checkpoint — copy it rather than paying for it twice.
if [ -f "$OUT/results/toolcall_graft_off.csv" ]; then
  cp "$OUT/results/toolcall_graft_off.csv" "$OUT/results/toolcall_w4a16_off.csv"
  echo "== off  (reused Gate 3 result on this checkpoint)"
fi

echo
echo "=== SWE-rebench mimic (1 instance) ==="
if [ -d /workspace/swe-mimic ]; then
  cd /workspace/swe-mimic
  .venv/bin/python run_agent.py \
    --base-url "$BASE_URL" --model-name local \
    --label GRAFT --reasoning-budget 2048 \
    2>&1 | tee "$OUT/logs/swe_graft.log" | tail -25
  cd $REPO
else
  echo "SKIPPED — /workspace/swe-mimic not present"
fi

stop_server
echo
echo "=== re-measure complete ==="
PYTHONPATH=src $REPO/.venv/bin/python scripts/summarize_reasoning_ladder.py

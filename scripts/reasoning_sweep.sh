#!/usr/bin/env bash
# Sweep Qwen3.8's reasoning knob against the tool-call holdout on a running
# OpenAI-compatible server.
#
# Why this matters beyond curiosity: the reasoning level competes with the
# answer for the SAME token budget. At --max-tokens 512 with the default
# reasoning_effort=xhigh, the model is explicitly told to "think carefully,
# validate key assumptions, consider plausible alternatives" and then gets
# truncated before it can emit a <tool_call> — which the scorer records as
# "chose not to call a tool". That is a serving-config artifact, not model
# quality, and this sweep quantifies it.
#
# Levels (verified against the shipped template):
#   xhigh  (default) - injects the "think carefully..." instruction
#   medium           - injects NOTHING; the model's native reasoning
#   low              - injects "keep your thinking brief and focused"
#   off              - enable_thinking=false; prompt ships a pre-closed
#                      <think></think>, so generation starts at the answer
#   high             - OpenAI-standard value; RAISES on the stock template,
#                      accepted as an xhigh alias by qwen3_8_safe_v2.jinja
#
#   bash scripts/reasoning_sweep.sh <label> [base_url]
set -uo pipefail

LABEL=${1:-w4a16}
BASE_URL=${2:-http://127.0.0.1:18080/v1}
REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
HOLDOUT=$REPO/out/exp-060-32k/eval/toolcall_holdout.jsonl
mkdir -p "$OUT/results" "$OUT/logs"

run() {
  local name=$1 kwargs=$2
  local csv="$OUT/results/toolcall_${LABEL}_${name}.csv"
  if [ -f "$csv" ]; then echo "== $name: already done, skipping"; return; fi
  echo "== $name  ($kwargs)"
  PYTHONPATH=src "$REPO/.venv/bin/python" "$REPO/scripts/eval_toolcall.py" \
    --base-url "$BASE_URL" \
    --holdout "$HOLDOUT" \
    --out "$csv" \
    --log-dir "$OUT/results/toolcall_${LABEL}_${name}_logs" \
    --temperature 0 --ctx 32768 --no-stop-on-fail \
    --chat-template-kwargs "$kwargs" \
    > "$OUT/logs/toolcall_${LABEL}_${name}.log" 2>&1
  grep -E "Tool selection accuracy|Param accuracy|Schema-valid" \
    "$OUT/logs/toolcall_${LABEL}_${name}.log" || echo "   (no result — check log)"
}

cd "$REPO"
run "xhigh"  '{"enable_thinking":true,"reasoning_effort":"xhigh"}'
run "medium" '{"enable_thinking":true,"reasoning_effort":"medium"}'
run "low"    '{"enable_thinking":true,"reasoning_effort":"low"}'
run "off"    '{"enable_thinking":false}'

echo
echo "=== summary: $LABEL ==="
printf '%-8s %-10s %-10s %-10s\n' level sel_acc param_acc schema
for lvl in xhigh medium low off; do
  f="$OUT/results/toolcall_${LABEL}_${lvl}.csv"
  [ -f "$f" ] && awk -F, -v L="$lvl" 'NR==2{printf "%-8s %-10.3f %-10.3f %-10.3f\n", L, $3, $4, $5}' "$f"
done

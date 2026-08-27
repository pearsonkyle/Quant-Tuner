#!/usr/bin/env bash
# Agent bench (Docker-free SWE-rebench mimic) across the exp-060 ladder.
#
# One llama-server per model, one agent episode per model, graded by actually running the
# instance's FAIL_TO_PASS / PASS_TO_PASS tests. See run_agent.py for why this is a smoke
# test and not the real benchmark.
#
# --jinja is REQUIRED: without it llama-server won't apply the chat template's tool-calling
# path, and the Agents SDK's `tools=` payload silently produces no tool calls.
set -uo pipefail
cd "$(dirname "$0")"

Q=/workspace/Quant-Tuner
BIN=${BIN:-$Q/vendor/llama.cpp/build/bin/llama-server}
# 18080, not 8080: the instance's Jupyter binds 0.0.0.0:8080, so a server started there
# comes up "healthy" against the wrong process and every agent episode dies.
PORT=${PORT:-18080}
CTX=${CTX:-32768}
MAXTURNS=${MAXTURNS:-60}
# Per-turn <think> cap, enforced by llama.cpp's reasoning-budget sampler (not by truncating
# the response). Set RBUDGET=-1 for unrestricted thinking to A/B against this.
RBUDGET=${RBUDGET:-2048}
# Suffix for the results CSV so a budgeted and an unbudgeted sweep don't append into the
# same file under the same labels.
TAG=${TAG:-}

# Default = the exp-060 ladder. Override with a file of "LABEL /path/to.gguf"
# lines (MODELS_FILE=...) to sweep a different set — e.g. the exp-062 AWQ rungs
# against their shipped controls — without forking this script and drifting from
# the harness settings the recorded numbers were produced under.
declare -a MODELS=(
  "IQ2_M   $Q/out/exp-060-32k/iq2_m/Qwen3.8-27B-IQ2_M.gguf"
  "IQ3_M   $Q/out/exp-060-32k/iq3_m/Qwen3.8-27B-IQ3_M.gguf"
  "IQ4_XS  $Q/out/exp-060-32k/iq4_xs/Qwen3.8-27B-IQ4_XS.gguf"
  "IQ4_NL  $Q/out/exp-060-32k/iq4_nl/Qwen3.8-27B-IQ4_NL.gguf"
  "Q5_K_M  $Q/out/exp-060-32k/q5_k_m/Qwen3.8-27B-Q5_K_M.gguf"
  "F16     $Q/out/exp-060/model-f16.gguf"
)
if [ -n "${MODELS_FILE:-}" ]; then
  [ -f "$MODELS_FILE" ] || { echo "FATAL: MODELS_FILE not found: $MODELS_FILE" >&2; exit 1; }
  MODELS=()
  while read -r line; do
    [ -z "${line// /}" ] && continue
    case "$line" in \#*) continue ;; esac
    MODELS+=("$line")
  done < "$MODELS_FILE"
  echo "model list: $MODELS_FILE (${#MODELS[@]} entries)"
fi

OUT=${OUT:-swe_mimic_results${TAG}.csv}
# extra llama-server flags (e.g. sampling penalties). llama-server CLI sampling params
# are the DEFAULTS for /v1 requests that omit those fields — the agents SDK sends only
# temperature/top_p, so --repeat-penalty/--presence-penalty set here take effect.
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
echo "reasoning budget: $RBUDGET | results -> $OUT"

for entry in "${MODELS[@]}"; do
  read -r LABEL PATH_ <<<"$entry"
  if [ ! -f "$PATH_" ]; then echo "SKIP (missing): $LABEL"; continue; fi

  echo "############ $LABEL ############"
  "$BIN" --model "$PATH_" --ctx-size "$CTX" --n-gpu-layers 999 --jinja \
         --flash-attn on --host 127.0.0.1 --port "$PORT" $EXTRA_SERVER_ARGS \
         > "logs_server_$LABEL.log" 2>&1 &
  SRV=$!

  # wait for health (model load on a 50 GiB F16 is slow)
  ok=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "server died early: $LABEL"; break; fi
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "server never became healthy: $LABEL"; tail -5 "logs_server_$LABEL.log"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; continue
  fi

  .venv/bin/python run_agent.py \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model-name "$LABEL" --label "$LABEL" \
      --max-turns "$MAXTURNS" --reasoning-budget "$RBUDGET" \
      --temperature "${TEMP:-0.25}" \
      --out "$OUT" 2>&1 | tail -25
  echo "agent exit: $?"

  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
done

echo "=== AGENT BENCH COMPLETE ==="
cat "$OUT" 2>/dev/null

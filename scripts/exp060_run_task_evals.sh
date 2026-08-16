#!/usr/bin/env bash
# Task-level evals for the exp-060 ladder: tool-call accuracy + MMLU-Pro, per rung.
#
# SWE-rebench (the repo's real agentic benchmark) cannot run on this box: it needs one
# Docker container per instance, and this is an unprivileged container with no container
# runtime, no docker.sock, no CAP_SYS_ADMIN, and nested user namespaces refused. These two
# evals are the Docker-free substitutes that still exercise the properties we calibrated
# for: emitting correct tool calls against real schemas, and coding/math reasoning.
#
# --no-stop-on-fail is deliberate. With the default stop-on-fail a weak rung halts early
# and is scored on FEWER, EASIER turns than a strong one, so the models are not compared on
# the same work. Disabling it gives every model the identical turn set.
#
# Greedy (temperature 0) for the same reason: we want quant-vs-quant differences, not
# sampling noise.
set -uo pipefail

cd "$(dirname "$0")/.."
RUN=${RUN:-exp-060-32k}
EVAL_DIR=out/$RUN/eval
CTX=${CTX:-32768}
mkdir -p "$EVAL_DIR/logs"

# MMLU-Pro is off by default: it is no longer reported on the model card, and it costs a
# full server load + generation pass per rung. RUN_MMLU=1 re-enables it.
RUN_MMLU=${RUN_MMLU:-0}

F16=out/exp-060/model-f16.gguf
# Space-separated MODELS env overrides the list, so a newly added rung can be scored
# without re-running the four already in toolcall_results.csv.
if [ -n "${MODELS:-}" ]; then
  read -r -a MODELS <<<"$MODELS"
else
  MODELS=(
    "$F16"
    out/$RUN/iq2_m/Qwen3.8-27B-IQ2_M.gguf
    out/$RUN/iq3_m/Qwen3.8-27B-IQ3_M.gguf
    out/$RUN/iq4_nl/Qwen3.8-27B-IQ4_NL.gguf
    out/$RUN/q5_k_m/Qwen3.8-27B-Q5_K_M.gguf
  )
fi

for m in "${MODELS[@]}"; do
  name=$(basename "$m")
  if [ ! -f "$m" ]; then echo "SKIP (missing): $m"; continue; fi

  echo "=== toolcall: $name ==="
  PYTHONPATH=src .venv/bin/python scripts/eval_toolcall.py \
    --model "$m" \
    --holdout "$EVAL_DIR/toolcall_holdout.jsonl" \
    --out "$EVAL_DIR/toolcall_results.csv" \
    --log-dir "$EVAL_DIR/logs" \
    --temperature 0 --ctx "$CTX" --ngl 99 --no-stop-on-fail \
    2>&1 | tail -14
  echo "toolcall exit: $?"

  [ "$RUN_MMLU" = "1" ] || continue

  echo "=== mmlu_pro: $name ==="
  PYTHONPATH=src .venv/bin/python scripts/eval_mmlu_pro.py \
    --model "$m" \
    --holdout "$EVAL_DIR/mmlu_pro_holdout.json" \
    --out "$EVAL_DIR/mmlu_pro_results.csv" \
    --log-dir "$EVAL_DIR/logs" \
    --temperature 0 --ctx "$CTX" --ngl 99 \
    2>&1 | tail -10
  echo "mmlu exit: $?"
done

echo "=== TASK EVALS COMPLETE ==="
echo "  toolcall: $EVAL_DIR/toolcall_results.csv"
echo "  mmlu_pro: $EVAL_DIR/mmlu_pro_results.csv"

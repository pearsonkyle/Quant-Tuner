#!/usr/bin/env bash
# Tool-call replay across the GGUF ladder, scored the same way for every rung.
#
# WHY THE bf16 GGUF IS AN ARM. The published bf16 numbers came from the
# in-process HF harness (eval_toolcall_local.py); these come through
# llama-server. Those are different stacks and their numbers do not compare --
# a reference model can score differently on each for reasons that have nothing
# to do with quantization. So the bf16 GGUF is scored here too, through the
# SAME server, and every quant is read against THAT, not against the HF number.
#
# THE FLAGS ARE THE MEASUREMENT. Three defaults in eval_toolcall.py would
# silently make these incomparable with the published table:
#
#   --no-stop-on-fail   the default STOPS a session at its first wrong tool.
#                       That is how an earlier run scored 32 and 70 turns
#                       instead of 107 and had to be excluded from the
#                       appendix: a rate over a denominator no other arm
#                       shares. Required.
#   --ctx 65536         default is 8192. Our sessions are longer; truncation
#                       is the single most likely cause of a spurious zero.
#   --max-tokens 2048   default is 512, which the published table did not use.
#
# --chat-template-kwargs is passed EMPTY on purpose: its default
# ('{"enable_thinking":false}') targets Qwen3, and Gemma 4's template already
# defaults enable_thinking to false. Omitting it reproduces exactly what the HF
# harness did rather than injecting a kwarg that harness never sent.
set -euo pipefail

REPO=/workspace/Quant-Tuner
PY=$REPO/.venv/bin/python
LADDER=$REPO/out/e4b-v65536/gguf-ladder
HOLDOUT=$REPO/out/e4b-v65536/eval/toolcall_holdout_quick.jsonl
OUT=$LADDER/toolcall.csv
BF16=/workspace/models/gguf-e4b-coder/gemma4-e4b-coder-BF16.gguf

[ -f "$HOLDOUT" ] || { echo "no holdout at $HOLDOUT"; exit 1; }
mkdir -p "$LADDER/toolcall-logs"

run_one() {
    local name=$1 model=$2
    [ -f "$model" ] || { echo "[toolcall] skip $name -- no file at $model"; return 0; }
    if grep -q "$name" "$OUT" 2>/dev/null; then
        echo "[toolcall] $name already scored -- skipping"; return 0
    fi
    echo "[toolcall] === $name ==="
    PYTHONPATH=$REPO/src "$PY" "$REPO/scripts/eval_toolcall.py" \
        --model "$model" \
        --holdout "$HOLDOUT" \
        --out "$OUT" \
        --log-dir "$LADDER/toolcall-logs/$name" \
        --max-turns-per-session 3 \
        --max-tokens 2048 \
        --ctx 65536 \
        --ngl 99 \
        --temperature 0 \
        --no-stop-on-fail \
        --chat-template-kwargs "" \
        || echo "[toolcall] $name FAILED"
}

run_one "BF16"   "$BF16"
run_one "IQ4_XS" "$LADDER/IQ4_XS/gemma4-e4b-coder-IQ4_XS.gguf"
run_one "IQ3_M"  "$LADDER/IQ3_M/gemma4-e4b-coder-IQ3_M.gguf"
run_one "IQ2_M"  "$LADDER/IQ2_M/gemma4-e4b-coder-IQ2_M.gguf"

echo
echo "=== results: $OUT ==="
[ -f "$OUT" ] && column -s, -t "$OUT"

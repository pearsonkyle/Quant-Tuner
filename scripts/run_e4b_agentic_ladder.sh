#!/usr/bin/env bash
# Small agentic benchmark across the gemma4-e4b-coder quant ladder.
#
# Answers the two questions a quant can fail on that KLD cannot see:
#   * does it still call tools correctly?   -> tool_error_rate, patch_rate
#   * does it get WEDGED?                   -> step_limit_rate
#
# step_limit_rate is the loop signal and it is why pass_rate alone is not enough:
# an agent stuck alternating two commands does not error and does not crash, it
# spends every step and submits nothing -- scored identically to a model that
# tried once and gave up. Only the first is a serving problem you can fix with
# sampling (which is why the harness defaults to temperature 0.25, not greedy).
#
# REQUIRES DOCKER. Each instance runs in its own SWE-rebench container, so this
# cannot run on an unprivileged box (a Vast.ai instance IS a container; there is
# no daemon inside it). Run it where a daemon exists; everything up to this point
# -- quantizing, KLD, tool-call eval -- does not need one.
#
#   ./scripts/run_e4b_agentic_ladder.sh <gguf-dir> [n_instances]
set -euo pipefail

GGUF_DIR="${1:?usage: run_e4b_agentic_ladder.sh <gguf-dir> [n_instances]}"
N="${2:-10}"
WS="out/e4b-agentic-ladder"
HOLDOUT="$WS/rebench_v2_${N}.jsonl"

mkdir -p "$WS"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: no Docker daemon. SWE-rebench runs each instance in a container." >&2
    exit 1
fi

# V2 spans 20 languages; V1 is Python-only. Balanced sampling keeps one language
# or repo from dominating a 10-instance run, where a single family would swamp it.
if [ ! -f "$HOLDOUT" ]; then
    uv run python scripts/build_swebench_holdout.py \
        --dataset nebius/SWE-rebench-V2 --split test \
        --n "$N" --seed 42 --out "$HOLDOUT"
fi

# The bf16 reference goes through the SAME harness. Cross-stack numbers are not
# comparable, so a quant is only ever scored against a reference served the same way.
for MODEL in "$GGUF_DIR"/*.gguf; do
    LABEL="$(basename "$MODEL" .gguf)"
    echo "=== $LABEL ==="
    uv run python scripts/run_swebench_eval.py \
        --models "$MODEL" \
        --holdout "$HOLDOUT" \
        --workspace "$WS/$LABEL" \
        --agent openai-agents \
        --max-steps 100 \
        --ctx 32768 \
        --temperature 0.25 \
        --reps 1
done

echo
echo "Per-model summaries in $WS/*/ ; the columns that matter here are"
echo "  pass_rate  patch_rate  tool_error_rate  step_limit_rate  mean_steps"

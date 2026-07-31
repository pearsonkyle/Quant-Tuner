#!/usr/bin/env bash
# Red-team comparison: gemma-4-moe vs gemma-4-31b, judged + simulated by minimax-m2-7.
#
# All three are remote vLLM endpoints on the same network, so the targets are
# passed via --base-url (no local llama-server is spawned). Both targets append
# to the same results.csv, so the two rows sit side-by-side for comparison.
#
# Requires the redteam extra:  uv sync --extra redteam
#
# Usage:
#   scripts/redteam_compare.sh                       # minimal config (default)
#   CONFIG=red_team scripts/redteam_compare.sh       # extensive sweep
#   ATTACKS_PER_TYPE=3 scripts/redteam_compare.sh    # override prompts per type
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Endpoints ────────────────────────────────────────────────────────────────
JUDGE_MODEL="minimax-m2-7"
JUDGE_URL="http://10.132.212.37:10200/v1"

# Simulator shares the judge endpoint (a strong model writes + grades attacks).
SIM_MODEL="$JUDGE_MODEL"
SIM_URL="$JUDGE_URL"

# Targets under comparison: "served-model-name|base-url"
TARGETS=(
  "gemma-4-moe|http://10.132.212.38:10200/v1"
  "gemma-4-31b|http://10.132.212.38:10201/v1"
)

# ── Knobs ────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-red_team_minimal}"
OUT_DIR="${OUT_DIR:-out/redteam}"
RESULTS="${RESULTS:-$OUT_DIR/compare_results.csv}"
ATTACKS_PER_TYPE="${ATTACKS_PER_TYPE:-}"   # empty → use the config's value

# How to invoke Python. Locally, "uv run python" manages the project venv. In
# the slim red-team container the deps are already system-installed (with
# PYTHONPATH=/project/src), so set PY="python" to skip uv entirely.
PY="${PY:-uv run python}"

mkdir -p "$OUT_DIR"

extra_args=()
[[ -n "$ATTACKS_PER_TYPE" ]] && extra_args+=(--attacks-per-type "$ATTACKS_PER_TYPE")

echo "==================================================================="
echo " Red-team comparison"
echo "   config    : $CONFIG"
echo "   judge     : $JUDGE_MODEL ($JUDGE_URL)"
echo "   simulator : $SIM_MODEL ($SIM_URL)"
echo "   results   : $RESULTS"
echo "   targets   : ${#TARGETS[@]}"
echo "==================================================================="

for entry in "${TARGETS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  echo ""
  echo ">>> Target: $name ($url)"
  $PY scripts/eval_redteam.py \
    --base-url "$url" \
    --target-model-name "$name" \
    --target-purpose "General-purpose instruction-following assistant" \
    --config "$CONFIG" \
    --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_URL" \
    --simulator-model "$SIM_MODEL" --simulator-base-url "$SIM_URL" \
    --out "$RESULTS" \
    --json-out "$OUT_DIR/${name}_$(date +%Y%m%d_%H%M%S).json" \
    "${extra_args[@]}"
done

echo ""
echo "==================================================================="
echo " Comparison table ($RESULTS):"
echo "==================================================================="
column -t -s, "$RESULTS" 2>/dev/null || cat "$RESULTS"

#!/usr/bin/env bash
# Reproduce the red-team safety sweep from scratch.
#
# What it does, end to end:
#   1. builds a Python <=3.12 venv (.venv-redteam) with the redteam extra, because
#      deepteam 1.0.7 imports `nntplib` (gone in 3.13) and the repo's main .venv is 3.13;
#   2. runs scripts/eval_redteam.py over one or more TARGET models served on a local
#      OpenAI-compatible endpoint, on ONE frozen attack bank, with a separate
#      (uncensored) model as simulator + judge;
#   3. if >1 target, pairs every rung against the first via scripts/redteam_ladder.py.
#
# The judge/simulator MUST be a separate, uncensored model — a safety-tuned model
# refuses to author attacks, and a target judging its own jailbreaks is worthless.
# Everything runs on hardware you control; nothing is sent to a hosted provider.
#
# This is authorized defensive testing: the disclosure_*.json artifacts it writes
# (full attack + response + the model's own reasoning + judge verdict for every
# NON-refusal) exist to be reported to a model's authors so they can harden it.
#
# Usage (all overridable via env):
#   scripts/reproduce_redteam.sh
#   TARGETS="ornith-1.0-35b" scripts/reproduce_redteam.sh          # single model
#   CONFIG=red_team_minimal MAX_TOKENS=1500 scripts/reproduce_redteam.sh
#
# The defaults reproduce the 2026-07-31 ornith-1.0-35b run (docs/benchmarks.md).
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config (override via env) ────────────────────────────────────────────────
VENV="${VENV:-.venv-redteam}"
PYVER="${PYVER:-3.12}"
CONFIG="${CONFIG:-red_team_broad}"
# Space-separated served model ids on ONE --base-url. Pass the reference FIRST —
# the frozen bank is simulated against it. The 2026-07-31 run used only ornith.
TARGETS="${TARGETS:-ornith-1.0-35b}"
TARGET_URL="${TARGET_URL:-http://localhost:1234/v1}"

JUDGE_MODEL="${JUDGE_MODEL:-Qwopus3.6-27B-uncensored-Q5_K_M}"
JUDGE_URL="${JUDGE_URL:-http://100.102.53.29:1234/v1}"
SIM_MODEL="${SIM_MODEL:-$JUDGE_MODEL}"
SIM_URL="${SIM_URL:-$JUDGE_URL}"

# Reasoning targets (Ornith/Qwen3/DeepSeek) spend the budget on chain-of-thought
# FIRST; too low and the answer is empty and mis-scored as "safe". 4000 is a safe
# floor. Timeouts are generous because the target box may be shared.
MAX_TOKENS="${MAX_TOKENS:-4000}"
TARGET_TIMEOUT="${TARGET_TIMEOUT:-1200}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-1200}"
OUTDIR="${OUTDIR:-out/redteam/$CONFIG}"

# ── 1. venv ──────────────────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
  echo "[repro] creating $VENV (Python $PYVER) with the redteam extra"
  uv venv --python "$PYVER" "$VENV"
  uv pip install --python "$VENV/bin/python" 'deepteam>=1.0.7' 'deepeval>=3.6.2' pyyaml requests openai
fi
"$VENV/bin/python" -c "import deepteam" 2>/dev/null \
  || { echo "[repro] deepteam import failed — is $VENV on Python <=3.12?"; exit 1; }

# ── 2. sweep ─────────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
TARGET_ARGS=(); for m in $TARGETS; do TARGET_ARGS+=(--target-model-name "$m"); done

echo "[repro] sweeping [$TARGETS] on $CONFIG (bank seeded on first target)"
PYTHONPATH=src PYTHONUNBUFFERED=1 "$VENV/bin/python" -u scripts/eval_redteam.py \
  --base-url "$TARGET_URL" "${TARGET_ARGS[@]}" \
  --config "$CONFIG" \
  --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_URL" \
  --simulator-model "$SIM_MODEL" --simulator-base-url "$SIM_URL" \
  --remote-no-think \
  --target-max-tokens "$MAX_TOKENS" \
  --target-timeout "$TARGET_TIMEOUT" --remote-timeout "$REMOTE_TIMEOUT" \
  --frozen-bank --bank-out "$OUTDIR/bank.json" \
  --out "$OUTDIR/results.csv" --json-dir "$OUTDIR"

# ── 3. paired ladder (only meaningful with >1 target) ────────────────────────
n_targets=$(echo "$TARGETS" | wc -w | tr -d ' ')
if [ "$n_targets" -gt 1 ]; then
  ref=$(echo "$TARGETS" | awk '{print $1}')
  echo "[repro] pairing rungs against reference '$ref'"
  PYTHONPATH=src "$VENV/bin/python" scripts/redteam_ladder.py \
    --per-case "$OUTDIR/results_per_case.csv" --reference "$ref" \
    --workspace "$OUTDIR/ladder"
fi

echo
echo "[repro] done. Disclosure evidence: $OUTDIR/disclosure_*.json"
echo "        Per-case CSV          : $OUTDIR/results_per_case.csv"

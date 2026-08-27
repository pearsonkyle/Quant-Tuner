#!/usr/bin/env bash
# Benchmark one ternary QAT export as an agent: can it actually patch a real repo?
#
#     bash scripts/run_swe_mimic.sh TAG [GGUF]
#     bash scripts/run_swe_mimic.sh sft32k_sw1
#
# Wraps /workspace/swe-mimic — the Docker-free SWE-rebench mimic on dask__dask-11393 —
# so a tagged run's agentic result lands in the same CSV as every other tag's, under the
# label the run registry joins on (TAG uppercased). One instance, one episode: this is a
# smoke test for "does the model function as an agent at all", not a pass-rate benchmark.
#
# WHY A WRAPPER RATHER THAN CALLING THE SWEEP DIRECTLY. The harness lives outside this
# repo and is NOT under version control, so nothing otherwise records which version of it
# produced a number. This records a fingerprint of run_agent.py alongside each result; if
# it changes between two runs, their rows are not comparable and the fingerprint says so.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

TAG="${1:?usage: run_swe_mimic.sh TAG [GGUF]}"
GGUF="${2:-out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf}"
[ -f "$GGUF" ] || { echo "[swe] no GGUF at $GGUF — export first"; exit 1; }

MIMIC="${MIMIC:-/workspace/swe-mimic}"
[ -d "$MIMIC" ] || { echo "[swe] harness not found at $MIMIC"; exit 1; }
OUT_CSV="${OUT_CSV:-$MIMIC/swe_mimic_ternary.csv}"
LABEL="${LABEL:-$(echo "$TAG" | tr '[:lower:]' '[:upper:]')-Q2_0}"

# The GPU must be free: the agent needs the whole card for a 32k-context server, and a
# leaked trainer makes a model look incapable when it merely never got to run.
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${BUSY:-0}" -gt 4096 ]; then
    echo "[swe] GPU busy (${BUSY} MiB) — refusing to benchmark; a starved server reads as a bad model"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
fi

FP=$(sha256sum "$MIMIC/run_agent.py" | cut -c1-12)
echo "[swe] tag=$TAG label=$LABEL harness=$FP"
echo "[swe] gguf=$GGUF"

MODELS_FILE=$(mktemp)
# Absolute path: the sweep cd's into its own directory.
echo "$LABEL $REPO/$GGUF" > "$MODELS_FILE"
trap 'rm -f "$MODELS_FILE"' EXIT

cd "$MIMIC"
# Q2_0 is ftype 41, which exists only in the prism fork — mainline llama-server cannot
# read these GGUFs at all. Pin the fork's binary explicitly rather than inheriting the
# sweep's mainline default.
BIN_OVERRIDE="$REPO/vendor/llama.cpp-prism/build/bin/llama-server"
[ -f "$BIN_OVERRIDE" ] || {
    echo "[swe] no prism llama-server at $BIN_OVERRIDE — a Q2_0 GGUF cannot load"; exit 1; }

BIN="$BIN_OVERRIDE" MODELS_FILE="$MODELS_FILE" OUT="$OUT_CSV" TAG="" \
    bash run_all_quants.sh 2>&1 | tail -40

cd "$REPO"
echo "[swe] harness fingerprint $FP recorded for $LABEL"
printf '%s,%s,%s\n' "$LABEL" "$FP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> out/exp-058/eval/swe_harness_fingerprints.csv

echo "[swe] anomaly analysis"
.venv/bin/python scripts/analyze_swe_anomalies.py --label "$LABEL" \
    --out "out/exp-058/eval/swe_anomalies_${TAG}.json" || true

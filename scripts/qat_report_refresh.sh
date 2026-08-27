#!/usr/bin/env bash
# Regenerate a QAT run's HTML report from whatever the run has produced so far.
#
#     bash scripts/qat_report_refresh.sh out/exp-058/trained_sft32k_sw1 "sft32k_sw1 — ternary QAT"
#
# Safe to run beside training, and idempotent — call it on a timer or at checkpoints. The
# chain is parse-log -> census -> report; this script exists because the step-0 census is
# the expensive link (it reads the shipped weights) and is CONSTANT for every run started
# from the same base model, so it is cached and never recomputed.
#
# All three stages are CPU-only and touch no GPU memory: the census reads only the tracked
# tensors named in flips.csv, not the whole model.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: qat_report_refresh.sh RUN_DIR [TITLE]}"
RUN="${RUN%/}"
TITLE="${2:-$(basename "$RUN") — ternary QAT}"

PY=${PY:-.venv/bin/python}
BASE_MODEL="${BASE_MODEL:-out/exp-057/model}"
TEL="$RUN/telemetry"
# Shared across runs from the same base — see the note above.
CENSUS0="${CENSUS0:-out/exp-058/census_step0.csv}"
STOP_CSV="${STOP_CSV:-out/exp-058/eval/stop_prob.csv}"
NOTES="${NOTES:-$RUN/notes.txt}"
OUT="${OUT:-$RUN/report.html}"

# Window/accum are recorded in the report header and used for the tokens/step figure; read
# them off the live process rather than hardcoding, so a report never mislabels its own run.
WINDOW="${WINDOW:-}"
ACCUM="${ACCUM:-}"
if [ -z "$WINDOW" ]; then
    WINDOW=$(sed 's/\r/\n/g' "$RUN/train.log" 2>/dev/null \
        | sed -n 's/.*windows x \([0-9]\+\).*/\1/p' | head -1)
fi
WINDOW="${WINDOW:-32768}"
if [ -z "$ACCUM" ]; then
    ACCUM=$(sed 's/\r/\n/g' "$RUN/train.log" 2>/dev/null \
        | sed -n 's/.*@ accum \([0-9]\+\).*/\1/p' | head -1)
fi
ACCUM="${ACCUM:-1}"

mkdir -p "$TEL"

# 1. log -> steps.csv / val.csv / flips.csv / summary.json
$PY scripts/parse_qat_log.py "$RUN/train.log" --out "$TEL" >/dev/null

if [ ! -s "$TEL/flips.csv" ]; then
    echo "[refresh] no flip rows yet (first checkpoint not reached) — report will be curves only"
fi

CENSUS_ARGS=()
if [ -s "$TEL/flips.csv" ]; then
    # 2. step-0 census, cached: the shipped codes never change.
    if [ ! -s "$CENSUS0" ]; then
        echo "[refresh] building step-0 census (once) -> $CENSUS0"
        mkdir -p "$(dirname "$CENSUS0")"
        $PY scripts/ternary_distribution.py census --model "$BASE_MODEL" \
            --tensors "$TEL/flips.csv" --out "$CENSUS0" >/dev/null
    fi
    CENSUS_ARGS+=(--census "$CENSUS0")

    # 3. latest census from the newest checkpoint, if one exists.
    CKPT=""
    for cand in "$RUN/trained_latents.pt" $(ls -1t "$RUN"/ckpt_*.pt 2>/dev/null); do
        [ -f "$cand" ] && { CKPT="$cand"; break; }
    done
    if [ -n "$CKPT" ]; then
        LATEST_STEP=$(python3 -c "
import json,sys
last=0
try:
    for line in open('$RUN/metrics.jsonl'):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except ValueError: continue
        if d.get('kind')=='step': last=d.get('step',last)
except OSError: pass
print(last)")
        $PY scripts/ternary_distribution.py census --latents "$CKPT" \
            --tensors "$TEL/flips.csv" --out "$TEL/census_latest.csv" >/dev/null
        CENSUS_ARGS+=(--latest "$TEL/census_latest.csv" --latest-step "$LATEST_STEP")
    fi
fi

EXTRA=()
[ -s "$STOP_CSV" ] && EXTRA+=(--stop-prob-csv "$STOP_CSV")
[ -s "$NOTES" ] && EXTRA+=(--notes "$NOTES")
[ -f "$BASE_MODEL/config.json" ] && EXTRA+=(--model-config "$BASE_MODEL/config.json")

PYTHONPATH=scripts $PY scripts/qat_report.py \
    --telemetry "$TEL" "${CENSUS_ARGS[@]}" "${EXTRA[@]}" \
    --window "$WINDOW" --grad-accum "$ACCUM" \
    --title "$TITLE" --out "$OUT"

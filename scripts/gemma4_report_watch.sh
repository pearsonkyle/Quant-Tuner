#!/usr/bin/env bash
# Keep out/gemma4-ternary/report.html current while the study runs.
#
#     bash scripts/gemma4_report_watch.sh [interval_s]
#
# Pure reading — parses JSON artifacts and train.logs, renders SVG. No model load, no
# GPU, safe beside a live trainer. Runs until killed.
set -uo pipefail
cd "$(dirname "$0")/.."
INT="${1:-300}"
while true; do
    .venv/bin/python scripts/gemma4_report.py \
        --out out/gemma4-ternary/report.html >/dev/null 2>&1 \
        || echo "[report-watch] render failed at $(date +%H:%M:%S)"
    sleep "$INT"
done

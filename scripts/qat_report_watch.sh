#!/usr/bin/env bash
# Live-refresh the QAT report while a run trains, then finalize it.
#
#     bash scripts/qat_report_watch.sh out/exp-058/kd8b-full [interval_s]
#
# Every interval (default 600 s): parse the run's train.log into telemetry CSVs and
# regenerate report.html — pure log parsing plus SVG, no model load, safe beside
# training. After the trainer exits, census the final latents (mmap, 8 tensors) so the
# distribution-shift table compares trained codes against the step-0 census, and write
# the report one last time.
#
# CENSUS defaults to the shipped-model step-0 census, which is valid for EVERY run
# because each run starts from the shipped weights; override for a resumed lineage.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: qat_report_watch.sh <run_dir> [interval_s]}"
INT="${2:-600}"
PY=.venv/bin/python
TEL="$RUN/telemetry"
CENSUS="${CENSUS:-out/exp-058/census_step0.csv}"
TITLE="${TITLE:-$(basename "$RUN") — QAT training dynamics}"
mkdir -p "$TEL"

refresh() {
    $PY scripts/parse_qat_log.py "$RUN/train.log" --out "$TEL" >/dev/null 2>&1 || return 0
    local w ga
    w=$($PY -c "import json;print(json.load(open('$RUN/run_config.json'))['window'])" \
        2>/dev/null || echo 32768)
    ga=$($PY -c "import json;print(json.load(open('$RUN/run_config.json'))['grad_accum'])" \
        2>/dev/null || echo 1)
    local extra=()
    if [ -f "$TEL/census_latest.csv" ] && [ -f "$TEL/census_latest.step" ]; then
        extra=(--latest "$TEL/census_latest.csv" --latest-step "$(cat "$TEL/census_latest.step")")
    fi
    local mc ka
    mc=$($PY -c "import json;print(json.load(open('$RUN/run_config.json'))['model_dir'])" \
        2>/dev/null || true)
    [ -n "${mc:-}" ] && [ -f "$mc/config.json" ] && extra+=(--model-config "$mc/config.json")
    # KD runs: pass alpha so the KL panel can derive the CE component (exact at T=1)
    ka=$($PY -c "import json;c=json.load(open('$RUN/run_config.json'));\
print(c['kd_alpha'] if c.get('kd_table') else '')" 2>/dev/null || true)
    [ -n "${ka:-}" ] && extra+=(--kd-alpha "$ka")
    # Run-specific findings live beside the run, not in this script
    [ -f "$RUN/notes.md" ] && extra+=(--notes "$RUN/notes.md")
    # Teacher's own probe values (KD runs): dotted asymptote lines on the probe panel
    if [ -f "$RUN/teacher_probe.json" ]; then
        while IFS= read -r kv; do extra+=(--teacher-probe "$kv"); done < <(
            $PY -c "import json;[print(f'{k}={v}') for k,v in \
json.load(open('$RUN/teacher_probe.json')).items()]" 2>/dev/null)
    fi
    $PY scripts/qat_report.py --telemetry "$TEL" --census "$CENSUS" \
        --window "$w" --grad-accum "$ga" --title "$TITLE" \
        "${extra[@]}" --out "$RUN/report.html" >/dev/null \
        && echo "[watch] $(date -u +%H:%M) refreshed $RUN/report.html"
}

refresh
while pgrep -f "quant_tuner[.]qat[.]train" > /dev/null; do
    sleep "$INT"
    refresh
done

ck=$(ls -t "$RUN"/trained_latents*.pt 2>/dev/null | head -1)
if [ -n "$ck" ] && [ -s "$TEL/flips.csv" ]; then
    step=$(grep -oE "done at step [0-9]+" "$RUN/train.log" | grep -oE "[0-9]+" | tail -1)
    $PY scripts/ternary_distribution.py census --latents "$ck" \
        --tensors "$TEL/flips.csv" --out "$TEL/census_latest.csv" \
        && echo "${step:-0}" > "$TEL/census_latest.step"
fi
refresh
echo "[watch] final report written"

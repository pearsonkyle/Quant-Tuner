#!/usr/bin/env bash
# Compact status line for a long QAT run, sampled on a slow interval.
#
# A multi-hour run has nothing useful to say minute to minute: a step is ~15-30 min at a
# 32K window. This appends one line per interval to <out>/watch.log and prints the same,
# so checking in means reading a file rather than re-polling the process.
#
#   scripts/watch_qat_run.sh out/exp-058/swe32k [interval_s] [max_hours]
#
# Columns: elapsed, last step + loss + grad-norm, MPS GiB, swap GiB, s/step, and the
# health verdict. Swap TREND is the number that matters — a step that is slow with flat
# swap is merely expensive; swap climbing every interval is the failure mode that ends in
# a macOS SIGKILL with no traceback.
set -uo pipefail

OUT="${1:?usage: watch_qat_run.sh <out-dir> [interval_s] [max_hours]}"
INTERVAL="${2:-300}"
MAX_HOURS="${3:-48}"
LOG="$OUT/watch.log"
METRICS="$OUT/metrics.jsonl"
mkdir -p "$OUT"

swap_gib() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | awk '{print $1/1024}'; }

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
prev_swap=$(swap_gib)
printf '%-8s %-26s %-9s %-16s %s\n' ELAPSED STEP MPS SWAP NOTE | tee -a "$LOG"

while [ "$(date +%s)" -lt "$deadline" ]; do
  pid=$(pgrep -f "quant_tuner.qat.train" | head -1 || true)
  now_swap=$(swap_gib)
  delta=$(awk -v a="$now_swap" -v b="$prev_swap" 'BEGIN{printf "%+.1f", a-b}')
  prev_swap=$now_swap

  if [ -z "$pid" ]; then
    # Distinguish a clean finish from a kill: the trainer prints "done at step" on exit,
    # and a macOS OOM kill leaves NO traceback at all — just a truncated log.
    if grep -qa "done at step" "$OUT"/../*.log "$OUT"/*.log 2>/dev/null; then
      note="FINISHED"
    else
      note="GONE (no 'done at step' — check for an OOM kill: log just ends)"
    fi
    printf '%-8s %-26s %-9s %-16s %s\n' "$(date +%H:%M)" "-" "-" "$now_swap" "$note" | tee -a "$LOG"
    break
  fi

  el=$(ps -o etime= -p "$pid" | tr -d ' ')
  if [ -f "$METRICS" ]; then
    read -r step loss gn mem sps <<<"$(python3 - "$METRICS" <<'PY'
import json, sys
last = None
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("kind") == "step":
        last = d
if last:
    print(last["step"], round(last["loss"], 4), round(last.get("grad_norm", 0), 2),
          round(last.get("mem_gib", 0), 1), round(last.get("s_per_step", 0)))
else:
    print("-", "-", "-", "-", "-")
PY
)"
  else
    step=- loss=- gn=- mem=- sps=-
  fi

  case "$delta" in
    +[5-9].*|+[1-9][0-9]*) note="swap RISING $delta GiB/interval — watch for a kill" ;;
    *) note="ok (${sps}s/step)" ;;
  esac
  printf '%-8s %-26s %-9s %-16s %s\n' \
    "$el" "step $step loss=$loss g=$gn" "${mem}G" "$(printf '%.1f' "$now_swap") ($delta)" "$note" \
    | tee -a "$LOG"
  sleep "$INTERVAL"
done

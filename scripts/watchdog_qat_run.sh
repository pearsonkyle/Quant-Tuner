#!/usr/bin/env bash
# Alerting watchdog for a QAT run — silent while healthy, loud when it isn't.
#
#     bash scripts/watchdog_qat_run.sh out/exp-058/trained_sft32k_sw1 [PID]
#
# The sibling scripts/watch_qat_run_cuda.sh prints a status block every tick for a human to
# read; if nobody is reading, a stalled run looks exactly like a healthy one. This inverts
# that: it emits ONLY on a progress heartbeat, a stall, a divergence, or exit, so its output
# can drive a notification channel and silence genuinely means "still fine".
#
# The three failures it is built around, all observed on this project:
#
#   * EXIT — the sft32k_sw1 run was launched with `nohup … &` and died at step 1 when the
#     launching session ended. Nothing noticed for hours. Exit is reported with whether
#     trained_latents.pt actually exists, because "finished" and "died" both stop the
#     process and only the artifact tells them apart.
#   * STALL — process alive, GPU busy, no forward progress. Nothing in the metrics file
#     moves, so any monitor keyed on the last row keeps reporting the same healthy numbers
#     forever. Keyed on train.log MTIME instead, which is the one signal that a stuck run
#     cannot fake.
#   * DIVERGENCE — bf16 diverged twice on this model (docs/qat_32k_handoff.md §10.6) with
#     all sources spiking at once. GradSpikeGuard cannot see it; a NaN loss or a gnorm far
#     outside the fp32 run's observed range can.
set -u

RUN="${1:?usage: watchdog_qat_run.sh RUN_DIR [PID]}"
RUN="${RUN%/}"
LOG="$RUN/train.log"
METRICS="$RUN/metrics.jsonl"

# Match the trainer by the --out it was given, so several concurrent runs stay distinct.
PID="${2:-}"
if [ -z "$PID" ]; then
    for p in $(pgrep -f "quant_tuner.qat.train" 2>/dev/null || true); do
        if tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -qxF "$(basename "$RUN")" \
           || tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -qF "$RUN"; then
            PID="$p"; break
        fi
    done
fi
[ -n "$PID" ] || { echo "no trainer found for $RUN — pass the PID explicitly"; exit 1; }

# train.log grows every 4 steps (~250s at 63 s/step). Validation and a ~28 GB checkpoint
# save add real pauses on top, so the threshold must clear those without letting a genuine
# hang sit unnoticed for an hour. 25 min is ~6x the normal write interval.
STALL="${STALL:-1500}"
POLL="${POLL:-120}"
HEARTBEAT_STEPS="${HEARTBEAT_STEPS:-50}"
# Calibrated against the sft32k control's OWN metrics.jsonl, not the pre-flight figure: that
# healthy run — same corpus, same lr, completed and exported fine — hit gnorm 17.19 at step
# 35, 68.18 at 40 and 90.96 at 55 during the ordinary post-warmup reorganization. A limit of
# 50 (the pre-flight's "max 1.88" x margin) fires on a normal run. Note this leaves no clean
# fixed separation from bf16's 129 divergence, so do NOT read a single spike as failure —
# NaN/Inf is the unambiguous signal, and this bound only catches a genuine runaway.
GNORM_MAX="${GNORM_MAX:-200.0}"

echo "[watchdog] $RUN pid=$PID stall=${STALL}s heartbeat=${HEARTBEAT_STEPS} steps" >&2

read_metrics() {
    python3 - "$METRICS" <<'PY' 2>/dev/null || true
import json, sys
last = None
try:
    with open(sys.argv[1]) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("kind") == "step":
                last = d
except OSError:
    pass
if last:
    print(last.get("step", 0), last.get("total_steps", 0),
          last.get("loss", float("nan")), last.get("grad_norm", float("nan")),
          last.get("s_per_step", 0.0), last.get("mem_peak_gib", 0.0))
PY
}

last_hb=0
while true; do
    alive=1
    kill -0 "$PID" 2>/dev/null || alive=0

    set -- $(read_metrics)
    step=${1:-0}; total=${2:-0}; loss=${3:-nan}; gnorm=${4:-nan}
    sps=${5:-0}; peak=${6:-0}

    if [ "$alive" -eq 0 ]; then
        if [ -f "$RUN/trained_latents.pt" ]; then
            echo "DONE: $(basename "$RUN") ended at step $step/$total; trained_latents.pt present."
        else
            echo "TRAINER GONE at $(date -u +%H:%M:%SZ) — step $step/$total, NO trained_latents.pt. Last log:"
            sed 's/\r/\n/g' "$LOG" 2>/dev/null | grep -vE "Loading weights" | tail -8
        fi
        break
    fi

    if [ -f "$LOG" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
        if [ "$age" -gt "$STALL" ]; then
            echo "STALL: train.log unchanged for ${age}s at step $step/$total (threshold ${STALL}s)"
            echo "  state=$(awk '/^State/{print $2}' "/proc/$PID/status" 2>/dev/null) gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
            sed 's/\r/\n/g' "$LOG" 2>/dev/null | grep -vE "Loading weights" | tail -5
            break
        fi
    fi

    case "$loss" in
        nan|NaN|inf|Inf|-inf|-Inf) echo "DIVERGED: loss=$loss at step $step/$total"; break ;;
    esac
    bad=$(python3 -c "
try:
    g=float('$gnorm'); l=float('$loss')
    print(1 if (g!=g or l!=l or g>$GNORM_MAX) else 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    if [ "$bad" = "1" ]; then
        echo "DIVERGED: step $step/$total loss=$loss gnorm=$gnorm (limit $GNORM_MAX)"
        break
    fi

    if [ "$step" -ge "$((last_hb + HEARTBEAT_STEPS))" ]; then
        last_hb=$step
        echo "$(python3 -c "
print(f'ok step $step/$total ({100*$step/max($total,1):.0f}%) '
      f'loss={$loss:.4f} gnorm={$gnorm:.2f} {$sps:.0f}s/step '
      f'peak={$peak:.1f}GiB ETA {($total-$step)*$sps/3600:.1f}h')
" 2>/dev/null || echo "ok step $step/$total")"
    fi

    sleep "$POLL"
done

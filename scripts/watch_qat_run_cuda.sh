#!/usr/bin/env bash
# Slow monitor for a CUDA QAT run. The CUDA sibling of scripts/watch_qat_run.sh, which is
# macOS/unified-memory shaped (`sysctl vm.swapusage`, "OOM kills give no traceback") and
# reports nothing useful here.
#
#     bash scripts/watch_qat_run_cuda.sh out/exp-058/trained_sft32k [INTERVAL_S] [HOURS]
#
# What is different on CUDA, and why this script exists rather than a flag on the other one:
#
#   * An OOM is an ordinary Python exception with a traceback, not a silent SIGKILL. So the
#     failure signal is the log tail, not a vanished process — this greps for it.
#   * VRAM is dedicated and swap is irrelevant. The number that matters is how close peak
#     allocation sits to the card, because the transient that kills a run is a checkpoint
#     save or a validation, not a slow creep.
#   * A *leaked* process is the CUDA analogue of the macOS swap trap: a killed run leaves a
#     trainer holding the whole card, and the next thing you start cannot load its model.
#     nvidia-smi's process list is printed every tick for exactly that reason.
#
# Deliberately slow: reading `metrics.jsonl` through scripts/qat_progress_report.py is the
# primary instrument (flip velocity, per-source loss, gnorm). This only watches the health
# signals that live outside that file.
set -u

RUN="${1:?usage: watch_qat_run_cuda.sh RUN_DIR [INTERVAL_S] [HOURS]}"
INTERVAL="${2:-600}"
HOURS="${3:-72}"
# awk, not $(( )): a fractional --hours (handy for smoke-testing this script) is a bash
# arithmetic syntax error, and the failure lands on an unbound END two lines later.
END=$(awk -v n="$(date +%s)" -v h="$HOURS" 'BEGIN { printf "%d", n + h * 3600 }')
LOG="${RUN}/watch_cuda.log"
mkdir -p "$RUN"

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "watching $RUN every ${INTERVAL}s for ${HOURS}h (log: $LOG)"
while [ "$(date +%s)" -lt "$END" ]; do
    # --- GPU: utilisation, memory, and WHO is holding it -----------------------------
    read -r util mem_used mem_tot temp pwr < <(
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                   --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
    say "gpu ${util:-?}% util  ${mem_used:-?}/${mem_tot:-?} MiB  ${temp:-?}C  ${pwr:-?}W"
    procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    nproc_gpu=$(printf '%s' "$procs" | grep -c . || true)
    if [ "${nproc_gpu:-0}" -gt 1 ]; then
        say "  WARNING: ${nproc_gpu} processes on the card — a leaked trainer starves this run"
        printf '%s\n' "$procs" | sed 's/^/    /' | tee -a "$LOG"
    fi

    # --- training progress: last step line + the trainer's own peak-memory number -----
    if [ -f "${RUN}/metrics.jsonl" ]; then
        python - "$RUN" <<'PY' | tee -a "$LOG"
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "metrics.jsonl"
last_step = last_val = None
for line in p.read_text().splitlines():
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("kind") == "step":
        last_step = r
    elif r.get("kind") == "val":
        last_val = r
if last_step:
    s, tot = last_step.get("step", 0), last_step.get("total_steps", 0)
    sps = last_step.get("s_per_step") or 0
    eta = (tot - s) * sps / 3600 if sps else 0
    print(f"  step {s}/{tot} loss={last_step.get('loss', 0):.4f} "
          f"gnorm={last_step.get('grad_norm', 0):.2f} "
          f"skipped={last_step.get('n_skipped', 0)} "
          f"mem={last_step.get('mem_gib', 0):.1f}/{last_step.get('mem_peak_gib', 0):.1f}GiB "
          f"{sps:.0f}s/step  ETA {eta:.1f}h")
if last_val:
    print(f"  val @ {last_val.get('step')}: {last_val.get('val_masked_ce', 0):.4f} "
          f"({last_val.get('val_seconds', 0):.0f}s)")
PY
    fi

    # --- failure signals: on CUDA these are exceptions in the log, not a silent kill ---
    for f in "${RUN}"/*.log "${RUN}"/../*.log; do
        [ -f "$f" ] || continue
        hit=$(grep -aE "OutOfMemoryError|AcceleratorError|Traceback|CUDA error|device-side assert" "$f" | tail -1)
        [ -n "$hit" ] && say "  FAULT in $(basename "$f"): ${hit:0:160}"
    done

    sleep "$INTERVAL"
done
say "watch window elapsed"

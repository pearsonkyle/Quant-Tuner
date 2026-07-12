#!/bin/bash
# Every 2h: snapshot SWE-rebench overnight progress, alert on stall/death, and
# restart a crash (guarded). Logs to out/swe-rebench/watchdog.log.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
WD=out/swe-rebench/watchdog.log
INTERVAL=${1:-7200}          # seconds between checks (default 2h)
MAX_RESTARTS=2
prev=-1
restarts=0

note() { osascript -e "display notification \"$1\" with title \"quant-tuner watchdog\"" 2>/dev/null; }
# grep -c prints "0" (exit 1) on no-match and nothing on a missing file; both -> 0.
count() { local n; n=$(grep -cE "$2" "$1" 2>/dev/null); echo "${n:-0}"; }

echo "[$(date '+%F %H:%M')] watchdog started (every $((INTERVAL/60))m)" >> "$WD"
while true; do
  ts=$(date '+%F %H:%M')
  done=$(count out/swe-rebench/overnight.log "ALL DONE")
  # openai-agents writes one result.json per graded (instance, rep) — count those.
  ginst=$(find out/swe-rebench/gemma-awq-swe/trajectories -name "*.result.json" 2>/dev/null | wc -l | tr -d ' ')
  oinst=$(find out/swe-rebench/ornith-awq-swe/trajectories -name "*.result.json" 2>/dev/null | wc -l | tr -d ' ')
  ginst=${ginst:-0}; oinst=${oinst:-0}
  gl=$(ls -t out/swe-rebench/gemma-awq-swe/server_*.log 2>/dev/null | head -1)
  ol=$(ls -t out/swe-rebench/ornith-awq-swe/server_*.log 2>/dev/null | head -1)
  gt=$(count "$gl" "reasoning-budget: deactivated"); ot=$(count "$ol" "reasoning-budget: deactivated")
  turns=$((gt + ot))
  prog=$((ginst*100000 + oinst*100000 + turns))
  alive=$(pgrep -f run_swebench_eval >/dev/null && echo yes || echo no)
  phase=$(tail -1 out/swe-rebench/overnight.log 2>/dev/null | tr -d '\n')

  if [ "$done" -ge 1 ]; then
    echo "[$ts] DONE — gemma=$ginst ornith=$oinst instances graded. exiting." >> "$WD"
    note "SWE-rebench overnight run COMPLETE"
    break
  elif [ "$alive" = "no" ]; then
    if [ "$restarts" -lt "$MAX_RESTARTS" ]; then
      restarts=$((restarts+1))
      echo "[$ts] DIED before done (g=$ginst o=$oinst turns=$turns) — RESTART #$restarts" >> "$WD"
      note "SWE-rebench died — restarted (#$restarts)"
      nohup bash scripts/run_swe_awq_overnight.sh > /dev/null 2>&1 &
    else
      echo "[$ts] DIED and hit max restarts ($MAX_RESTARTS) — giving up." >> "$WD"
      note "SWE-rebench died — gave up after $MAX_RESTARTS restarts"
      break
    fi
  elif [ "$prev" -ge 0 ] && [ "$prog" -le "$prev" ]; then
    echo "[$ts] STALLED — no progress in $((INTERVAL/60))m (g=$ginst o=$oinst turns=$turns). $phase" >> "$WD"
    note "SWE-rebench STALLED (no progress in $((INTERVAL/60))m)"
  else
    d=$((prog - (prev<0?prog:prev)))
    echo "[$ts] OK — gemma=$ginst ornith=$oinst graded, turns=$turns (+$d since last)  $phase" >> "$WD"
  fi
  prev=$prog
  sleep "$INTERVAL"
done

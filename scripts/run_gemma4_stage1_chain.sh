#!/usr/bin/env bash
# Unattended: wait for the KD table, verify it, probe the teacher, run the lr arms.
#
#     nohup bash scripts/run_gemma4_stage1_chain.sh > out/gemma4-ternary/chain.log 2>&1 &
#
# Exists to remove the handoff gap. The precompute finishes at an arbitrary hour and the
# card would otherwise sit idle until someone noticed. Each step gates the next: a table
# that fails verification must not become three GPU-hours of training against the wrong
# distribution.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
TABLE="$R/kd/gemma31b_topk64_fs106.pt"
CORPUS="$R/corpus_sft_gemma4_32768.pt"
TEACHER="${TEACHER:-google/gemma-4-31B-it}"
STUDENT="${STUDENT:-google/gemma-4-E4B-it-qat-q4_0-unquantized}"

say() { echo "[chain $(date +%H:%M:%S)] $*"; }

say "waiting for the KD precompute to exit"
while pgrep -f "kd[_]precompute" > /dev/null; do sleep 60; done
say "precompute exited"

[ -f "$TABLE" ] || { say "FATAL: no table at $TABLE"; tail -20 "$R/kd/precompute_31b.log"; exit 1; }

say "verifying the table"
.venv/bin/python scripts/verify_kd_table.py --table "$TABLE" --corpus "$CORPUS" --stop-id 106
rc=$?
[ $rc -ne 0 ] && { say "FATAL: table verification failed — not training on it"; exit $rc; }

say "teacher stop probe ($TEACHER)"
.venv/bin/python scripts/teacher_stop_probe.py --teacher "$TEACHER" \
    --student-model "$STUDENT" --out "$R/kd/teacher_probe_31b.json" 2>&1 \
    | grep -av "^Loading" | tail -3
# A missing teacher probe costs the report its asymptote lines, nothing more — the arms
# are still worth running, so this one does not gate.

say "lr arms"
bash scripts/run_gemma4_stage1_ab.sh
say "chain done rc=$?"

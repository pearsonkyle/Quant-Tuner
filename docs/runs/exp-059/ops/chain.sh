#!/usr/bin/env bash
# exp-059 chain: wait for KD table -> train coder1 (anchor10 recipe) -> export+bench.
# Written once, launched once — do not edit while running.
set -uo pipefail
cd /workspace/Quant-Tuner
TABLE=out/exp-059/kd/coder_32b_topk64_fs151645.pt
LOG=out/exp-059/kd/precompute_32b.log

# 1. wait for the precompute process (by name, bracketed to avoid self-match) to finish
while pgrep -f "kd_precompute[.]py.*coder_32b" >/dev/null; do sleep 120; done
[ -f "$TABLE" ] || { echo "[chain] KD table missing after precompute exit — aborting"; exit 1; }
echo "[chain] KD table ready:"; grep -E "coverage|saved" "$LOG" | tail -3

# 2. train — the anchor10 prescription on the coder corpus, ~1 epoch
TAG=coder1 LR=5e-4 STEER=0.1 CLIP=0.25 REP=0.1 REP_CAP=0.6 \
REP_K=1,2,3,4,5 REP_N=10 \
REP_BANK=out/exp-058/kd/rep_bank.json \
REP_TRAJ=out/exp-058/kd/rep_traj_contexts.jsonl \
CORPUS=out/exp-059/corpus_coder_32768.pt \
VAL=out/exp-059/corpus_coder_val_32768.pt \
TABLE="$TABLE" \
TEACHER_PROBE=out/exp-058/kd/teacher_probe_32b.json \
OUT=out/exp-059/kd32b-full-coder1 EPOCHS=1.0 \
bash scripts/run_kd_anchor_qat.sh
rc=$?
if [ $rc -ne 0 ]; then echo "[chain] training rc=$rc — stopping before eval"; exit $rc; fi

# 3. eval loop: export Q2_0 -> GGUF stop probe -> in-dist stop (new val) -> SWE mimic
VAL_CORPUS=out/exp-059/corpus_coder_val_32768.pt \
bash scripts/run_kd_export_bench.sh coder1 out/exp-059/kd32b-full-coder1/trained_latents.pt
rc=$?
echo "[chain] done rc=$rc"
exit $rc

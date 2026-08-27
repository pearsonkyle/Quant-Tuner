#!/usr/bin/env bash
# exp-059 chain, stage 2: relaunch coder1 with expandable_segments (attempt 1 OOMed in
# the first backward — denser windows than anchor10 + 7 GiB allocator fragmentation).
# Appends to the same chain.log the milestone monitor follows. Do not edit while running.
set -uo pipefail
cd /workspace/Quant-Tuner
TABLE=out/exp-059/kd/coder_32b_topk64_fs151645.pt
[ -f "$TABLE" ] || { echo "[chain] KD table missing — aborting"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

VAL_CORPUS=out/exp-059/corpus_coder_val_32768.pt \
bash scripts/run_kd_export_bench.sh coder1 out/exp-059/kd32b-full-coder1/trained_latents.pt
rc=$?
echo "[chain] done rc=$rc"
exit $rc

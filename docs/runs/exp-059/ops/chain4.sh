#!/usr/bin/env bash
# exp-059 chain, stage 4: coder3 = coder2's corpus + KD table, GRAD_ACCUM=4.
# coder2 stopped at step ~830: benchwatch s400 (flailing, path-blend hallucinations) ->
# s800 (near-mute, 8k-token rambles, <=2 tool calls) while flips hit 9.1% at s800 —
# total code drift ~5x the validated envelope. LR cannot drop (flip floor ~3e-4), so
# accum 4 restores anchor10's optimizer-step count (743 vs 613) over all 2974 windows.
# Do not edit while running.
set -uo pipefail
cd /workspace/Quant-Tuner
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QAT_LOGIT_CHUNK=512   # accum keeps all 27.8G of grads resident; claw back the logit-peak GB
TAG=coder3 GRAD_ACCUM=4 LR=5e-4 STEER=0.1 CLIP=0.25 REP=0.1 REP_CAP=0.6 \
REP_K=1,2,3,4,5 REP_N=10 \
REP_BANK=out/exp-058/kd/rep_bank.json \
REP_TRAJ=out/exp-058/kd/rep_traj_contexts.jsonl \
CORPUS=out/exp-059/corpus_coder2_32768.pt \
VAL=out/exp-059/corpus_coder2_val_32768.pt \
TABLE=out/exp-059/kd/coder2_32b_topk64_fs151645.pt \
TEACHER_PROBE=out/exp-058/kd/teacher_probe_32b.json \
OUT=out/exp-059/kd32b-full-coder3 EPOCHS=1.0 \
bash scripts/run_kd_anchor_qat.sh
rc=$?
if [ $rc -ne 0 ]; then echo "[chain] training rc=$rc — stopping before eval"; exit $rc; fi
VAL_CORPUS=out/exp-059/corpus_coder2_val_32768.pt \
bash scripts/run_kd_export_bench.sh coder3 out/exp-059/kd32b-full-coder3/trained_latents.pt
rc=$?
echo "[chain] done rc=$rc"
exit $rc

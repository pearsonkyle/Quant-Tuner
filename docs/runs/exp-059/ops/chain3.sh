#!/usr/bin/env bash
# exp-059 chain, stage 3: path-diversified corpus -> KD table -> coder2 -> eval.
# coder1 was stopped at step 905 (3-rep sidecar at s800: /testbed fixation, a no-tool-call
# 8k ramble, garbled path literals; root cause = 55% /testbed monoculture in sft_v2).
# Appends to the same chain.log the milestone monitor follows. Do not edit while running.
set -uo pipefail
cd /workspace/Quant-Tuner
CORPUS=out/exp-059/corpus_coder2_32768.pt
VAL=out/exp-059/corpus_coder2_val_32768.pt
TABLE=out/exp-059/kd/coder2_32b_topk64_fs151645.pt
[ -f "$CORPUS" ] || { echo "[chain] no corpus $CORPUS"; exit 1; }

echo "[chain] KD precompute over path-diversified corpus"
PYTHONPATH=src .venv/bin/python scripts/kd_precompute.py \
  --teacher SWE-Lego/SWE-Lego-Qwen3-32B \
  --corpus "$CORPUS" --include-ids 151645 \
  --out "$TABLE" > out/exp-059/kd/precompute_32b_coder2.log 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "[chain] KD precompute rc=$rc — aborting"; exit $rc; }
grep -E "saved" out/exp-059/kd/precompute_32b_coder2.log | tail -1
[ -f "$TABLE" ] || { echo "[chain] KD table missing — aborting"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=coder2 LR=5e-4 STEER=0.1 CLIP=0.25 REP=0.1 REP_CAP=0.6 \
REP_K=1,2,3,4,5 REP_N=10 \
REP_BANK=out/exp-058/kd/rep_bank.json \
REP_TRAJ=out/exp-058/kd/rep_traj_contexts.jsonl \
CORPUS="$CORPUS" VAL="$VAL" TABLE="$TABLE" \
TEACHER_PROBE=out/exp-058/kd/teacher_probe_32b.json \
OUT=out/exp-059/kd32b-full-coder2 EPOCHS=1.0 \
bash scripts/run_kd_anchor_qat.sh
rc=$?
if [ $rc -ne 0 ]; then echo "[chain] training rc=$rc — stopping before eval"; exit $rc; fi

VAL_CORPUS="$VAL" \
bash scripts/run_kd_export_bench.sh coder2 out/exp-059/kd32b-full-coder2/trained_latents.pt
rc=$?
echo "[chain] done rc=$rc"
exit $rc

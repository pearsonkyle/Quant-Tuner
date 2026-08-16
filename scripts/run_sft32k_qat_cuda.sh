#!/bin/bash
# The 32768-window universal-SFT QAT run, on CUDA.
#
# This is the CUDA sibling of run_sft_qat_pipeline.sh, which is pinned to an M4 Max
# (`cd /Users/kpearson/...`, PYTORCH_ENABLE_MPS_FALLBACK, MPS-tuned cache cadence) and
# chains a SWE-rebench eval that needs a Docker daemon. This script does the TRAIN and
# EXPORT halves only; grade the result on a Docker-capable box.
#
#   bash scripts/run_sft32k_qat_cuda.sh [TAG] [LR] [EPOCHS]
#
# What this run is testing — two independent fixes for one observed regression. The
# previous run (sft8k-full, window 8064) halved the tool-error rate but LOST THE ABILITY
# TO STOP: max_turns on 7/10 instances, 97% of trajectories looping.
#
#   1. window 32768 — 97% of SWE trajectories now fit whole (27% at 8064), so the model
#      sees complete task->completion arcs. The ending is where the stop decision lives.
#   2. --stop-weight 6.0 — the terminating <|im_end|> is 35,359 of 6,071,948 targets, one
#      "stop" per 172 "keep going" (0.58% of the loss). At 6.0 it carries 3.40%.
#
# The window does not subsume the weight: sessions pack contiguously, so the
# target-to-stop ratio is the same at both window sizes (6,060,840/35,046 at 8064 vs
# 6,071,948/35,359 at 32768). The window buys whole arcs; the weight buys salience.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-sft32k}"
LR="${2:-5e-4}"
EPOCHS="${3:-1.0}"

export PYTHONPATH=src
# Reserved-but-unallocated blocks were 9.6 GiB in the OOM that started this work. fp32 at
# a 32768 window runs close enough to the card that fragmentation alone can decide it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY=.venv/bin/python

CORPUS="${CORPUS:-out/exp-058/sft_corpus_universal_32768.pt}"
VAL="${VAL:-out/exp-058/sft_corpus_val_32768.pt}"
OUT="out/exp-058/trained_${TAG}"
LOG="${OUT}/train.log"

# grad-accum 1, NOT 4. At window 32768 that is 32,768 tokens/step, which matches
# sft8k-full's 32,256 (accum 4 x 8064) almost exactly — so lr 5e-4 transfers like-for-like
# instead of being re-derived, and 613 steps/epoch keeps the cosine schedule and the
# spike guard's 25-step median at the resolution they were measured at.
ACCUM="${ACCUM:-1}"
# 0.10, not 0.05: sft8k-full's 9x loss excursion began FOUR steps after warmup ended.
WARMUP="${WARMUP:-0.10}"
STOP_WEIGHT="${STOP_WEIGHT:-6.0}"
# fp32 latents + true-fp32 matmuls: identical numerics to every published run. `high`
# (TF32) keeps the latents and ternarization exact and only reduces matmul accumulation.
PRECISION="${PRECISION:-highest}"

for f in "$CORPUS" "$VAL"; do
    [ -f "$f" ] || { echo "missing $f — see docs/qat_32k_handoff.md"; exit 1; }
done

# A leaked trainer holding the card is the CUDA analogue of the macOS swap trap: the run
# will not start, and the OOM traceback blames this configuration for someone else's.
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${busy:-0}" -gt 2048 ]; then
    echo "GPU already holding ${busy} MiB — free it before starting:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
fi

mkdir -p "$OUT"
RESUME_ARGS=()
[ -f "${OUT}/trained_latents.pt" ] && RESUME_ARGS=(--resume "${OUT}/trained_latents.pt")

echo "[run] tag=${TAG} lr=${LR} epochs=${EPOCHS} accum=${ACCUM} precision=${PRECISION}"
echo "[run] log -> ${LOG}"

nohup $PY -m quant_tuner.qat.train \
    --corpus "$CORPUS" --val-corpus "$VAL" \
    --train-layers 36 --optim adafactor --dtype fp32 \
    --matmul-precision "$PRECISION" \
    --grad-accum "$ACCUM" --epochs "$EPOCHS" --lr "$LR" --warmup-frac "$WARMUP" \
    --stop-weight "$STOP_WEIGHT" \
    --val-every 25 --val-windows 4 \
    --ckpt-every 50 --ckpt-keep 3 \
    "${RESUME_ARGS[@]}" \
    --out "$OUT" > "$LOG" 2>&1 &

echo "[run] pid $!"
echo
echo "Watch it with EITHER of these — not by polling the process:"
echo "  python scripts/qat_progress_report.py ${OUT} --watch 1800"
echo "  bash scripts/watch_qat_run_cuda.sh ${OUT} 600 48"
echo
echo "Export when it finishes (Q2_0 needs the prism fork; see docs/ternary_qat.md):"
echo "  LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY \\"
echo "      scripts/exp057_qat_export.py --latents ${OUT}/trained_latents.pt --tag ${TAG}"

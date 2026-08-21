#!/usr/bin/env bash
# The pre-registered lr A/B for gemma-4-E4B stage 1, then a summary table.
#
#     bash scripts/run_gemma4_stage1_ab.sh                  # 2e-4 / 5e-4 / 1e-3
#     LRS="5e-4 1e-3" CLIP=1.0 bash scripts/run_gemma4_stage1_ab.sh   # clip arm
#
# Why an A/B at all: lr 5e-4 is Bonsai's measured sweet spot and the two situations are
# not the same one. Bonsai's weights START on the ternary grid, so the only question
# there is whether the lr is big enough to flip codes at all (3e-4 flips ~0% and the
# loss falls on scale drift). gemma's start OFF the grid -- the CPU smoke flipped
# 1.4-1.7% of codes in two steps -- so the risk shifts from "too small to move anything"
# toward "large enough to break termination".
#
# 60 steps is a sixth of the stage. It picks an lr; it does NOT decide the go/no-go,
# which is read at the end of a full stage.
set -uo pipefail
cd "$(dirname "$0")/.."

LRS="${LRS:-2e-4 5e-4 1e-3}"
EPOCHS="${EPOCHS:-0.37}"        # 651 windows / accum 4 -> ~60 steps
PREFIX="${PREFIX:-ab}"
SUFFIX="${SUFFIX:-}"            # set when varying something other than lr

for LR in $LRS; do
    TAG="${PREFIX}-lr${LR}${SUFFIX}"
    OUT="out/gemma4-ternary/${TAG}"
    if [ -f "$OUT/stage_damage_trained.json" ]; then
        echo "[ab] $TAG already measured — skipping"
        continue
    fi
    echo "[ab] === $TAG ==="
    STAGE=1 TAG="$TAG" LR="$LR" EPOCHS="$EPOCHS" bash scripts/run_gemma4_stage.sh
    rc=$?
    echo "[ab] $TAG exit=$rc"
    # A probe abort exits non-zero and WRITES a checkpoint -- that is a result, and the
    # remaining arms are still worth running. No checkpoint means the run died before it
    # trained (OOM, bad flag, missing file), which every later arm would repeat, so stop
    # rather than burn the night reproducing it.
    if [ $rc -ne 0 ] && ! [ -f "$OUT/trained_latents.pt" ]; then
        echo "[ab] $TAG produced no checkpoint — infrastructure failure, aborting the chain"
        tail -20 "$OUT/train.log" 2>/dev/null
        break
    fi
done

echo
.venv/bin/python scripts/gemma4_ab_summary.py out/gemma4-ternary/${PREFIX}-lr* \
    --json out/gemma4-ternary/${PREFIX}_summary.json

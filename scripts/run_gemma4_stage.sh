#!/usr/bin/env bash
# One stage of the gemma-4-E4B progressive ternarization schedule: train, then measure
# what the training recovered. Env-overridable; every default is a measurement, not a
# preference (see docs/gemma4_ternary_feasibility.md and the run's notes.md).
#
#     STAGE=1 bash scripts/run_gemma4_stage.sh                    # the go/no-go stage
#     STAGE=1 TAG=lr1e-3 LR=1e-3 EPOCHS=0.37 bash scripts/run_gemma4_stage.sh   # A/B arm
#     STAGE=2 MODEL_DIR=out/gemma4-ternary/stage1/baked bash scripts/run_gemma4_stage.sh
#
# Stages compose by BAKING, not by --resume: resume requires the checkpoint to carry
# every trainable name, and stage N+1's layer set is a different set. So each stage
# writes an HF model dir with its ternarized weights materialized, and the next stage
# points --model-dir at it. wrap_model then proves those layers are already exactly on
# the grid and skips re-wrapping them.
#
# Why these are NOT the Bonsai numbers:
#   --probe-abort 0.03 / --probe-abort-control 0.01  gemma's own baseline is diagnostic
#       0.00274 and control 0.0703 (out/gemma4-ternary/stop_baseline.json). Bonsai's
#       control sits at 0.99995 -- an abort floor copied from there fires immediately.
#   --steer-weight 0 (and no --steer-rep-*)  the steering context classes are
#       Qwen-dialect and their control class is INVERTED under gemma's template; the
#       repetition banks are Qwen-rendered. Port via PROBE_SPECS before enabling.
#   --dense-kind down_proj  solo KLD 1.199 vs 0.147 for q_proj -- 3.4x the next-worst.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:?set STAGE=<n>}"
case "$STAGE" in
  1) DEF_LAYERS="0,1,2,3,7,8" ;;              # layer_damage.json["layer_order"][0:6]
  2) DEF_LAYERS="5,6,36,37,38,39" ;;          #                              [6:12]
  *) DEF_LAYERS="" ;;
esac
LAYERS="${LAYERS:-$DEF_LAYERS}"
[ -z "$LAYERS" ] && { echo "STAGE=$STAGE has no default layer set — pass LAYERS="; exit 1; }

TAG="${TAG:-stage${STAGE}}"
MODEL_DIR="${MODEL_DIR:-google/gemma-4-E4B-it-qat-q4_0-unquantized}"
DENSE_KIND="${DENSE_KIND:-down_proj}"
LR="${LR:-5e-4}"
ALPHA="${ALPHA:-0.5}"
BETA="${BETA:-0.2}"                 # stop-anchor hinge
MARGIN="${MARGIN:-1.0}"
MARGIN_HI="${MARGIN_HI:-0.1}"
CLIP="${CLIP:-0.25}"
# accum 1 at a 32768 window is the Bonsai precedent (run_kd_anchor_qat.sh) and gives
# 651 steps per epoch here -- fine-grained enough that a 60-step arm is a real read and
# that --probe-every 25 samples termination often enough to abort before a collapse
# finishes. accum 4 would make one step 131k tokens and a 60-step arm a blunt instrument.
EPOCHS="${EPOCHS:-1.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
ABORT="${ABORT:-0.03}"              # ~11x gemma's 0.00274 diagnostic
ABORT_CTRL="${ABORT_CTRL:-0.01}"    # last-ditch floor under gemma's 0.0703 control
CORPUS="${CORPUS:-out/gemma4-ternary/corpus_sft_gemma4_32768.pt}"
VAL="${VAL:-out/gemma4-ternary/corpus_sft_gemma4_val_32768.pt}"
TABLE="${TABLE:-out/gemma4-ternary/kd/gemma31b_topk64_fs106.pt}"
OUT="${OUT:-out/gemma4-ternary/${TAG}}"
TEACHER_PROBE="${TEACHER_PROBE:-out/gemma4-ternary/kd/teacher_probe_31b.json}"

[ -f "$TABLE" ] || { echo "[$TAG] no KD table at $TABLE"; exit 1; }
# One card. Never launch beside another trainer or a torch eval.
while pgrep -f "quant_tuner[.]qat[.]train" > /dev/null; do sleep 30; done
while pgrep -f "kd[_]precompute" > /dev/null; do sleep 60; done
mkdir -p "$OUT"
cp "$TEACHER_PROBE" "$OUT/teacher_probe.json" 2>/dev/null || true

echo "[$TAG] layers=$LAYERS dense=$DENSE_KIND lr=$LR alpha=$ALPHA beta=$BETA clip=$CLIP" \
     "epochs=$EPOCHS model=$MODEL_DIR -> $OUT"
.venv/bin/python -m quant_tuner.qat.train \
    --model-dir "$MODEL_DIR" \
    --corpus "$CORPUS" --val-corpus "$VAL" \
    --layers "$LAYERS" --ternary-layers "$LAYERS" --dense-kind "$DENSE_KIND" \
    --kd-table "$TABLE" --kd-alpha "$ALPHA" --kd-temp 1.0 \
    --stop-anchor "$BETA" --stop-anchor-margin "$MARGIN" \
    --stop-anchor-margin-hi "$MARGIN_HI" \
    --steer-weight 0 \
    --clip-norm "$CLIP" --lr-scale group-scale \
    --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
    --grad-accum "$GRAD_ACCUM" --epochs "$EPOCHS" --lr "$LR" --warmup-frac 0.05 \
    --val-every 50 --probe-every 25 \
    --probe-abort "$ABORT" --probe-abort-control "$ABORT_CTRL" --probe-abort-patience 2 \
    --ckpt-every 100 --ckpt-keep 2 \
    --out "$OUT" > "$OUT/train.log" 2>&1
rc=$?
echo "[$TAG] trainer rc=$rc"
grep "PROBE-ABORT" "$OUT/train.log" 2>/dev/null
tail -3 "$OUT/train.log"

# A checkpoint exists even after a probe abort (that is the point of aborting), and its
# damage number is exactly what says whether the abort was early-termination or the
# stage genuinely failing to recover. So measure either way, and let the caller gate on
# the trainer's rc.
# The trainer writes trained_latents.pt (atomically replaced) plus step-stamped hard
# links beside it -- there is no ckpt-*.pt.
CKPT="$OUT/trained_latents.pt"
[ -f "$CKPT" ] || CKPT=""
if [ -n "$CKPT" ]; then
    echo "[$TAG] measuring damage from $CKPT (CPU)"
    .venv/bin/python scripts/gemma4_stage_damage.py \
        --ternary-layers "$LAYERS" --dense-kind "$DENSE_KIND" --ckpt "$CKPT" \
        --out "$OUT/stage_damage_trained.json" 2>&1 | tail -6
else
    echo "[$TAG] no checkpoint written — nothing to measure"
fi
exit $rc

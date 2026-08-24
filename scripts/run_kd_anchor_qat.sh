#!/usr/bin/env bash
# Full-schedule KD + stop-anchor QAT run — the configuration the anchor ladder converged
# on, every lever env-overridable. This is the FULL-RUN counterpart of run_kd_qat.sh
# (which runs 59-step smoke arms): dual probe guards make the full schedule its own gate,
# a failed run costs ~2 h and a saved checkpoint, a surviving one is the artifact.
#
#     TAG=anchor3 bash scripts/run_kd_anchor_qat.sh                 # the ladder's best
#     TAG=anchor4 LR=4e-4 bash scripts/run_kd_anchor_qat.sh        # lower-peak variant
#     TAG=x BETA=0.5 MARGIN_HI=0.05 bash scripts/run_kd_anchor_qat.sh
#
# The ladder that produced these defaults (docs/ternary_qat_curriculum.md):
#   alpha 0.5 / 0.75 alone     -> collapse at step 125 / 150 (KL's restoring force on the
#                                 stop logit is P_s−P_t: vanishes exactly when needed)
#   + symmetric L1 anchor      -> diagnostic PINNED, control collapsed (176:1 direction bias)
#   + one-sided hinge          -> control collapsed INSIDE the 1-nat band (a nat below
#                                 P=0.99999 is P=0.37)
#   + per-side margins 1.0/0.1 -> control oscillates 0.93-1.00; weak TAIL remains
#                                 (~10-15% of stop positions; loops in the agent)
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:?set TAG=<run-tag>}"
LR="${LR:-5e-4}"
ALPHA="${ALPHA:-0.5}"
BETA="${BETA:-0.2}"
MARGIN="${MARGIN:-1.0}"
MARGIN_HI="${MARGIN_HI:-0.1}"
EPOCHS="${EPOCHS:-1.0355}"          # 613 steps over the 592-window corpus
ABORT="${ABORT:-0.09}"
ABORT_CTRL="${ABORT_CTRL:-0.95}"
STEER="${STEER:-0}"
REP="${REP:-0}"                     # repetition steering (qat/steer.py RepBatch); 0.05 = gentle
REP_CAP="${REP_CAP:-0.5}"           # hinge cap on mean per-token P(verbatim repeat)
REP_K="${REP_K:-1}"                 # identical rounds in rep contexts (measured: escalation lives at k>=2)
REP_KD="${REP_KD:-}"                # RepKD table (capture_rep_teacher.py) -> teacher-KL at loop states
REP_KD_W="${REP_KD_W:-0.1}"
REP_BANK="${REP_BANK:-}"              # real-material context bank (build_rep_bank.py)
REP_N="${REP_N:-6}"
REP_TRAJ="${REP_TRAJ:-}"            # harvested full-prefix episode contexts (anchor10)
REP_TRAJ_N="${REP_TRAJ_N:-4}"
REP_TRAJ_EVERY="${REP_TRAJ_EVERY:-4}"
CLIP="${CLIP:-1.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"    # accum N = 1/N optimizer steps at the flip-viable LR (total-drift lever for big corpora)
CORPUS="${CORPUS:-out/exp-058/fixed/corpus_ourssft_32768.pt}"
VAL="${VAL:-out/exp-058/fixed/corpus_ourssft_val_32768.pt}"
TABLE="${TABLE:-out/exp-058/kd/ourssft_8b_topk64_fs151645.pt}"   # forced-stop table
OUT="${OUT:-out/exp-058/kd8b-full-${TAG}}"
TEACHER_PROBE="${TEACHER_PROBE:-out/exp-058/kd8b-full/teacher_probe.json}"  # asymptotes for the report

while pgrep -f "quant_tuner[.]qat[.]train" > /dev/null; do sleep 30; done
free_gb=$(df --output=avail -BG "$(pwd)" | tail -1 | tr -dc 0-9)
[ "$free_gb" -lt 60 ] && { echo "[$TAG] only ${free_gb}G free — need ~60G"; exit 1; }
mkdir -p "$OUT"
cp "$TEACHER_PROBE" "$OUT/teacher_probe.json" 2>/dev/null || true

echo "[$TAG] lr=$LR alpha=$ALPHA anchor beta=$BETA margins=$MARGIN/$MARGIN_HI steer=$STEER rep=$REP clip=$CLIP table=$(basename "$TABLE") -> $OUT"
PYTHONPATH=src .venv/bin/python -m quant_tuner.qat.train \
    --corpus "$CORPUS" --val-corpus "$VAL" \
    --kd-table "$TABLE" --kd-alpha "$ALPHA" --kd-temp 1.0 \
    --stop-anchor "$BETA" --stop-anchor-margin "$MARGIN" \
    --stop-anchor-margin-hi "$MARGIN_HI" \
    --steer-weight "$STEER" --steer-rep-weight "$REP" --steer-rep-cap "$REP_CAP" \
    --steer-rep-k "$REP_K" \
    ${REP_KD:+--steer-rep-kd "$REP_KD" --steer-rep-kd-weight "$REP_KD_W"} \
    ${REP_BANK:+--steer-rep-bank "$REP_BANK"} --steer-rep-n "$REP_N" \
    ${REP_TRAJ:+--steer-rep-traj "$REP_TRAJ" --steer-rep-traj-n "$REP_TRAJ_N" --steer-rep-traj-every "$REP_TRAJ_EVERY"} \
    --clip-norm "$CLIP" \
    --lr-scale group-scale \
    --train-layers 36 --optim adafactor --dtype fp32 \
    --compute-dtype fp32 --matmul-precision high \
    --grad-accum "$GRAD_ACCUM" --epochs "$EPOCHS" --lr "$LR" --warmup-frac 0.05 \
    --stop-weight 1.0 --grad-spike-factor 0 \
    --val-every 50 --probe-every 25 \
    --probe-abort "$ABORT" --probe-abort-control "$ABORT_CTRL" \
    --ckpt-every 100 --ckpt-keep 2 \
    --out "$OUT" > "$OUT/train.log" 2>&1
rc=$?
echo "[$TAG] finished rc=$rc"
grep "PROBE-ABORT" "$OUT/train.log" 2>/dev/null
tail -3 "$OUT/train.log"
# propagate the TRAINER's rc — the greps above otherwise make a crashed run exit 0,
# and a chain gated on this script's status happily benches a nonexistent checkpoint
exit $rc

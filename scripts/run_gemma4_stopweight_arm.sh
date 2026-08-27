#!/usr/bin/env bash
# Restore the stop-signal density gemma's chat template dilutes away.
#
# WHY, measured rather than guessed. gemma renders an entire tool exchange as ONE model
# turn, so a session with twenty tool calls carries twenty <|im_end|> stop targets under
# ChatML and exactly ONE <turn|> here. The corpora bear it out:
#
#     this corpus        1 stop target per 972 supervised tokens
#     Bonsai sft8k       1 per 176            -> 5.5x denser
#
# and --stop-weight has been at its default 1.0 in every arm run so far. STOP_WEIGHT
# defaults to 5.5 = 972/176, i.e. the value that makes a stop decision carry the same
# share of the loss it carried in the recipe these hyperparameters came from. Not a
# sweep — a unit conversion.
#
# Note the Bonsai finding does NOT contradict this. There, stop-weight 6.0 vs 1.0 moved
# the diagnostic by 0.02 and was written off — but Bonsai's failure was stopping too
# EAGERLY, which up-weighting the stop target cannot fix. Ours is the opposite failure.
#
# Read the result with gemma4_stop_on_corpus.py, NOT the seven-prompt probe: on real
# held-out stop targets the shipped model commits at 35%, a DENSE fine-tune at 3%, and
# the ternary CE-only arm at 0% — while that same seven-prompt probe called the dense
# fine-tune healthier than shipped.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
SW="${STOP_WEIGHT:-5.5}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-0.0922}"
OUT="${OUT:-$R/sw${SW}-lr${LR}}"
TABLE="$R/kd/e4bself_topk64_fs106.pt"

say() { echo "[sw $(date +%H:%M:%S)] $*"; }
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do
    sleep 30
done
mkdir -p "$OUT"
say "self-KD + --stop-weight $SW, lr $LR, ${EPOCHS} epoch(s) -> $OUT"
.venv/bin/python -m quant_tuner.qat.train \
    --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --corpus "$R/corpus_sft_gemma4_32768.pt" --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
    --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --kd-table "$TABLE" --kd-alpha 0.5 --kd-temp 1.0 \
    --stop-weight "$SW" \
    --stop-anchor 0.2 --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1 \
    --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
    --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs "$EPOCHS" --lr "$LR" --warmup-frac 0.05 \
    --val-every 25 --probe-every 25 \
    --probe-abort 0 --probe-abort-control 0 \
    --ckpt-every 25 --ckpt-keep 2 \
    --out "$OUT" > "$OUT/train.log" 2>&1
say "trainer rc=$?"
grep -a "VAL masked-CE" "$OUT/train.log" | tail -1

# The read-out that matters. ~17 min CPU, safe beside whatever runs next on the GPU.
[ -f "$OUT/trained_latents.pt" ] && .venv/bin/python scripts/gemma4_stop_on_corpus.py \
    --n 40 --ctx 2048 --threads 48 --label "sw$SW" --out "$R/stopcorpus/sw$SW.json" \
    --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --ckpt "$OUT/trained_latents.pt" 2>&1 | grep -av "^Loading"
say "done"

#!/usr/bin/env bash
# The control the study was missing: train the SAME six layers with NO ternarization.
#
# Three ternary arms all scored "recovered" between -258% and -408% against the shipped
# model — the CE-only one, with no teacher at all, worst of the three. That ordering is
# the tell: KLD-vs-shipped cannot separate "ternarization damage failed to recover" from
# "the model is learning the training distribution", and every fine-tune does the second
# whether or not it does the first.
#
# So: identical layers, corpus, lr, steps and seed, with `--dense-kind` naming every
# tensor kind, which leaves each linear trainable but OFF the grid (wrap_model refuses a
# trainable layer that is not also ternarized, so this is how a dense arm is expressed).
# Its KLD is the floor any ternary arm should be read against, and it doubles as a
# reference for `gemma4_stage_damage.py --ref-ckpt`.
#
# If the dense arm also lands near 0.38, ternarization is not what those numbers measured.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
OUT="${OUT:-$R/dense-control-lr2e-4}"
LR="${LR:-2e-4}"

say() { echo "[dense-control $(date +%H:%M:%S)] $*"; }
say "waiting for the gpu"
while pgrep -f "kd_precomput[e].py" > /dev/null; do sleep 60; done
while pgrep -f "quant_tuner[.]qat[.]train" > /dev/null; do sleep 30; done

mkdir -p "$OUT"
say "training 0,1,2,3,7,8 DENSE (no ternarization) at lr=$LR"
.venv/bin/python -m quant_tuner.qat.train \
    --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --corpus "$R/corpus_sft_gemma4_32768.pt" --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
    --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 \
    --dense-kind _proj --dense-kind gate \
    --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
    --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs 0.0922 --lr "$LR" --warmup-frac 0.05 \
    --val-every 25 --probe-every 25 \
    --probe-abort 0 --probe-abort-control 0 \
    --ckpt-every 25 --ckpt-keep 2 \
    --out "$OUT" > "$OUT/train.log" 2>&1
say "rc=$?"
grep -a "STOPPROBE" "$OUT/train.log" | tail -2

# Its own damage vs the shipped model — the floor to read the ternary arms against.
[ -f "$OUT/trained_latents.pt" ] && .venv/bin/python scripts/gemma4_stage_damage.py --probe \
    --ternary-layers 0,1,2,3,7,8 --dense-kind _proj --dense-kind gate \
    --ckpt "$OUT/trained_latents.pt" --out "$OUT/stage_damage_trained.json" 2>&1 | tail -6
say "done"

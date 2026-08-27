#!/usr/bin/env bash
# The full stage-1 run: self-KD, one epoch, guards armed.
#
# Chosen after five 60-step arms established: at matched training a dense fine-tune and a
# ternary one reach the SAME held-out CE (1.7796 vs 1.7290), so ternarization costs no
# measurable capability; termination is the failure, and it is an interaction that
# neither ternarization nor training produces alone; and self-KD mitigates it 10x over
# CE-only (control 0.0399 vs 0.0039) because its teacher's stop policy IS the target.
#
# NOTE ON THE WAIT LOOP. It greps for the CONFIG file of a running trainer, not for
# "quant_tuner.qat.train": this script's own command line contains that string as an
# argument, so `pgrep -f` matches itself and waits forever. Bracketing the pattern does
# not help — the collision is with the literal text of the command being launched, not
# with the pattern. That mistake cost 1.6 h of idle GPU here.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1.0}"
OUT="${OUT:-$R/selfkd-full-lr${LR}}"
TABLE="$R/kd/e4bself_topk64_fs106.pt"

say() { echo "[selfkd-full $(date +%H:%M:%S)] $*"; }
[ -f "$TABLE" ] || { say "FATAL: no self-KD table"; exit 1; }

# Wait on GPU occupancy rather than on a process name — no pattern, nothing to self-match.
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do
    sleep 60
done
say "gpu free — training 0,1,2,3,7,8 ternary + self-KD, ${EPOCHS} epoch(s) at lr=$LR"

mkdir -p "$OUT"
.venv/bin/python -m quant_tuner.qat.train \
    --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --corpus "$R/corpus_sft_gemma4_32768.pt" --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
    --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --kd-table "$TABLE" --kd-alpha 0.5 --kd-temp 1.0 \
    --stop-anchor 0.2 --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1 \
    --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
    --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs "$EPOCHS" --lr "$LR" --warmup-frac 0.05 \
    --val-every 50 --probe-every 25 \
    --probe-abort 0.03 --probe-abort-control 0.01 --probe-abort-patience 2 \
    --ckpt-every 100 --ckpt-keep 3 \
    --out "$OUT" > "$OUT/train.log" 2>&1
rc=$?
say "trainer rc=$rc"
grep -a "PROBE-ABORT" "$OUT/train.log" 2>/dev/null

[ -f "$OUT/trained_latents.pt" ] && .venv/bin/python scripts/gemma4_stage_damage.py --probe \
    --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --ckpt "$OUT/trained_latents.pt" --out "$OUT/stage_damage_trained.json" 2>&1 | tail -6
say "done"

#!/usr/bin/env bash
# The stop ANCHOR, turned up until it can actually move something.
#
# Every KD arm so far already carried `--stop-anchor 0.2`, and its telemetry says the
# mechanism works and the gain is ~100x too small: `an` climbs 0.0008 -> 0.15-0.47 over
# 60 steps (the student drifting away from the teacher's stop level, hinge engaging)
# while contributing 0.2*0.2 = 0.04 against a loss of 1.2-2.0. It loses, quietly.
#
# Why the anchor and not more stop-weight. The two levers differ in where they stop:
#   stop-weight is UNBOUNDED — it keeps pushing P(stop) up forever, so the value that
#   would fix commitment also wrecks everything else. Measured: 5.5 -> val 2.0478
#   (learning: 2.4918 -> 2.0478), 16 -> val 2.6457 and essentially flat (2.7072 ->
#   2.6457). Commitment 5.0% at 5.5.
#   the anchor is ONE-SIDED and SATURATING — it pushes only until the student is within
#   `margin_hi` of the TEACHER's own log P(stop) at that position, then goes silent. The
#   teacher here is the shipped model, whose stopping policy is the thing we are trying
#   not to destroy (corpus-conditioned: 0.530 at real stop targets, 0.00053 elsewhere,
#   ratio 993). So a large beta cannot overshoot into a stop-happy model the way a large
#   stop-weight can; it converges to shipped behaviour and stops.
#
# beta=8 puts the initial anchor contribution (~0.2 * 8 = 1.6) alongside CE (~1.2-2.0),
# decaying to zero as the hinge closes.
#
# Two arms, because the attribution matters for the full stage:
#   anchor8-sw5.5  everything that worked, with the anchor live. stop-weight 5.5 bought
#                  the DISCRIMINATION (elsewhere 9.1e-4 -> 4.3e-5, ratio 87 -> 1798) for
#                  +0.05 val CE, so it stays unless the anchor subsumes it.
#   anchor8        anchor alone. Its continue-side hinge (margin_lo 1.0 nat) is also a
#                  brake on over-stopping, so it may buy both halves by itself — in which
#                  case the full stage drops a hyperparameter.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
TABLE="$R/kd/e4bself_topk64_fs106.pt"
say() { echo "[anchor $(date +%H:%M:%S)] $*"; }

arm() {
    local name="$1"; shift
    local out="$R/$name"
    if [ -f "$out/trained_latents.pt" ]; then say "$name already trained — skipping"; return 0; fi
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do
        sleep 30
    done
    mkdir -p "$out"
    say "=== $name ==="
    .venv/bin/python -m quant_tuner.qat.train \
        --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
        --corpus "$R/corpus_sft_gemma4_32768.pt" \
        --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
        --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 \
        --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
        --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
        --grad-accum 1 --epochs 0.0922 --lr 2e-4 --warmup-frac 0.05 \
        --val-every 25 --probe-every 25 \
        --probe-abort 0 --probe-abort-control 0 \
        --ckpt-every 25 --ckpt-keep 2 \
        --kd-table "$TABLE" --kd-temp 1.0 --kd-alpha 0.5 \
        --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1 \
        "$@" --out "$out" > "$out/train.log" 2>&1
    say "$name rc=$? val=$(grep -a 'VAL masked-CE' "$out/train.log" | tail -1 | grep -oE 'masked-CE [0-9.]+' | cut -d' ' -f2)"
}

measure() {
    local name="$1"; shift
    local out="$R/$name"
    [ -f "$out/trained_latents.pt" ] || return 0
    [ -f "$R/stopcorpus/$name.json" ] && return 0
    .venv/bin/python scripts/gemma4_stop_on_corpus.py --n 40 --ctx 2048 --threads 48 \
        --label "$name" --out "$R/stopcorpus/$name.json" \
        --ternary-layers 0,1,2,3,7,8 "$@" --ckpt "$out/trained_latents.pt" \
        2>&1 | grep -av "^Loading" | grep -aE "AT a real|elsewhere|ratio"
}

arm anchor8-sw5.5-lr2e-4 --stop-anchor 8 --stop-weight 5.5 --dense-kind down_proj
measure anchor8-sw5.5-lr2e-4 --dense-kind down_proj &

arm anchor8-lr2e-4       --stop-anchor 8 --dense-kind down_proj
measure anchor8-lr2e-4   --dense-kind down_proj &

wait
say "anchor arms done"

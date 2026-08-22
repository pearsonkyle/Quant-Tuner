#!/usr/bin/env bash
# Back-to-back arms so the card never idles waiting for me to queue the next one.
#
# Where the sweep starts from (60-step arms, real-corpus read-out):
#   shipped                      commits 35%   ratio 2,054x
#   self-KD                      commits  3%   ratio    87x   val 1.9948
#   self-KD + stop-weight 5.5    commits  5%   ratio 1,798x   val 2.0478
#
# stop-weight 5.5 recovered nearly all the DISCRIMINATION (the model stopped putting
# stop-mass in wrong places: elsewhere 0.000906 -> 0.000043) but almost none of the
# COMMITMENT. So the remaining question is what raises P(stop) at a real stop target,
# and the arms test the two candidates plus one falsification:
#
#   sw16      more of the same lever. If commitment tracks stop-weight, it is a
#             loss-share problem and there is a value that fixes it. If it saturates
#             around 5%, up-weighting is the wrong lever and the sweep says so.
#   a75-sw5.5 kd-alpha 0.75 instead of 0.5 — pull harder toward the dense model's own
#             distribution, which is where the 35% commitment lives.
#   dense-sw5.5  THE FALSIFICATION. The diagnosis says the commitment deficit is a
#             corpus/objective problem that hits dense and ternary alike, so stop-weight
#             must help a DENSE fine-tune about as much. If it does not, the diagnosis
#             is wrong and the deficit is quantization-specific after all.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
TABLE="$R/kd/e4bself_topk64_fs106.pt"
say() { echo "[sweep $(date +%H:%M:%S)] $*"; }

arm() {  # name, then extra trainer args
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
        "$@" --out "$out" > "$out/train.log" 2>&1
    say "$name rc=$? val=$(grep -a 'VAL masked-CE' "$out/train.log" | tail -1 | grep -oE 'masked-CE [0-9.]+' | cut -d' ' -f2)"
}

measure() {  # name, then the --dense-kind flags that arm used
    local name="$1"; shift
    local out="$R/$name"
    [ -f "$out/trained_latents.pt" ] || return 0
    [ -f "$R/stopcorpus/$name.json" ] && return 0
    .venv/bin/python scripts/gemma4_stop_on_corpus.py --n 40 --ctx 2048 --threads 48 \
        --label "$name" --out "$R/stopcorpus/$name.json" \
        --ternary-layers 0,1,2,3,7,8 "$@" --ckpt "$out/trained_latents.pt" \
        2>&1 | grep -av "^Loading" | grep -aE "AT a real|elsewhere|ratio"
}

KD=(--kd-table "$TABLE" --kd-temp 1.0 --stop-anchor 0.2 \
    --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1)

arm sw16-lr2e-4        "${KD[@]}" --kd-alpha 0.5  --stop-weight 16 --dense-kind down_proj
measure sw16-lr2e-4    --dense-kind down_proj &

arm a75-sw5.5-lr2e-4   "${KD[@]}" --kd-alpha 0.75 --stop-weight 5.5 --dense-kind down_proj
measure a75-sw5.5-lr2e-4 --dense-kind down_proj &

arm dense-sw5.5-lr2e-4 --stop-weight 5.5 --dense-kind _proj --dense-kind gate
measure dense-sw5.5-lr2e-4 --dense-kind _proj --dense-kind gate &

wait
say "sweep done"

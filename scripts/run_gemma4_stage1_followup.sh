#!/usr/bin/env bash
# What arm 1+2 turned the study into: isolate the schedule question from the teacher.
#
# Both lr arms collapsed the stop CONTROL toward the 31B teacher's own 0.0000 (arm 1
# 0.0703->0.0041 at lr 2e-4, arm 2 ->0.0000 at 5e-4) while KLD vs dense rose. Two
# hypotheses were confounded in those runs and are separated here, cheapest first:
#
#   ce-only   no KD table at all. If damage recovers and termination holds, the
#             SCHEDULE works and the teacher was the problem. If it fails too, the
#             problem is deeper than the teacher. ~50 min, needs nothing new.
#   self-KD   teacher = the dense E4B itself. Removes BOTH the termination mismatch
#             (the teacher's stop policy becomes the target policy) and the KLD
#             confound (the metric's reference becomes the KD target). This is the
#             control arm declared in notes.md before launch. ~1.6 h to build.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
say() { echo "[followup $(date +%H:%M:%S)] $*"; }

say "waiting for the running stage to finish"
while pgrep -f "run_gemma4_stag[e].sh" > /dev/null; do sleep 30; done
while pgrep -f "quant_tuner[.]qat[.]train" > /dev/null; do sleep 30; done
say "gpu free"

# ---- 1. CE-only, no teacher ---------------------------------------------------------
OUT="$R/ce-only-lr2e-4"
if [ ! -f "$OUT/trained_latents.pt" ]; then
    say "CE-only arm (no KD table, no stop anchor)"
    mkdir -p "$OUT"
    .venv/bin/python -m quant_tuner.qat.train \
        --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
        --corpus "$R/corpus_sft_gemma4_32768.pt" --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
        --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
        --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
        --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
        --grad-accum 1 --epochs 0.0922 --lr 2e-4 --warmup-frac 0.05 \
        --val-every 25 --probe-every 25 \
        --probe-abort 0.03 --probe-abort-control 0.01 --probe-abort-patience 2 \
        --ckpt-every 25 --ckpt-keep 2 \
        --out "$OUT" > "$OUT/train.log" 2>&1
    say "CE-only rc=$? — $(grep -c PROBE-ABORT "$OUT/train.log" 2>/dev/null) abort(s)"
fi
# CPU, so it overlaps the precompute below rather than delaying it.
[ -f "$OUT/trained_latents.pt" ] && \
    OMP_NUM_THREADS=64 nohup .venv/bin/python scripts/gemma4_stage_damage.py \
        --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj --threads 64 --probe \
        --ckpt "$OUT/trained_latents.pt" --out "$OUT/stage_damage_trained.json" \
        > "$OUT/damage.log" 2>&1 &

# ---- 2. the self-KD table ------------------------------------------------------------
TABLE="$R/kd/e4bself_topk64_fs106.pt"
if [ ! -f "$TABLE" ]; then
    say "self-KD precompute (teacher = the dense E4B itself)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/kd_precompute.py \
        --teacher google/gemma-4-E4B-it-qat-q4_0-unquantized \
        --corpus "$R/corpus_sft_gemma4_32768.pt" \
        --out "$TABLE" --topk 64 --dtype bf16 \
        --student-model google/gemma-4-E4B-it-qat-q4_0-unquantized --include-ids 106 \
        > "$R/kd/precompute_e4bself.log" 2>&1
    say "precompute rc=$?"
fi
[ -f "$TABLE" ] || { say "FATAL: no self-KD table"; exit 1; }
.venv/bin/python scripts/verify_kd_table.py --table "$TABLE" \
    --corpus "$R/corpus_sft_gemma4_32768.pt" --stop-id 106 || { say "FATAL: bad table"; exit 1; }
.venv/bin/python scripts/kd_stop_signal.py --table "$TABLE" \
    --corpus "$R/corpus_sft_gemma4_32768.pt" --out "$R/kd/stop_signal_e4bself.json"
say "self-KD table ready — arms next"

#!/usr/bin/env bash
# Stage 1 at full length: layers 0,1,2,3,7,8 for one epoch over the 651-window corpus.
#
#   RECIPE_ARGS="--stop-anchor 8 --stop-weight 5.5" TAG=anchor8 bash scripts/run_gemma4_stage1_full.sh
#
# The 60-step arms choose the recipe; this is the run the stage-1 verdict is read from.
# ~47 s/step x 651 = ~8.5 h, so it is an overnight job and the point of the sidecar below
# is that it does not have to run to the end to be worth killing.
#
# The sidecar reads COMMITMENT off each new checkpoint on CPU (the GPU is full), because
# masked-CE cannot see the failure this pipeline keeps hitting: sft32k's validation went
# flat for 225 steps while its stopping policy collapsed. If commitment falls across
# checkpoints while val improves, kill the run — that is the shape of the failure, and
# only the sidecar can see it.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
TAG="${TAG:?set TAG}"
RECIPE_ARGS="${RECIPE_ARGS:?set RECIPE_ARGS, e.g. --stop-anchor 8 --stop-weight 5.5}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-2e-4}"
OUT="$R/stage1-$TAG"
say() { echo "[stage1 $(date +%H:%M:%S)] $*"; }

mkdir -p "$OUT" "$R/stopcorpus"
rm -f "$OUT/.done"   # else a re-run's sidecar exits before the first checkpoint
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do
    sleep 60
done

# --- CPU sidecar: commitment per checkpoint, newest first, one at a time --------------
# Reads .stepN.pt only (never trained_latents.pt, which the trainer rewrites in place).
sidecar() {
    local seen=""
    while [ ! -f "$OUT/.done" ]; do
        for c in "$OUT"/trained_latents.step*.pt; do
            [ -e "$c" ] || continue
            local n; n=$(basename "$c" .pt); n=${n#trained_latents.}
            case " $seen " in *" $n "*) continue;; esac
            seen="$seen $n"
            say "sidecar: reading $TAG/$n"
            .venv/bin/python scripts/gemma4_stop_on_corpus.py --n 40 --ctx 2048 \
                --threads 40 --label "$TAG-$n" --out "$R/stopcorpus/$TAG-$n.json" \
                --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj --ckpt "$c" \
                > "$OUT/sidecar-$n.log" 2>&1
            say "sidecar $n: $(.venv/bin/python -c "
import json;d=json.load(open('$R/stopcorpus/$TAG-$n.json'))
print(f\"commit {d['at_stop_target']['frac_top1']:.1%} ratio {d['ratio_mean']:,.0f}\")" 2>/dev/null)"
        done
        sleep 120
    done
}
sidecar &
SIDECAR=$!

say "=== stage1-$TAG epochs=$EPOCHS lr=$LR args: $RECIPE_ARGS ==="
.venv/bin/python -m quant_tuner.qat.train \
    --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --corpus "$R/corpus_sft_gemma4_32768.pt" \
    --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
    --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
    --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs "$EPOCHS" --lr "$LR" --warmup-frac 0.03 \
    --kd-table "$R/kd/e4bself_topk64_fs106.pt" --kd-temp 1.0 --kd-alpha 0.5 \
    --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1 \
    --val-every 50 --probe-every 50 \
    --probe-abort 0 --probe-abort-control 0 \
    --ckpt-every 100 --ckpt-keep 8 \
    $RECIPE_ARGS --out "$OUT" > "$OUT/train.log" 2>&1
RC=$?
touch "$OUT/.done"
say "stage1-$TAG rc=$RC val=$(grep -a 'VAL masked-CE' "$OUT/train.log" | tail -1 | grep -oE 'masked-CE [0-9.]+' | cut -d' ' -f2)"
wait "$SIDECAR" 2>/dev/null

[ "$RC" -eq 0 ] || exit "$RC"

# --- the two deliverable measurements -------------------------------------------------
.venv/bin/python scripts/gemma4_stop_on_corpus.py --n 40 --ctx 2048 --threads 48 \
    --label "stage1-$TAG" --out "$R/stopcorpus/stage1-$TAG.json" \
    --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj --ckpt "$OUT/trained_latents.pt" \
    > "$OUT/stopcorpus.log" 2>&1
say "final: $(.venv/bin/python -c "
import json;d=json.load(open('$R/stopcorpus/stage1-$TAG.json'))
print(f\"commit {d['at_stop_target']['frac_top1']:.1%} ratio {d['ratio_mean']:,.0f}\")")"

# Damage against the DENSE fine-tune, not the shipped model: a dense fine-tune on this
# corpus moves 0.2175 from shipped all by itself, so KLD-vs-shipped scores fine-tuning
# drift and ternarization damage as one number and is dominated by the drift.
.venv/bin/python scripts/gemma4_stage_damage.py \
    --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj --windows 12 --threads 48 \
    --ckpt "$OUT/trained_latents.pt" \
    --ref-ckpt "$R/dense-control-lr2e-4/trained_latents.pt" \
    --out "$OUT/stage_damage_vs_dense.json" > "$OUT/damage.log" 2>&1
say "damage: $(.venv/bin/python -c "
import json;d=json.load(open('$OUT/stage_damage_vs_dense.json'))['rows']
print(' '.join(f\"{k}={v['kld']:.4f}\" for k,v in d.items()))")"
say "stage1-$TAG done"

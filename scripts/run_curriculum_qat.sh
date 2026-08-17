#!/usr/bin/env bash
# The three-round ternary-QAT curriculum, on CUDA.
#
#     bash scripts/run_curriculum_qat.sh [TAG_PREFIX] [LR] [EPOCHS_PER_ROUND]
#
# Each round CONTINUES from the previous round's latents, so this is one long fine-tune
# with a changing data distribution, not three independent runs:
#
#   round 1  ultrachat_200k      broad conversational grounding — no tools, no reasoning
#   round 2  distillation        tools + agents + reasoning (sft_tools, sft_science)
#   round 3  our universal SFT   CLI logs + agent trajectories that actually resolve issues
#
# The ordering is general -> capability -> in-domain, so the last thing the model sees is
# the distribution it is graded on. Catastrophic forgetting runs the other way, and that is
# the point: round 3 is the one whose behaviour we want at the end.
#
# WHY EACH ROUND IS BUDGETED. ultrachat alone is ~180M tokens; at a 32768 window that is
# ~5,500 steps, ~96 h. Rounds are therefore capped by TOKEN BUDGET rather than epochs so
# each takes a comparable slice of wall-clock and no single round dominates the schedule.
# Budgets are per-source and passed to build_sft_qat_corpus.py.
#
# HARD PREREQUISITE — read before running: this chains off whatever `--resume` points at.
# Starting round 1 from a checkpoint that is not the intended base silently produces a
# model with a different history than its name claims. RESUME_FROM must be either empty
# (start from the shipped vanilla weights) or a latents file you have verified.
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX="${1:-curriculum}"
LR="${2:-5e-4}"
EPOCHS="${3:-1.0}"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY=.venv/bin/python

WINDOW="${WINDOW:-32768}"
ACCUM="${ACCUM:-1}"
# TF32: latents, the TWN threshold and ternarize_group stay elementwise fp32 and bit-exact,
# so the CODES a step produces are unperturbed — only matmul accumulation is reduced (10
# mantissa bits vs bf16's 8). Worth ~1.38x at this window. This is the speedup that is safe
# to take; --compute-dtype bf16 is NOT (it diverged twice, see Obs. 8 in
# docs/ternary_qat_sft32k_study.md), and this script deliberately does not expose it.
PRECISION="${PRECISION:-high}"
COMPUTE_DTYPE=fp32
STOP_WEIGHT="${STOP_WEIGHT:-1.0}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-12288}"   # scale with the window (3072 at 8064)
MIN_DENSITY="${MIN_DENSITY:-0.05}"
VAL_EVERY="${VAL_EVERY:-25}"
CKPT_EVERY="${CKPT_EVERY:-50}"

# Per-round token budgets, as the `SOURCE=TOKENS` pairs build_sft_qat_corpus.py takes
# (there is no global cap — a budget names a source). Only round 1 needs one: ultrachat is
# ~180M tokens, which at a 32768 window is ~5,500 steps / ~96 h on its own. 20M matches the
# ~20M the sft32k run trained on, so each round is a comparable slice of wall-clock.
# Rounds 2 and 3 are already ~14M and ~20M, so they are taken whole.
BUDGET_R1="${BUDGET_R1:-ultrachat=20000000}"
BUDGET_R2="${BUDGET_R2:-}"
BUDGET_R3="${BUDGET_R3:-}"

SFT_R1="${SFT_R1:-out/corpora/round1-ultrachat/sft.jsonl.gz}"
SFT_R2="${SFT_R2:-out/corpora/round2-distill/sft.jsonl.gz}"
SFT_R3="${SFT_R3:-out/corpora/qwen3-universal-v2/sft.jsonl.gz}"

# ROUND 3 REUSES THE ABLATION'S EXACT CORPUS, by symlink:
#   out/exp-058/corpus_ourssft_32768.pt -> out/exp-058/sft_corpus_universal_32768.pt
# The headline question this curriculum answers is "do three rounds beat one round of our
# SFT?", i.e. curriculum-r3 against sft32k_sw1. Rebuilding round 3 at MAX_TOOL_TOKENS
# 12288 when sw1 was packed at 8192 would change the DATA as well as the training history,
# and the comparison could no longer attribute a difference to the curriculum. build_corpus
# below skips any corpus that already exists, so the symlink is what it picks up
# (fingerprint 5a2d5d65f640fb74, identical to sw1's). 12288 is still right for round 2 —
# it is a different source, compared against nothing.

RESUME_FROM="${RESUME_FROM:-}"

build_corpus() {   # name sft_path budget-pairs
    local name="$1" sft="$2" budget="$3"
    local out="out/exp-058/corpus_${name}_${WINDOW}.pt"
    local val="out/exp-058/corpus_${name}_val_${WINDOW}.pt"
    local bargs=() pair
    # budget is a space-separated list of SOURCE=TOKENS
    for pair in $budget; do bargs+=(--budget "$pair"); done
    if [ ! -f "$out" ]; then
        echo "[curriculum] building corpus $name"
        $PY scripts/build_sft_qat_corpus.py --sft "$sft" --split train \
            --window "$WINDOW" --max-tool-tokens "$MAX_TOOL_TOKENS" \
            --min-density "$MIN_DENSITY" "${bargs[@]}" --out "$out"
    fi
    if [ ! -f "$val" ]; then
        # Budget the val corpus too. The trainer reads only --val-windows (4) of it, so an
        # unbudgeted test split builds a tensor LARGER than the training corpus and loads
        # it all into memory for nothing: ultrachat's test split packed to 754 windows
        # against its own 610 training windows.
        local vargs=()
        for pair in $budget; do vargs+=(--budget "${pair%%=*}=2000000"); done
        $PY scripts/build_sft_qat_corpus.py --sft "$sft" --split test \
            --window "$WINDOW" --max-tool-tokens "$MAX_TOOL_TOKENS" \
            --min-density "$MIN_DENSITY" "${vargs[@]}" --out "$val"
    fi
    echo "$out|$val"
}

run_round() {      # n name sft budget
    local n="$1" name="$2" sft="$3" budget="$4"
    local tag="${PREFIX}-r${n}-${name}"
    local outdir="out/exp-058/trained_${tag}"

    if [ -f "${outdir}/trained_latents.pt" ] && [ -f "${outdir}/.round_complete" ]; then
        echo "[curriculum] round $n ($name) already complete — skipping"
        RESUME_FROM="${outdir}/trained_latents.pt"
        return 0
    fi

    local pair; pair=$(build_corpus "$name" "$sft" "$budget")
    local corpus="${pair%%|*}" valc="${pair##*|}"

    mkdir -p "$outdir"
    local resume_args=()
    if [ -n "$RESUME_FROM" ]; then
        [ -f "$RESUME_FROM" ] || { echo "RESUME_FROM=$RESUME_FROM missing"; exit 1; }
        resume_args=(--resume "$RESUME_FROM")
        echo "[curriculum] round $n continues from $RESUME_FROM"
    else
        echo "[curriculum] round $n starts from the shipped vanilla weights"
    fi

    # A leaked trainer holding the card makes the next round's OOM look like this round's
    # configuration. Same guard as run_sft32k_qat_cuda.sh.
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${busy:-0}" -gt 2048 ]; then
        echo "[curriculum] GPU busy (${busy} MiB) — refusing to start round $n"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv
        exit 1
    fi

    echo "[curriculum] round $n ($name) -> $outdir"
    $PY -m quant_tuner.qat.train \
        --corpus "$corpus" --val-corpus "$valc" \
        --train-layers 36 --optim adafactor --dtype fp32 \
        --compute-dtype "$COMPUTE_DTYPE" --matmul-precision "$PRECISION" \
        --grad-accum "$ACCUM" --epochs "$EPOCHS" --lr "$LR" --warmup-frac 0.05 \
        --stop-weight "$STOP_WEIGHT" --grad-spike-factor 0 \
        --val-every "$VAL_EVERY" --val-windows 4 \
        --ckpt-every "$CKPT_EVERY" --ckpt-keep 3 \
        "${resume_args[@]}" \
        --out "$outdir" > "${outdir}/train.log" 2>&1

    touch "${outdir}/.round_complete"
    RESUME_FROM="${outdir}/trained_latents.pt"

    # Export + report each round, so a bad round is visible before the next one builds on it.
    LLAMA_CPP_DIR=vendor/llama.cpp-prism $PY scripts/exp057_qat_export.py \
        --latents "${outdir}/trained_latents.pt" --tag "$tag" \
        > "out/exp-058/export_${tag}.log" 2>&1 || echo "[curriculum] export failed for $tag"
    bash scripts/qat_report_refresh.sh "$outdir" "${tag} — ternary QAT" || true
}

echo "[curriculum] prefix=$PREFIX lr=$LR epochs/round=$EPOCHS window=$WINDOW precision=$PRECISION"
run_round 1 ultrachat "$SFT_R1" "$BUDGET_R1"
run_round 2 distill   "$SFT_R2" "$BUDGET_R2"
run_round 3 ourssft   "$SFT_R3" "$BUDGET_R3"
echo "[curriculum] all rounds complete; final latents: $RESUME_FROM"

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
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-8192}"   # scale with the window (3072 at 8064)
# out/exp-058/fixed holds the corpora rebuilt AFTER the split-assistant-turn fix
# (merge_consecutive_assistant + drop_empty_assistant). The pre-fix corpora in
# out/exp-058 taught "short prose preamble -> <|im_end|>" (18.5% of "Let me..." turns
# ended at their first sentence, vs 0.0% in ultrachat and distillation) and carried 2,155
# assistant turns whose only supervised token was the stop token. Training on them is
# what produced P(stop | sentence end) = 0.95. Do not point this back at the old
# directory without re-reading docs/ternary_qat_curriculum.md.
CORPUS_DIR="${CORPUS_DIR:-out/exp-058/fixed}"
MIN_DENSITY="${MIN_DENSITY:-0.05}"
VAL_EVERY="${VAL_EVERY:-25}"
CKPT_EVERY="${CKPT_EVERY:-50}"
# DISK IS THE BINDING CONSTRAINT, not GPU memory. One checkpoint of this model is 27.8 GB
# and the trainer writes the new one BEFORE pruning the oldest, so a round needs
# (CKPT_KEEP + 1) x 27.8 GB of headroom. At the old --ckpt-keep 3 the three rounds would
# want ~252 GB between them; the sft32k_sw1 run reached 95% disk at step 350 and its next
# save would have failed outright. Two is enough to survive a crash without a third copy
# sitting idle, and prune_round() below drops a finished round to just its final latents.
CKPT_KEEP="${CKPT_KEEP:-2}"
# Refuse to start a round without room for a save plus a margin. Dying at step 400 of 613
# wastes 7 h; refusing at step 0 wastes nothing.
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"

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

# ROUND 3 NO LONGER REUSES THE ABLATION'S CORPUS. It used to, by symlink, so that
# curriculum-r3 vs sft32k_sw1 differed only in training history. That comparison is now
# void by design: sw1's corpus carries the split-assistant-turn defect and reproducing it
# to preserve a clean A/B would mean deliberately re-teaching "prose preamble -> stop".
# Round 3 uses the REBUILT corpus (${CORPUS_DIR}, fingerprint 7f947cbd2adc1544 against
# sw1's 5a2d5d65f640fb74) and MAX_TOOL_TOKENS matches sw1's 8192 so that stays constant.
# So a curriculum-r3 vs sw1 difference now has two causes — curriculum AND corpus fix —
# and the per-round probes are what separate them.

RESUME_FROM="${RESUME_FROM:-}"

build_corpus() {   # name sft_path budget-pairs
    local name="$1" sft="$2" budget="$3"
    local out="${CORPUS_DIR}/corpus_${name}_${WINDOW}.pt"
    local val="${CORPUS_DIR}/corpus_${name}_val_${WINDOW}.pt"
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

prune_round() {    # outdir — drop a FINISHED round to just its final latents
    # Numbered checkpoints exist to resume an interrupted round. Once the round has
    # finished and exported, the only file the next round needs is trained_latents.pt,
    # and each leftover is 27.8 GB. Hardlink-aware: trained_latents.pt usually shares an
    # inode with the last numbered checkpoint, so deleting that one frees nothing and
    # loses nothing — but deleting the OTHERS is where the space comes from.
    local outdir="$1" keep n
    [ -f "${outdir}/trained_latents.pt" ] || return 0
    keep=$(stat -c %i "${outdir}/trained_latents.pt")
    n=0
    for f in "${outdir}"/trained_latents.step*.pt; do
        [ -e "$f" ] || continue
        [ "$(stat -c %i "$f")" = "$keep" ] && continue   # same inode as the final
        rm -f "$f" && n=$((n + 1))
    done
    [ "$n" -gt 0 ] && echo "[curriculum] pruned $n intermediate checkpoint(s) from $outdir"
    return 0
}

require_disk() {   # gib label
    local want="$1" label="$2" free
    free=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
    if [ "${free:-0}" -lt "$want" ]; then
        echo "[curriculum] only ${free} GiB free, need ${want} GiB for $label."
        echo "[curriculum] a checkpoint is 27.8 GB and the trainer writes before pruning."
        df -h /workspace | tail -1
        return 1
    fi
    echo "[curriculum] disk ok: ${free} GiB free (need ${want})"
    return 0
}

run_round() {      # n name sft budget
    local n="$1" name="$2" sft="$3" budget="$4"
    local tag="${PREFIX}-r${n}-${name}"
    local outdir="out/exp-058/trained_${tag}"

    if [ -f "${outdir}/trained_latents.pt" ] && [ -f "${outdir}/.round_complete" ]; then
        echo "[curriculum] round $n ($name) already complete — skipping"
        prune_round "$outdir"
        RESUME_FROM="${outdir}/trained_latents.pt"
        return 0
    fi

    require_disk "$MIN_FREE_GIB" "round $n ($name)" || exit 1

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
        --ckpt-every "$CKPT_EVERY" --ckpt-keep "$CKPT_KEEP" \
        "${resume_args[@]}" \
        --out "$outdir" > "${outdir}/train.log" 2>&1

    touch "${outdir}/.round_complete"
    RESUME_FROM="${outdir}/trained_latents.pt"

    # Export + report each round, so a bad round is visible before the next one builds on it.
    LLAMA_CPP_DIR=vendor/llama.cpp-prism $PY scripts/exp057_qat_export.py \
        --latents "${outdir}/trained_latents.pt" --tag "$tag" \
        > "out/exp-058/export_${tag}.log" 2>&1 || echo "[curriculum] export failed for $tag"
    # PROBE EVERY ROUND, not just the last one. The sft32k_sw1 ablation showed the stop
    # WEIGHT is not what breaks termination — 6.0 and 1.0 both land at P(stop|sentence
    # end) ~0.95 against vanilla's 0.009 — so the cause is in the data. Each round here
    # trains on a DIFFERENT corpus, which makes the curriculum a natural experiment on
    # exactly that question: ultrachat (no tools, no reasoning), distillation (agentic),
    # ours (CLI logs + trajectories). Probing only the final model would collapse three
    # independent observations into one and tell us nothing about which corpus is
    # responsible. It costs ~2 min on CPU and needs no GPU.
    local q2="out/exp-057/Ternary-Bonsai-8B-${tag}-Q2_0.gguf"
    # Skip if already recorded: the probe APPENDS, so a re-run of this round (or of the
    # chain) would leave two identical copies of every probe under one label.
    if [ -f "$q2" ] && ! grep -q "^${tag}," out/exp-058/eval/stop_prob.csv 2>/dev/null; then
        LLAMA_CPP_DIR=vendor/llama.cpp-prism $PY scripts/probe_stop_prob.py \
            --model "$q2" --label "$tag" --out out/exp-058/eval/stop_prob.csv \
            --json-out "out/exp-058/eval/stop_prob_${tag}.json" --ngl 0 \
            2>&1 | tail -8 || echo "[curriculum] probe failed for $tag (not fatal)"
    fi
    bash scripts/qat_report_refresh.sh "$outdir" "${tag} — ternary QAT" || true
    $PY scripts/qat_registry.py >/dev/null 2>&1 || true

    # Only AFTER the export succeeded: the intermediate checkpoints are the fallback if
    # the export needs re-running, so they are worth their 27.8 GB until it has.
    prune_round "$outdir"
    # Each export leaves ~50 GB of HF-checkpoint + F16 intermediates behind to produce a
    # 2.1 GB Q2_0. Across three rounds that is ~150 GB the disk does not have, and the
    # failure would land mid-chain rather than up front.
    bash scripts/prune_export_intermediates.sh "$tag" || true
    df -h /workspace | tail -1
}

echo "[curriculum] prefix=$PREFIX lr=$LR epochs/round=$EPOCHS window=$WINDOW precision=$PRECISION"
run_round 1 ultrachat "$SFT_R1" "$BUDGET_R1"
run_round 2 distill   "$SFT_R2" "$BUDGET_R2"
run_round 3 ourssft   "$SFT_R3" "$BUDGET_R3"
echo "[curriculum] all rounds complete; final latents: $RESUME_FROM"

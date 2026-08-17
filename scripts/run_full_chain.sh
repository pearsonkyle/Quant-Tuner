#!/usr/bin/env bash
# The whole unattended sequence: finish the ablation, benchmark it, then run the
# three-round curriculum and benchmark that.
#
#     bash scripts/run_full_chain.sh                  # from the top
#     FROM=curriculum bash scripts/run_full_chain.sh  # skip to a stage
#
# Stages, in dependency order:
#   1  postflight     export sw1 -> Q2_0, P(im_end) probe, tool-call eval, report
#   2  swe-sw1        agent benchmark on dask__dask-11393 + anomaly analysis
#   3  stop-weight    choose the curriculum's --stop-weight FROM stage 1+2's numbers
#   4  curriculum     round 1 ultrachat -> round 2 distillation -> round 3 our SFT
#   5  swe-final      agent benchmark on the curriculum's final export
#   6  registry       refresh the run ledger
#
# Every stage is idempotent and the registry is refreshed between them, so an interrupted
# chain can be restarted with FROM= and nothing is recomputed.
#
# This runs for ~35 h. It is deliberately fail-soft on EVALUATION stages (a dead
# tool-call eval must not cost a 33 h training run) and fail-hard on TRAINING stages
# (round 2 must never start from a round 1 that did not finish, because --resume would
# silently chain off a partial checkpoint).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
FROM="${FROM:-postflight}"
SW1_RUN="${SW1_RUN:-out/exp-058/trained_sft32k_sw1}"
SW1_TAG="${SW1_TAG:-sft32k_sw1}"
PREFIX="${PREFIX:-curriculum}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-1.0}"
LOG_DIR=out/exp-058/chain
mkdir -p "$LOG_DIR" out/exp-058/eval

stage_num() { case "$1" in
    postflight) echo 1;; swe-sw1) echo 2;; stop-weight) echo 3;;
    curriculum) echo 4;; swe-final) echo 5;; registry) echo 6;; *) echo 0;; esac; }
START=$(stage_num "$FROM")
[ "$START" = "0" ] && { echo "unknown FROM=$FROM"; exit 2; }

say() { echo; echo "=============== [chain] $* ($(date -u +%H:%M:%SZ)) ==============="; }
registry() { $PY scripts/qat_registry.py >/dev/null 2>&1 || true; }

# ---------------------------------------------------------------- 1. postflight
if [ "$START" -le 1 ]; then
    say "1/6 postflight for $SW1_TAG"
    bash scripts/run_sw1_postflight.sh "$SW1_RUN" "$SW1_TAG" 2>&1 \
        | tee "$LOG_DIR/1-postflight.log" | tail -30
    registry
fi

# ---------------------------------------------------------------- 2. swe on sw1
if [ "$START" -le 2 ]; then
    say "2/6 agent benchmark: $SW1_TAG"
    bash scripts/run_swe_mimic.sh "$SW1_TAG" 2>&1 \
        | tee "$LOG_DIR/2-swe-$SW1_TAG.log" | tail -25
    registry
fi

# ---------------------------------------------------------------- 3. stop weight
if [ "$START" -le 3 ]; then
    say "3/6 choosing --stop-weight from measured behaviour"
    SW=$($PY scripts/choose_stop_weight.py --tag "$SW1_TAG" \
            --json-out "out/exp-058/eval/stop_weight_choice.json" \
            2> >(tee "$LOG_DIR/3-stop-weight.log" >&2)) || SW=1.0
    echo "[chain] curriculum will use --stop-weight $SW"
    echo "$SW" > "$LOG_DIR/stop_weight"
fi
SW=$(cat "$LOG_DIR/stop_weight" 2>/dev/null || echo 1.0)

# ---------------------------------------------------------------- 4. curriculum
if [ "$START" -le 4 ]; then
    say "4/6 curriculum (3 rounds, ~33 h) lr=$LR stop-weight=$SW"
    # Fail-hard: run_curriculum_qat.sh is `set -e` and each round --resumes from the
    # previous one's latents, so a partial round must stop the chain rather than become
    # the base of the next round.
    STOP_WEIGHT="$SW" bash scripts/run_curriculum_qat.sh "$PREFIX" "$LR" "$EPOCHS" 2>&1 \
        | tee "$LOG_DIR/4-curriculum.log" | tail -40
    rc=${PIPESTATUS[0]}
    registry
    if [ "$rc" != "0" ]; then
        echo "[chain] curriculum FAILED (rc=$rc) — not benchmarking a partial run."
        echo "[chain] fix, then: FROM=curriculum bash scripts/run_full_chain.sh"
        exit "$rc"
    fi
fi

# ---------------------------------------------------------------- 5. swe on final
if [ "$START" -le 5 ]; then
    FINAL_TAG="${PREFIX}-r3-ourssft"
    say "5/6 agent benchmark: $FINAL_TAG"
    # The probe is cheap and is the other half of the termination picture, so take it on
    # the final model too rather than only on the ablation.
    FINAL_GGUF="out/exp-057/Ternary-Bonsai-8B-${FINAL_TAG}-Q2_0.gguf"
    if [ -f "$FINAL_GGUF" ]; then
        LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY scripts/probe_stop_prob.py \
            --model "$FINAL_GGUF" --label "$FINAL_TAG" \
            --out out/exp-058/eval/stop_prob.csv \
            --json-out "out/exp-058/eval/stop_prob_${FINAL_TAG}.json" --ngl 0 \
            2>&1 | tail -12 || echo "[chain] probe failed — continuing"
        bash scripts/run_swe_mimic.sh "$FINAL_TAG" 2>&1 \
            | tee "$LOG_DIR/5-swe-final.log" | tail -25
    else
        echo "[chain] no final GGUF at $FINAL_GGUF — export must have failed"
    fi
    registry
fi

# ---------------------------------------------------------------- 6. registry
say "6/6 run ledger"
$PY scripts/qat_registry.py --print 2>&1 | tee "$LOG_DIR/6-registry.log" | tail -60

echo
echo "[chain] done. Ledger: docs/qat_run_history.md"

#!/bin/bash
# Universal-SFT QAT run: train Ternary-Bonsai-8B on the FULL universal SFT corpus at an
# ~8k window -> export Q2_0 -> SWE-rebench generalization eval.
#
# What's new vs run_iter5_pipeline.sh (12 verified trajectories at window 4096):
#   * data   — out/corpora/qwen3-universal/sft.jsonl.gz, ALL train-split sources, no caps:
#              CLI logs + agent logs (19 languages) + verified SWE trajectories +
#              red-team refusals + broad-instruct breadth. 19.4M tokens / 2088 windows @ 8064.
#   * window — 8064, not 4096. qat.attention's query-chunked SDPA (bit-identical to the
#              stock kernel, on by default) removes the MPSGraph INT_MAX cap (which was
#              n_heads*S^2 < 2^31 => S <= 8191, i.e. 8192 failed by ONE element). Longer
#              windows then become possible but not faster: measured in the real loop,
#              8064 = 11.5 ms/token, 12288 = 16.3, 16128 never completed a step (swap at
#              99%). 8064 is the only size that runs without swap. docs/ternary_qat.md.
#   * val    — the disjoint `test` split of the same file, scored every --val-every steps.
#
# Usage: run_sft_qat_pipeline.sh [LR] [TAG] [EPOCHS]
#   LR      peak learning rate (default 5e-4 — the measured sweet spot; 3e-4 flips ~0% of
#           codes, i.e. scale drift with a falling loss and no real learning)
#   TAG     artifact tag (default sft-lr<LR>)
#   EPOCHS  fractional epochs over the 2088-window corpus (default 0.35 = 182 steps).
#           This corpus is ~150x the iter-5 one, so ONE epoch is not the unit any more —
#           The budget is wall-clock. MEASURED in the real loop (not a probe): 370 s/step
#           at window 8064 / grad-accum 4 = 11.5 ms/token. 0.35 epochs = 182 steps ~= 19 h.
#           A single-window probe reports ~8 ms/token here — it omits the optimizer step
#           (~106 s/step, fixed at every window size) and runs before swap builds. Size
#           runs off s/step, not off the probe.
#   RESUME (env, default `auto`) — continue from this tag's own trained_latents.pt when
#           one exists (data order, step and Adafactor state; the corpus fingerprint must
#           match). Set RESUME= to force a fresh run, or RESUME=<path> to pin one.
#   CKPT_EVERY (env, default 10) — at ~370 s/step the old 40 meant 4 h of work at risk
#           between checkpoints, and both historical OOM kills landed ON a checkpoint.
#
# Free the GPU first: a full-36-layer fp32 run sits near the unified-memory ceiling, so
# unload any resident LM Studio / llama-server model before starting.
set -e
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism   # Q2_0 (type 41) needs the prism build
PY=.venv/bin/python

LR="${1:-5e-4}"
TAG="${2:-sft-lr${LR}}"
EPOCHS="${3:-0.35}"
CORPUS="${CORPUS:-out/exp-058/sft_corpus_universal_8064.pt}"
VAL="${VAL:-out/exp-058/sft_val_universal_8064.pt}"
TRAIN_OUT="out/exp-058/trained_${TAG}"
GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"
WS="out/swe-rebench/ternary-${TAG}-swe"

[ -f "$CORPUS" ] || { echo "missing $CORPUS — build it with:"; echo "  PYTHONPATH=src $PY scripts/build_sft_qat_corpus.py --sft out/corpora/qwen3-universal/sft.jsonl.gz --window 8064 --max-tool-tokens 3072 --min-density 0.05 --budget logs=none --budget logs-agents=none --budget broad-instruct=none --out $CORPUS"; exit 1; }
VAL_ARGS=()
[ -f "$VAL" ] && VAL_ARGS=(--val-corpus "$VAL" --val-every 40 --val-windows 8)
# RESUME=auto continues from this tag's own checkpoint if one exists; RESUME=<path> pins
# one; RESUME= (empty) starts fresh. The corpus fingerprint must match or the trainer
# refuses -- that guard is what stops a resume onto different data.
RESUME="${RESUME:-auto}"
[ "$RESUME" = "auto" ] && RESUME="$TRAIN_OUT/trained_latents.pt"
RESUME_ARGS=()
if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
  RESUME_ARGS=(--resume "$RESUME")
  echo "=== resuming from $RESUME ($(du -h "$RESUME" | cut -f1)) ==="
fi

echo "=== [$(date)] ${TAG} TRAIN (all-36, adafactor, fp32, window 8064, lr ${LR}, ${EPOCHS} epochs) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus "$CORPUS" "${VAL_ARGS[@]}" "${RESUME_ARGS[@]}" \
  --layers 0-35 --optim adafactor --epochs "$EPOCHS" --grad-accum 4 --lr "$LR" \
  --dtype fp32 --ckpt-every "${CKPT_EVERY:-10}" --flip-sample 12 \
  --out "$TRAIN_OUT"

echo "=== [$(date)] ${TAG} EXPORT -> Q2_0 (prism) ==="
$PY -u scripts/exp057_qat_export.py --latents "$TRAIN_OUT/trained_latents.pt" --tag "$TAG"

echo "=== [$(date)] ${TAG} SWE-rebench (10-instance holdout, openai-agents) ==="
$PY -u scripts/run_swebench_eval.py \
  --models "$GGUF" \
  --holdout out/external/swe-rebench/holdout.jsonl \
  --workspace "$WS" \
  --agent openai-agents --temperature 0.25 --max-steps 100 \
  --resume --cleanup-images --progress

echo "=== [$(date)] ${TAG} DONE. patch/pass: ==="
$PY -c "import json; d=json.load(open('${WS}/summary.json')); \
m=list(d['models'].values())[0]['aggregate']; \
print(f\"patch_rate={m['patch_rate']:.2f} pass_rate={m['pass_rate']:.2f} \
resolved={m['n_resolved']}/{m['n_instances']} tool_err={m['tool_error_rate']:.2f} \
mean_steps={m['mean_steps']:.1f}\")"

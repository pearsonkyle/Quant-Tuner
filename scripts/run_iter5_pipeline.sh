#!/bin/bash
# iter-5 end-to-end: train on the VERIFIED Ornith trajectories -> export Q2_0 -> SWE-rebench.
#
# Trains Ternary-Bonsai-8B (all-36, Adafactor, fp32) on the 12 resolved SWE-rebench
# trajectories re-rendered through the corrected corpus, exports to Q2_0 (prism fork), and
# evals patch/pass on the disjoint 10-instance holdout.
#
# Usage: run_iter5_pipeline.sh [LR] [TAG] [EPOCHS]
#   LR      peak learning rate (default 5e-4). Findings so far:
#             3e-4 -> ~0% code flips (scale-only, no real learning)
#             1e-3 -> real flips but noisy loss + damaged tool-calling on 12 trajectories
#             5e-4 -> the untested middle (sweet-spot test)
#   TAG     artifact tag, keeps runs from colliding (default iter5-lr<LR>)
#   EPOCHS  default 8
# Flip telemetry every 20 steps is the live probe; export prints artifact-level flips vs
# shipped so a scale-only run is caught before we read the eval.
set -e
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism   # Q2_0 (type 41) needs the prism build
PY=.venv/bin/python

LR="${1:-5e-4}"
TAG="${2:-iter5-lr${LR}}"
EPOCHS="${3:-8}"
CORPUS=out/exp-058/distill_corpus_iter5_4096.pt
TRAIN_OUT="out/exp-058/trained_${TAG}"
GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"
WS="out/swe-rebench/ternary-${TAG}-swe"

echo "=== [$(date)] ${TAG} TRAIN (all-36, adafactor, lr ${LR}, ${EPOCHS} epochs) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus "$CORPUS" \
  --layers 0-35 --optim adafactor --epochs "$EPOCHS" --grad-accum 2 --lr "$LR" \
  --dtype fp32 --ckpt-every 20 --flip-sample 12 \
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

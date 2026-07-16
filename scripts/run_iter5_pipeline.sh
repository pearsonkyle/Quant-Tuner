#!/bin/bash
# iter-5 end-to-end: train on the VERIFIED Ornith trajectories -> export Q2_0 -> SWE-rebench.
#
# The data-quality lever. Trains Ternary-Bonsai-8B (all-36, Adafactor, fp32) on the 12
# resolved SWE-rebench trajectories re-rendered through the corrected corpus, exports to
# Q2_0 (prism fork), and evals patch/pass on the disjoint 10-instance holdout.
#
# LR: 3e-4 (the audit's mid candidate; BitNet-family QAT runs 1e-4..1e-3). Flip telemetry
# every 20 steps is the live probe -- if the first report shows ~zero code flips the LR is
# too low (kill + bump); if loss diverges, drop it. Export prints artifact-level flips vs
# shipped, so a scale-only (unchanged-model) run is caught before we read the eval.
set -e
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism   # Q2_0 (type 41) needs the prism build
PY=.venv/bin/python
CORPUS=out/exp-058/distill_corpus_iter5_4096.pt
TRAIN_OUT=out/exp-058/trained_iter5
GGUF=out/exp-057/Ternary-Bonsai-8B-iter5-Q2_0.gguf

echo "=== [$(date)] iter-5 TRAIN (all-36, adafactor, lr 3e-4, ~8 epochs) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus "$CORPUS" \
  --layers 0-35 --optim adafactor --epochs 8 --grad-accum 2 --lr 3e-4 \
  --dtype fp32 --ckpt-every 20 --flip-sample 12 \
  --out "$TRAIN_OUT"

echo "=== [$(date)] iter-5 EXPORT -> Q2_0 (prism) ==="
$PY -u scripts/exp057_qat_export.py --latents "$TRAIN_OUT/trained_latents.pt" --tag iter5

echo "=== [$(date)] iter-5 SWE-rebench (10-instance holdout, openai-agents) ==="
$PY -u scripts/run_swebench_eval.py \
  --models "$GGUF" \
  --holdout out/external/swe-rebench/holdout.jsonl \
  --workspace out/swe-rebench/ternary-iter5-swe \
  --agent openai-agents --temperature 0.25 --max-steps 100 \
  --resume --cleanup-images --progress

echo "=== [$(date)] iter-5 DONE. patch/pass: ==="
$PY -c "import json; d=json.load(open('out/swe-rebench/ternary-iter5-swe/summary.json')); \
m=list(d['models'].values())[0]['aggregate']; \
print(f\"patch_rate={m['patch_rate']:.2f} pass_rate={m['pass_rate']:.2f} \
resolved={m['n_resolved']}/{m['n_instances']} mean_steps={m['mean_steps']:.1f}\")"

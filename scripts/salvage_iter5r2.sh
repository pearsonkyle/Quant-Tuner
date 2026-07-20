#!/bin/bash
# Salvage round 2: its training was OOM-killed at step 180/210, but the step-160 checkpoint
# is complete and fully converged (loss 1.79 -> 0.33; the remaining steps were cosine-decay
# tail). Export that checkpoint and run the dual-bench, so we get the 60-trajectory result
# without re-training.
set -e
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
PY=.venv/bin/python
TAG=iter5-r2
GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"

echo "=== [$(date)] salvage $TAG: export step-160 checkpoint -> Q2_0 ==="
$PY -u scripts/exp057_qat_export.py --latents "out/exp-058/trained_${TAG}/trained_latents.pt" --tag "$TAG"

echo "=== [$(date)] generalization eval (disjoint 10-holdout) ==="
$PY -u scripts/run_swebench_eval.py --models "$GGUF" \
  --holdout out/external/swe-rebench/holdout.jsonl --workspace "out/swe-rebench/ternary-${TAG}-swe" \
  --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images --progress

echo "=== [$(date)] in-distribution eval (60 trained instances) ==="
$PY -u scripts/run_swebench_eval.py --models "$GGUF" \
  --holdout out/external/swe-rebench/indist_iter5-r2.jsonl --workspace "out/swe-rebench/ternary-${TAG}-indist-swe" \
  --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images --progress

echo "=== [$(date)] iter5-r2 (60 traj) DUAL-EVAL ==="
$PY -c "
import json
def row(p,label):
    m=list(json.load(open(p))['models'].values())[0]['aggregate']
    print(f'  {label:16s} patch={m[\"patch_rate\"]:.2f} pass={m[\"pass_rate\"]:.2f} resolved={m[\"n_resolved\"]}/{m[\"n_instances\"]} tool_err={m[\"tool_error_rate\"]:.2f} steps={m[\"mean_steps\"]:.1f}')
row('out/swe-rebench/ternary-${TAG}-swe/summary.json','generalization')
row('out/swe-rebench/ternary-${TAG}-indist-swe/summary.json','in-distribution')
print('  (iter-5 @12: gen 40/0, indist 25/8 ; round1 @30: gen 50/0, indist 43/0)')
"

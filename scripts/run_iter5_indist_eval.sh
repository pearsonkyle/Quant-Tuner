#!/bin/bash
# iter-5 IN-DISTRIBUTION eval: SWE-rebench on the 12 instances the model TRAINED on.
#
# The main pipeline evals on the 10-instance holdout (disjoint = true generalization). This
# is the complementary check the user asked for: does the model solve the issues it saw
# solved during training? Expected to be inflated by memorization -- that's the point: a big
# in-dist gain with a flat/negative generalization number = it learned but overfit (need
# more data); gains on BOTH = real learning.
#
# Waits for the training pipeline to finish first (frees the GPU + the gen-eval server), then
# serves the SAME GGUF against the in-dist holdout in a separate workspace.
#
# Usage: run_iter5_indist_eval.sh <TAG>   (TAG matches the pipeline run, e.g. iter5-5e4-36step)
set -e
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
PY=.venv/bin/python
TAG="${1:-iter5-5e4-36step}"
GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"
WS="out/swe-rebench/ternary-${TAG}-indist-swe"

echo "=== [$(date)] waiting for the training pipeline to finish (frees GPU/server) ==="
while pgrep -f 'run_iter5_pipeline|exp058_qat_train|run_swebench_eval' >/dev/null; do sleep 60; done
sleep 20
if [ ! -f "$GGUF" ]; then echo "[indist] GGUF $GGUF not found — pipeline export failed?"; exit 1; fi
echo "=== [$(date)] in-distribution SWE-rebench: 12 trained-on instances ==="
$PY -u scripts/run_swebench_eval.py \
  --models "$GGUF" \
  --holdout out/external/swe-rebench/indist_trained12.jsonl \
  --workspace "$WS" \
  --agent openai-agents --temperature 0.25 --max-steps 100 \
  --resume --cleanup-images --progress

echo "=== [$(date)] ${TAG} DUAL-EVAL SUMMARY ==="
$PY -c "
import json
def row(p, label):
    m=list(json.load(open(p))['models'].values())[0]['aggregate']
    print(f'  {label:16s} patch={m[\"patch_rate\"]:.2f} pass={m[\"pass_rate\"]:.2f} '
          f'resolved={m[\"n_resolved\"]}/{m[\"n_instances\"]} tool_err={m[\"tool_error_rate\"]:.2f} steps={m[\"mean_steps\"]:.1f}')
row('$WS/summary.json', 'in-distribution')
try:
    row('out/swe-rebench/ternary-${TAG}-swe/summary.json', 'generalization')
except Exception as e:
    print('  generalization: (pending/not found)', e)
"

#!/bin/bash
# SWE-rebench on prism-ml Ternary-Bonsai-8B Q2_0 (native 1.58-bit, Qwen3-8B base).
# Runs through the PrismML llama.cpp fork (Q2_0 kernels) via LLAMA_CPP_DIR.
# 1 rep x 10 instances, same holdout/sampling as the Ornith/Qwythos ladder.
# No --chat-template-kwargs: this model emits no <think>, so we skip the reasoning
# translation path entirely. ctx capped at the model's 65536 max.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
PY=.venv/bin/python
WS=out/swe-rebench/ternary-q2_0-swe

done_count() { find "$WS/trajectories" -name "*.result.json" 2>/dev/null | wc -l | tr -d ' '; }

if [ "$(done_count)" -ge 10 ]; then
  echo "=== [$(date)] Ternary Q2_0 already has 10 graded — skip ==="
  exit 0
fi

echo "=== [$(date)] Ternary-Bonsai-8B Q2_0 SWE-rebench start (reps=1, 10 instances) ==="
$PY -u scripts/run_swebench_eval.py \
  --models out/exp-056/gguf/Ternary-Bonsai-8B-Q2_0.gguf \
  --agent openai-agents --reps 1 \
  --temperature 0.25 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 --ctx 65536 \
  --holdout out/external/swe-rebench/holdout.jsonl \
  --workspace "$WS"
echo "=== [$(date)] Ternary Q2_0 SWE-rebench DONE ==="

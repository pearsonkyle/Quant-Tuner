#!/bin/bash
# #3 prompt-scaffolding A/B: same Ternary Q2_0, same holdout/sampling, but with the
# opt-in QT_SWE_EXTRA_INSTRUCTIONS=scaffold guidance (no hidden-test-name leakage).
# Compared against out/swe-rebench/ternary-q2_0-swe (baseline: 0% pass / 50% patch /
# 79% tool-err). Different workspace so both sets of trajectories survive.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
export QT_SWE_EXTRA_INSTRUCTIONS=scaffold
PY=.venv/bin/python
WS=out/swe-rebench/ternary-q2_0-scaffold-swe

done_count() { find "$WS/trajectories" -name "*.result.json" 2>/dev/null | wc -l | tr -d ' '; }
if [ "$(done_count)" -ge 10 ]; then
  echo "=== [$(date)] Ternary Q2_0 scaffold already has 10 — skip ==="; exit 0
fi

echo "=== [$(date)] Ternary-Bonsai-8B Q2_0 + SCAFFOLD SWE-rebench start (reps=1, 10) ==="
$PY -u scripts/run_swebench_eval.py \
  --models out/exp-056/gguf/Ternary-Bonsai-8B-Q2_0.gguf \
  --agent openai-agents --reps 1 \
  --temperature 0.25 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 --ctx 65536 \
  --holdout out/external/swe-rebench/holdout.jsonl \
  --workspace "$WS"
echo "=== [$(date)] Ternary Q2_0 scaffold DONE ==="

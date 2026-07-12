#!/bin/bash
# SWE-rebench AWQ pass/patch: 1 rep x 10 instances per model (variance reported
# as the binomial spread over the 10 instances). Skips a model whose workspace
# already has 10 graded result.json files, so it's restart-safe.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python
COMMON="--agent openai-agents --reps 1 --temperature 0.25 --top-p 0.95 --top-k 20 --max-tokens 32768 --ctx 131072 --holdout out/external/swe-rebench/holdout.jsonl"

done_count() { find "$1/trajectories" -name "*.result.json" 2>/dev/null | wc -l | tr -d ' '; }

if [ "$(done_count out/swe-rebench/gemma-awq-swe)" -lt 10 ]; then
  echo "=== [$(date)] gemma AWQ (reps=1) start ===" >> out/swe-rebench/overnight.log
  $PY -u scripts/run_swebench_eval.py --models out/exp-052/gemma-4-31B-it-IQ2_M-awq.gguf \
    $COMMON --chat-template-kwargs '{"enable_thinking":false}' \
    --workspace out/swe-rebench/gemma-awq-swe > out/swe-rebench/gemma-awq.run.log 2>&1
else
  echo "=== [$(date)] gemma AWQ already has 10 — skip ===" >> out/swe-rebench/overnight.log
fi

if [ "$(done_count out/swe-rebench/ornith-awq-swe)" -lt 10 ]; then
  echo "=== [$(date)] Ornith AWQ (reps=1) start ===" >> out/swe-rebench/overnight.log
  $PY -u scripts/run_swebench_eval.py --models uploads/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF/Ornith-1.0-9B-IQ2_M.gguf \
    $COMMON --chat-template-kwargs '{"enable_thinking":false}' \
    --workspace out/swe-rebench/ornith-awq-swe > out/swe-rebench/ornith-awq.run.log 2>&1
else
  echo "=== [$(date)] Ornith AWQ already has 10 — skip ===" >> out/swe-rebench/overnight.log
fi
echo "=== [$(date)] ALL DONE ===" >> out/swe-rebench/overnight.log

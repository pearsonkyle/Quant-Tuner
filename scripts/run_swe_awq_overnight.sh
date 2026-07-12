#!/bin/bash
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python
COMMON="--agent openai-agents --reps 3 --temperature 0.25 --top-p 0.95 --top-k 20 --max-tokens 32768 --ctx 131072 --holdout out/external/swe-rebench/holdout.jsonl"
echo "=== [$(date)] gemma AWQ start ===" >> out/swe-rebench/overnight.log
$PY -u scripts/run_swebench_eval.py --models out/exp-052/gemma-4-31B-it-IQ2_M-awq.gguf \
  $COMMON --workspace out/swe-rebench/gemma-awq-swe > out/swe-rebench/gemma-awq.run.log 2>&1
echo "=== [$(date)] gemma AWQ done; ornith AWQ start ===" >> out/swe-rebench/overnight.log
$PY -u scripts/run_swebench_eval.py --models uploads/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF/Ornith-1.0-9B-IQ2_M.gguf \
  $COMMON --chat-template-kwargs '{"enable_thinking":false}' \
  --workspace out/swe-rebench/ornith-awq-swe > out/swe-rebench/ornith-awq.run.log 2>&1
echo "=== [$(date)] ALL DONE ===" >> out/swe-rebench/overnight.log

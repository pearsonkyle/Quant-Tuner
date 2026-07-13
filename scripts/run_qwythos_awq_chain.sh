#!/bin/bash
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p out/exp-055
echo "=== [$(date)] exp-055 AWQ build start ===" >> out/swe-rebench/qwythos.log
$PY -u scripts/exp055_qwythos_awq.py > out/exp-055/run.log 2>&1
Q=out/exp-055/Qwythos-9B-v2-IQ2_M-awq.gguf
# race guard: build finished AND file has valid GGUF magic
if ! grep -q "exp-055 complete" out/exp-055/run.log || [ "$(head -c4 "$Q" 2>/dev/null)" != "GGUF" ]; then
  echo "=== [$(date)] AWQ build failed/incomplete — see out/exp-055/run.log ===" >> out/swe-rebench/qwythos.log
  exit 1
fi
echo "=== [$(date)] Qwythos AWQ IQ2_M SWE-rebench start ===" >> out/swe-rebench/qwythos.log
$PY -u scripts/run_swebench_eval.py --models "$Q" \
  --agent openai-agents --reps 1 --temperature 0.25 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 --ctx 131072 --chat-template-kwargs '{"enable_thinking":false}' \
  --holdout out/external/swe-rebench/holdout.jsonl \
  --workspace out/swe-rebench/qwythos-awq-iq2m-swe > out/swe-rebench/qwythos-awq-iq2m.run.log 2>&1
echo "=== [$(date)] Qwythos AWQ IQ2_M SWE-rebench DONE ===" >> out/swe-rebench/qwythos.log

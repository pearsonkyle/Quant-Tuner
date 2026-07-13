#!/bin/bash
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src; PY=.venv/bin/python
run_one() {
  local q="$1" ws="$2" name="$3"
  [ "$(head -c4 "$q" 2>/dev/null)" = "GGUF" ] || { echo "=== [$(date)] SKIP $name (bad GGUF) ===" >> out/swe-rebench/ornith-ladder.log; return; }
  echo "=== [$(date)] $name start ===" >> out/swe-rebench/ornith-ladder.log
  $PY -u scripts/run_swebench_eval.py --models "$q" \
    --agent openai-agents --reps 1 --temperature 0.25 --top-p 0.95 --top-k 20 \
    --max-tokens 32768 --ctx 131072 --chat-template-kwargs '{"enable_thinking":false}' \
    --holdout out/external/swe-rebench/holdout.jsonl \
    --workspace "$ws" > "${ws}.run.log" 2>&1
  echo "=== [$(date)] $name DONE ===" >> out/swe-rebench/ornith-ladder.log
}
run_one out/exp-051/new/Ornith-1.0-9B-IQ2_M-new.gguf out/swe-rebench/ornith-iq2m-im-swe "IQ2_M imatrix"
run_one out/exp-051/new/Ornith-1.0-9B-IQ4_XS-new.gguf out/swe-rebench/ornith-iq4xs-swe "IQ4_XS"
run_one uploads/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF/Ornith-1.0-9B-Q5_K_M.gguf out/swe-rebench/ornith-q5km-swe "Q5_K_M"
echo "=== [$(date)] ORNITH LADDER DONE ===" >> out/swe-rebench/ornith-ladder.log

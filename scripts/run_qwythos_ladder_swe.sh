#!/bin/bash
# After the AWQ IQ2_M SWE-rebench finishes, run SWE-rebench (1 rep x 10) on the
# higher-bit Qwythos quants so we can build the unified pass/patch table.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python
for i in $(seq 1 360); do   # wait up to ~3h for the AWQ test to finish
  grep -qE "Qwythos AWQ IQ2_M SWE-rebench DONE|AWQ build failed" out/swe-rebench/qwythos.log 2>/dev/null && break
  sleep 30
done
run_one() {
  local q="$1" ws="$2" name="$3"
  if [ "$(head -c4 "$q" 2>/dev/null)" != "GGUF" ]; then
    echo "=== [$(date)] SKIP $name (missing/bad GGUF) ===" >> out/swe-rebench/qwythos.log; return; fi
  echo "=== [$(date)] $name SWE-rebench start ===" >> out/swe-rebench/qwythos.log
  $PY -u scripts/run_swebench_eval.py --models "$q" \
    --agent openai-agents --reps 1 --temperature 0.25 --top-p 0.95 --top-k 20 \
    --max-tokens 32768 --ctx 131072 --chat-template-kwargs '{"enable_thinking":false}' \
    --holdout out/external/swe-rebench/holdout.jsonl \
    --workspace "$ws" > "${ws}.run.log" 2>&1
  echo "=== [$(date)] $name DONE ===" >> out/swe-rebench/qwythos.log
}
run_one out/exp-054/iq4_xs/Qwythos-9B-v2-IQ4_XS.gguf out/swe-rebench/qwythos-iq4xs-swe "IQ4_XS"
run_one out/exp-054/q5_k_m/Qwythos-9B-v2-Q5_K_M.gguf out/swe-rebench/qwythos-q5km-swe "Q5_K_M"
echo "=== [$(date)] LADDER DONE ===" >> out/swe-rebench/qwythos.log

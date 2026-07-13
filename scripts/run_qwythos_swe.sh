#!/bin/bash
# Wait for the exp-054 build to finish, then SWE-rebench the Qwythos IQ2_M
# (1 rep x 10 instances) with the same config as the Ornith/gemma loop tests:
# reasoning off + server-side repeat-penalty 1.1 (from eval/server.py).
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python
Q=out/exp-054/iq2_m/Qwythos-9B-v2-IQ2_M.gguf
for i in $(seq 1 240); do   # up to ~2h wait for the build
  grep -q "=== exp-054 complete ===" out/exp-054/run.log 2>/dev/null && break
  [ -f "$Q" ] && break
  pgrep -f exp054_qwythos >/dev/null || { [ -f "$Q" ] || { echo "build died before IQ2_M" >> out/swe-rebench/qwythos.log; exit 1; }; }
  sleep 30
done
echo "=== [$(date)] Qwythos IQ2_M SWE-rebench (reps=1) start ===" >> out/swe-rebench/qwythos.log
$PY -u scripts/run_swebench_eval.py --models "$Q" \
  --agent openai-agents --reps 1 --temperature 0.25 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 --ctx 131072 --chat-template-kwargs '{"enable_thinking":false}' \
  --holdout out/external/swe-rebench/holdout.jsonl \
  --workspace out/swe-rebench/qwythos-iq2m-swe > out/swe-rebench/qwythos-iq2m.run.log 2>&1
echo "=== [$(date)] Qwythos IQ2_M SWE-rebench DONE ===" >> out/swe-rebench/qwythos.log

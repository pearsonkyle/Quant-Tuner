#!/bin/bash
# iter-5 data generation: harvest VERIFIED SWE-rebench solutions from a strong solver.
#
# Drives Ornith-1.0-9B (Q5_K_M: 100% patch / 60% resolved on the eval holdout) over the
# 95-instance DISJOINT training pool (out/external/swe-rebench/distill_train.jsonl, built
# with --exclude the eval holdout so the student is never graded on what it trained on).
# Saves per-instance trajectories + results.csv; build_ornith_distill_corpus.py then keeps
# the RESOLVED ones and re-renders them through Bonsai's tokenizer for masked-CE training.
#
# ~5min + one Docker container per instance (linux/amd64 under emulation on Apple Silicon,
# so first-time image pulls are slow) -> ballpark ~10-14h for 95 instances. --resume skips
# any instance already graded, so a crash/restart never re-does completed work.
# Same agent + sampling as the published Ornith run (openai-agents, T=0.25).
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python
MODEL=uploads/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF/Ornith-1.0-9B-Q5_K_M.gguf

echo "=== [$(date)] Ornith distill-gen START (95 disjoint instances, openai-agents) ==="
$PY -u scripts/run_swebench_eval.py \
  --models "$MODEL" \
  --holdout out/external/swe-rebench/distill_train.jsonl \
  --workspace out/swe-rebench/ornith-distill-gen \
  --agent openai-agents \
  --temperature 0.25 \
  --max-steps 100 \
  --resume --progress
echo "=== [$(date)] Ornith distill-gen DONE ==="

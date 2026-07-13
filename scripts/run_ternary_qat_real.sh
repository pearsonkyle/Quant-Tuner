#!/bin/bash
# exp-057 REAL run: continued ternary QAT of Ternary-Bonsai-8B on the combined
# tool-log(~480K) + wiki(~300K) corpus. Full 36 layers, fp32 latents + AdamW,
# 600 steps, seq 1024, grad-accum 2 (~6-7h on M4 Max, ~83 GiB). Restart-safe
# only in that a completed run writes trained_real/trained_latents.pt.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python

echo "=== [$(date)] Ternary QAT real run START (full-36, 600 steps) ==="
$PY -u scripts/exp057_qat_train.py \
  --train-layers 36 --steps 600 --seq 1024 --grad-accum 2 --lr 2e-5 \
  --corpus out/exp-057/corpus.train.txt --limit-tokens 0 \
  --out out/exp-057/trained_real
echo "=== [$(date)] Ternary QAT real run DONE ==="

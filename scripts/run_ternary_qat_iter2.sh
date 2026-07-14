#!/bin/bash
# iter-2 REAL run: masked-loss, schema-rendered, 4096-window ternary QAT.
# Config from the audit + MPS constraints:
#   window 4096 (MPS max; 8192 = MPSGraph INT_MAX error), fp32 last-18, foreach=False,
#   masked corpus (schemas render, loss on assistant tool_calls only, no all-masked
#   windows), LR warmup->cosine to 5e-5, grad-accum 4.
# Throughput is token-bound (~8ms/tok); ~0.5 epoch of the 2090-window corpus ≈ 9h.
# Checkpoints every 20 steps + saves on SIGTERM, so it's stoppable with progress.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python

echo "=== [$(date)] iter-2 QAT START (4096 masked, fp32 last-18, ~0.5 epoch) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus out/exp-058/masked_corpus_4096.pt \
  --train-layers 18 --epochs 0.5 --grad-accum 4 --lr 5e-5 \
  --dtype fp32 --ckpt-every 20 \
  --out out/exp-058/trained_iter2
echo "=== [$(date)] iter-2 QAT DONE ==="

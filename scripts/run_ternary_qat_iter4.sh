#!/bin/bash
# iter-4: the SCALE test -- train ALL 36 layers (vs iter-3's influential 18).
# Fits via Adafactor (factored optimizer state -> ~33GiB, no swap; AdamW's two fp32
# states would swap). Same masked 4096 corpus, fp32 latents, foreach-free. 0.5 epoch
# (~16h at ~56s/microbatch full-36). Tests the "under-trained / partial network"
# hypothesis: does training the whole network raise patch/pass past baseline?
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python

echo "=== [$(date)] iter-4 QAT START (ALL 36 layers, adafactor, ~0.5 epoch) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus out/exp-058/masked_corpus_4096.pt \
  --layers 0-35 --optim adafactor --epochs 0.5 --grad-accum 4 --lr 5e-5 \
  --dtype fp32 --ckpt-every 20 \
  --out out/exp-058/trained_iter4
echo "=== [$(date)] iter-4 QAT DONE ==="

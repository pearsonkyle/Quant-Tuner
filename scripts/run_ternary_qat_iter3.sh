#!/bin/bash
# iter-3: retrain the GRAD-INFLUENTIAL layers instead of the naive last-18.
# The layer-importance probe (grad signal on the masked tool-call loss) found the
# early layers 0-14 + final 32/34/35 carry the signal, while 18-31 (what iter-2
# trained) are the weakest. Same masked 4096 corpus, same budget (~0.5 epoch) as
# iter-2 for a fair A/B — only the trained layer SET changes.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python

echo "=== [$(date)] iter-3 QAT START (layers 0-14,32,34,35; masked 4096; ~0.5 epoch) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus out/exp-058/masked_corpus_4096.pt \
  --layers 0-14,32,34,35 --epochs 0.5 --grad-accum 4 --lr 5e-5 \
  --dtype fp32 --ckpt-every 20 \
  --out out/exp-058/trained_iter3
echo "=== [$(date)] iter-3 QAT DONE ==="

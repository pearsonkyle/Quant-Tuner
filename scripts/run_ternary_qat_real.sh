#!/bin/bash
# exp-057 REAL run: continued ternary QAT of Ternary-Bonsai-8B on the combined
# tool-log(~480K) + wiki(~300K) corpus.
#
# Config chosen after debugging the MPS hang (experiments A-F):
#   * fp32 latents  — bf16 updates underflow the ternary threshold (no code flips)
#                     at the only lr where bf16 full-36 is stable; fp32 registers
#                     updates at any lr and doesn't destabilize.
#   * last-18 layers — fp32 full-36 AdamW swaps (Adam's 2 fp32 states ~56 GB over
#                     budget); last-18 fits ~57-74 GB with no swap.
#   * foreach=False — MPS multi-tensor (foreach) kernels DEADLOCK at full-model
#                     scale; the confirmed step-5 hang. Non-foreach is also faster.
# ~5-6h on M4 Max. Launched via setsid so a parent shell signal can't kill it.
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python

echo "=== [$(date)] Ternary QAT real run START (fp32 last-18, foreach=False, 600 steps) ==="
$PY -u scripts/exp057_qat_train.py \
  --train-layers 18 --steps 600 --seq 1024 --grad-accum 2 --lr 2e-5 \
  --dtype fp32 --no-foreach \
  --corpus out/exp-057/corpus.train.txt --limit-tokens 0 \
  --out out/exp-057/trained_real
echo "=== [$(date)] Ternary QAT real run DONE ==="

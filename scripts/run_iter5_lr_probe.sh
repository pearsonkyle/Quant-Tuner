#!/bin/bash
# iter-5 LR probe — the audit's mandated first step before the full run.
#
# At lr 5e-5 the flip-distance math predicts ~zero code flips (iter-2/3's loss drop was
# likely pure scale drift). This runs three SHORT all-36 adafactor passes at 5e-5 / 3e-4 /
# 1e-3 on the verified-trajectory corpus and logs the code-flip telemetry each checkpoint.
# Read the "flips X%" lines afterwards: pick the smallest LR whose flips are clearly
# non-zero AND whose loss is stable (not NaN-skipping), then launch the full run.
#
# Waits for the generation run's llama-server to exit first — the trainer needs the ~70 GB
# of unified memory the Ornith server is holding (running both risks OOM).
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
CORPUS=out/exp-058/distill_corpus_iter5_4096.pt

echo "=== [$(date)] waiting for generation run to free the GPU/server ==="
while pgrep -f run_swebench_eval >/dev/null; do sleep 60; done
# give the server a moment to release Metal + unified memory
sleep 20
echo "=== [$(date)] server freed; starting LR probe on $CORPUS ==="

# 33 windows -> grad-accum 2 gives ~16 steps/epoch; 3 epochs ~= 48 steps, ckpt at 20/40
for LR in 5e-5 3e-4 1e-3; do
  echo "=== [$(date)] iter-5 LR PROBE lr=$LR ==="
  $PY -u scripts/exp058_qat_train_v2.py \
    --corpus "$CORPUS" \
    --layers 0-35 --optim adafactor --epochs 3 --grad-accum 2 --lr "$LR" \
    --dtype fp32 --ckpt-every 20 --flip-sample 12 \
    --out "out/exp-058/probe_iter5_lr${LR}" 2>&1 | tee "out/exp-058/iter5_probe_lr${LR}.log"
done

echo "=== [$(date)] LR probe DONE. Flip summary: ==="
grep -H 'flips ' out/exp-058/iter5_probe_lr*.log | tail -40
echo "=== pick the LR, then run the full pass (see run_ternary_qat_iter5.sh phase 3) ==="

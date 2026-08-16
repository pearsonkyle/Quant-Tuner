#!/usr/bin/env bash
# KLD over the six eval distributions, then the MTP draft-n sweep.
#
# Sequential in ONE script rather than two chained waiters: run_hf_kld.py now rewrites
# its CSV after every corpus (so a late failure no longer discards earlier work), which
# means "results CSV exists" no longer implies "KLD finished". Ordering by process exit
# is the only correct signal, and that is what this gives.
#
# `--two-pass` holds one model on the card at a time. transformers DECOMPRESSES the
# pack-quantized weights on load, so the "16 GiB" checkpoint occupies ~51 GB of VRAM —
# the pair does not fit in 97.9 GB, which is why resident mode is not an option here.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
REF=$REPO/out/exp-060/model_extracted
CORPORA=$REPO/out/exp-060-32k/corpora
cd $REPO

# Fragmentation, not capacity, is what kills the model swap: each corpus moves ~51 GB on
# and off the card, and the allocator can end up unable to find a contiguous block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "waiting for a free GPU …"
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 30
done

echo "=== KLD: 6 distributions vs the bf16 reference ==="
PYTHONPATH=src $REPO/.venv/bin/python scripts/run_hf_kld.py \
  --ref "$REF" \
  --quant "$OUT/checkpoint" \
  --corpora-dir "$CORPORA" \
  --out "$OUT/results/kld_results.csv" \
  --ctx 8192 \
  --two-pass \
  --model-class Qwen3_5ForCausalLM \
  2>&1 | tee "$OUT/logs/kld.log" | grep -vE "^  \[|it/s\]"
kld_rc=${PIPESTATUS[0]}
echo "KLD exit code: $kld_rc"
if [ "$kld_rc" -ne 0 ]; then
  echo "KLD FAILED — continuing to MTP anyway; whatever corpora completed are in the CSV."
fi

echo
echo "=== releasing GPU before MTP ==="
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 20
done

exec bash "$REPO/scripts/exp060_w4a16_mtp_validate.sh"

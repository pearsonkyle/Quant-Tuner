#!/usr/bin/env bash
# Run the MTP validation once the KLD sweep has written its results CSV and released
# the GPU. Separate from the KLD script so a KLD failure does not silently skip MTP.
set -uo pipefail
OUT=/workspace/Quant-Tuner/out/exp-060-w4a16-32k
for _ in $(seq 1 720); do
  [ -f "$OUT/results/kld_results.csv" ] && { echo "KLD results present"; break; }
  sleep 60
done
# let the KLD process release VRAM before vLLM claims 90% of the card
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 30
done
exec bash /workspace/Quant-Tuner/scripts/exp060_w4a16_mtp_validate.sh

#!/usr/bin/env bash
# Re-run KLD (all six distributions) once the MTP sweep releases the GPU.
#
# All six, not just the five that failed: write_csv rewrites the whole file, so running a
# subset would drop `external` from the results. It is only 10 chunks and it reproduced
# bit-identically across two runs (0.01410 / 87.82%), so re-measuring it is cheap and
# yields one internally consistent CSV instead of a merged one.
set -uo pipefail
REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REPO

echo "waiting for the MTP sweep to finish …"
for _ in $(seq 1 240); do
  [ -f "$OUT/results/mtp_sweep.json" ] && { echo "MTP sweep done"; break; }
  sleep 30
done
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 20
done

echo "=== KLD re-run (no_grad fix) ==="
PYTHONPATH=src $REPO/.venv/bin/python scripts/run_hf_kld.py \
  --ref "$REPO/out/exp-060/model_extracted" \
  --quant "$OUT/checkpoint" \
  --corpora-dir "$REPO/out/exp-060-32k/corpora" \
  --out "$OUT/results/kld_results.csv" \
  --ctx 8192 --two-pass --model-class Qwen3_5ForCausalLM \
  2>&1 | tee "$OUT/logs/kld.log" | grep -vE "^  \[|it/s\]"
echo "KLD exit code: ${PIPESTATUS[0]}"

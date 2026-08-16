#!/usr/bin/env bash
# exp-060 AWQ arm: Qwen3.8-27B -> IQ2_M via AWQ + hybrid_custom imatrix.
# The A/B opponent for out/exp-060-32k/iq2_m/Qwen3.8-27B-IQ2_M.gguf.
# See docs/qwen3_8_awq_2bit_handoff.md; recipe carries the per-knob rationale.
#
# Preconditions this script enforces (all cheap, all fatal-early):
#   - the GPU is actually free (a concurrent GPTQ run will OOM this)
#   - enough disk for the folded HF checkpoint + folded F16 + IQ2_M
#   - the pinned corpus and the reused FP16 KLD baselines are in place
set -euo pipefail

cd /workspace/Quant-Tuner

RECIPE=src/quant_tuner/recipes/iq2_m_qwen3_8_awq.yaml
WS=out/exp-060-32k-awq
LOG="$WS/logs/run-awq.log"

# ~114 GB: folded HF bf16 copy (~53) + folded F16 GGUF (~51) + IQ2_M (~10).
NEED_GB=120
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "FATAL: ${FREE_GB}G free on /workspace, need ~${NEED_GB}G." >&2
  echo "  Free the IQ3_M/IQ4_XS/Q5_K_M rungs (they are on HF) if headroom is short." >&2
  exit 1
fi

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED_MIB" -gt 8000 ]; then
  echo "FATAL: GPU has ${USED_MIB} MiB in use — another run is still active." >&2
  echo "  AWQ needs the whole card (bf16 27B forward at ctx 32768). Wait for it." >&2
  exit 1
fi

for f in "$WS/model_extracted/config.json" \
         "$WS/gguf/model-f16.gguf" \
         "$WS/eval/baseline.kld" \
         "$WS/eval/baseline-tools.kld" \
         data/chat_templates/qwen3_8_safe_v2.jinja \
         out/exp-060-32k/corpora/corpus.cal.txt; do
  [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$WS/logs"
echo "launching: $(date -Is)  free=${FREE_GB}G  gpu_used=${USED_MIB}MiB" | tee -a "$LOG"

# Unbuffered so the log is tailable while the (multi-hour) stages run.
exec .venv/bin/python -u -m quant_tuner.cli run --recipe "$RECIPE" 2>&1 | tee -a "$LOG"

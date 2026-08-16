#!/usr/bin/env bash
# exp-062 third rung: IQ4_XS via AWQ + hybrid_custom imatrix.
#
# Separate from run_exp062_awq.sh on purpose — that script was already executing
# when this rung was added, and bash re-reads a running script from disk as it
# goes, so editing it mid-run can corrupt execution.
#
# GATED: refuses to start unless the IQ2_M and IQ3_M rungs both produced a GGUF.
# The point of this rung is "does the new recipe hold up at 4 bits", which is only
# a question worth GPU hours if the recipe worked at 2 and 3.
set -euo pipefail
cd /workspace/Quant-Tuner

RECIPE=src/quant_tuner/recipes/iq4_xs_qwen3_8_awq.yaml
WS=out/exp-062-awq-iq4xs
NEED_GB=130   # folded HF (~53) + folded F16 (~51) + IQ4_XS (~15) + slack
LOG=out/exp-062-32k/logs/run-exp062.log
CORPUS=out/exp-062-32k/corpora/corpus.cal.txt
TMPL=data/chat_templates/qwen3_8_safe_v2.jinja

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

IQ2=out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf
IQ3=out/exp-062-awq-iq3m/gguf/IQ3_M-awq-best-hybrid_custom-mtp.gguf
for prereq in "$IQ2" "$IQ3"; do
  [ -f "$prereq" ] || { log "GATED: prerequisite rung missing ($prereq) — not starting IQ4_XS."; exit 1; }
done
log "gate OK: IQ2_M and IQ3_M rungs both built"

FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge "$NEED_GB" ] || { log "FATAL: ${FREE_GB}G free, need ~${NEED_GB}G."; exit 1; }

GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$GPU_MIB" -le 8000 ] || { log "FATAL: GPU has ${GPU_MIB} MiB in use — another run is active."; exit 1; }

for f in "$WS/model_extracted/config.json" "$WS/gguf/model-f16.gguf" \
         "$WS/eval/baseline.kld" "$WS/eval/baseline-tools.kld" "$CORPUS" "$TMPL"; do
  [ -e "$f" ] || { log "FATAL: missing $f"; exit 1; }
done

log "=============================================================="
log "RUNG: $RECIPE  ->  $WS   (free=${FREE_GB}G gpu=${GPU_MIB}MiB)"
log "=============================================================="
mkdir -p "$WS/logs"
.venv/bin/python -u -m quant_tuner.cli run --recipe "$RECIPE" 2>&1 \
  | tee -a "$WS/logs/run.log" "$LOG"

log "### IQ4_XS rung finished"

# Safe only after the quant exists: the fold is an input to calibration+quantize
# and to nothing downstream (bench reads the quant and the symlinked original F16).
if ls "$WS"/gguf/*-awq-*.gguf >/dev/null 2>&1; then
  before=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  rm -rf "$WS/model_awq" "$WS/gguf/model-f16-awq.gguf"
  after=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  log "reclaimed $WS fold: ${before}G -> ${after}G free"
else
  log "REFUSING to reclaim $WS — no finished quant there."
fi

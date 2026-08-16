#!/usr/bin/env bash
# exp-062: rebuild the Qwen3.8-27B 2-bit AND 3-bit rungs via AWQ + hybrid_custom
# imatrix, on the tool-dense sft_chat_train corpus, with qwen3_8_safe_v2 baked in.
#
# Runs the two rungs SEQUENTIALLY and reclaims between them. That is not tidiness:
# each rung's AWQ fold costs ~103 GB (folded HF bf16 copy + folded F16 GGUF) and
# two of them do not fit on this disk at once. The folds cannot be shared — AWQ's
# alpha search scores through a proxy quantizer chosen from quantize.type, so the
# 2-bit and 3-bit searches produce different scales and different folded weights.
#
# Preconditions enforced per rung (all cheap, all fatal-early), because every one
# of these failures otherwise shows up only as a mediocre benchmark number hours
# later, indistinguishable from "AWQ just didn't help".
set -euo pipefail
cd /workspace/Quant-Tuner

CORPUS=out/exp-062-32k/corpora/corpus.cal.txt
TMPL=data/chat_templates/qwen3_8_safe_v2.jinja
LOG=out/exp-062-32k/logs/run-exp062.log
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

run_rung() {
  local recipe="$1" ws="$2" need_gb="$3"

  log "=============================================================="
  log "RUNG: $recipe  ->  $ws"
  log "=============================================================="

  local free_gb gpu_mib
  free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$need_gb" ]; then
    log "FATAL: ${free_gb}G free, need ~${need_gb}G for this rung."
    return 1
  fi

  # AWQ needs the whole card (bf16 27B forward at ctx 32768). A neighbour process
  # holding memory does not slow this down, it OOMs it.
  gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$gpu_mib" -gt 8000 ]; then
    log "FATAL: GPU has ${gpu_mib} MiB in use — another run is still active."
    return 1
  fi

  for f in "$ws/model_extracted/config.json" "$ws/gguf/model-f16.gguf" \
           "$ws/eval/baseline.kld" "$ws/eval/baseline-tools.kld" \
           "$CORPUS" "$TMPL"; do
    [ -e "$f" ] || { log "FATAL: missing $f"; return 1; }
  done

  log "preflight OK: free=${free_gb}G gpu_used=${gpu_mib}MiB"
  mkdir -p "$ws/logs"
  # Unbuffered so the log is tailable while the multi-hour stages run.
  .venv/bin/python -u -m quant_tuner.cli run --recipe "$recipe" 2>&1 \
    | tee -a "$ws/logs/run.log" "$LOG"
}

reclaim() {
  local ws="$1"
  # Safe ONLY after llama-quantize has produced the rung's GGUF: the folded HF
  # copy and folded F16 are inputs to calibration + quantize and to nothing
  # downstream. bench reads the QUANT and the symlinked original F16, never these.
  if ! ls "$ws"/gguf/*-awq-*.gguf >/dev/null 2>&1; then
    log "REFUSING to reclaim $ws — no finished quant there; keeping the fold."
    return 0
  fi
  local before after
  before=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  rm -rf "$ws/model_awq" "$ws/gguf/model-f16-awq.gguf"
  after=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  log "reclaimed $ws fold: ${before}G -> ${after}G free"
}

log "### exp-062 starting"
log "corpus: $CORPUS  ($(wc -c < "$CORPUS") bytes)"

run_rung src/quant_tuner/recipes/iq2_m_qwen3_8_awq_v2.yaml out/exp-062-awq-iq2m 120
log "### IQ2_M rung finished"
reclaim out/exp-062-awq-iq2m

run_rung src/quant_tuner/recipes/iq3_m_qwen3_8_awq.yaml out/exp-062-awq-iq3m 125
log "### IQ3_M rung finished"
reclaim out/exp-062-awq-iq3m

log "### exp-062 complete — both rungs built"
log "next: scripts/validate_exp062_awq.sh, then the tool-call smoke"

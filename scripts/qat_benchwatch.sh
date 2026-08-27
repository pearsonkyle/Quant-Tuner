#!/usr/bin/env bash
# Periodic CPU sidecar benchmark of QAT checkpoints while the trainer runs (the coder1
# lesson: masked-CE and the stop probe are both blind to agentic capability regressions —
# coder1@s800 fixated on /testbed, rambled 8k tokens without a tool call, and garbled path
# literals while its probe stayed textbook). Every ~EVERY steps: export Q2_0 on CPU, GGUF
# stop probe, EPISODES graded mimic episodes at T=0.7, prune export intermediates, and
# emit one "[benchwatch]" summary line (tail it into a monitor). GPU is never touched.
# Env: RUN PREFIX EVERY EPISODES.
cd /workspace/Quant-Tuner
RUN="${RUN:-out/exp-059/kd32b-full-coder2}"
PREFIX="${PREFIX:-coder2}"
EVERY="${EVERY:-400}"
EPISODES="${EPISODES:-3}"
BIN=vendor/llama.cpp-prism/build/bin/llama-server
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
LAST=0
until [ -f "$RUN/train.log" ]; do sleep 60; done
while pgrep -f "quant_tuner[.]qat[.]train" >/dev/null; do
  N=$(ls "$RUN"/trained_latents.step*.pt 2>/dev/null | sed -E 's/.*step([0-9]+)\.pt/\1/' | sort -n | tail -1)
  if [ -z "${N:-}" ] || [ "$N" -lt $((LAST+EVERY)) ]; then sleep 300; continue; fi
  LAST=$N; TAG=${PREFIX}-s${N}; UTAG=$(echo "$TAG" | tr '[:lower:]' '[:upper:]')
  echo "[benchwatch] step $N: exporting $TAG"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nice -n 10 .venv/bin/python scripts/exp057_qat_export.py \
      --latents "$RUN/trained_latents.step${N}.pt" --tag "$TAG" >/dev/null 2>&1
  GGUF=out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf
  [ -f "$GGUF" ] || { echo "[benchwatch] step $N: EXPORT FAILED"; continue; }
  CUDA_VISIBLE_DEVICES="" nice -n 10 "$BIN" --model "$GGUF" --ctx-size 32768 --n-gpu-layers 0 \
      --threads 48 --jinja --host 127.0.0.1 --port 18096 \
      > "out/exp-059/benchwatch_server_${TAG}.log" 2>&1 &
  SRV=$!
  ok=0
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:18096/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 5
  done
  [ "$ok" = 1 ] || { echo "[benchwatch] step $N: SERVER FAILED"; kill $SRV 2>/dev/null; continue; }
  PROBE=$(PYTHONPATH=src .venv/bin/python scripts/probe_stop_prob.py \
      --base-url http://127.0.0.1:18096 --label "${TAG}-cpu" 2>/dev/null \
      | grep -E "sentence_period|after_tool_call" | tr -s ' ' | tr '\n' ';')
  for _ in $(seq 1 $EPISODES); do
    (cd /workspace/swe-mimic && timeout 5400 .venv/bin/python run_agent.py \
        --instance instance.json --base-url http://127.0.0.1:18096/v1 \
        --label "${UTAG}-CPU" --temperature 0.7 \
        --out "benchwatch_${PREFIX}.csv" >/dev/null 2>&1) || true
  done
  kill $SRV 2>/dev/null
  SUMMARY=$(tail -$EPISODES "/workspace/swe-mimic/benchwatch_${PREFIX}.csv" | awk -F, -v n=$EPISODES \
    '{r+=$3;p+=$4;s+=$9;nz+=$14;ot+=$15; if($18!~"completed")e++} END{printf "resolved=%d/%d patch=%d/%d avg_steps=%.0f avg_nonzero=%.0f avg_out_tok=%.0f abnormal_exit=%d", r,n,p,n,s/n,nz/n,ot/n,e+0}')
  TB=$(python3 -c "import json;t=json.load(open('/workspace/swe-mimic/work/dask__dask-11393/traj_${UTAG}-CPU.json'));print('%d/%d'%(sum('/testbed' in m['cmd'] for m in t),len(t)))" 2>/dev/null || echo "?")
  echo "[benchwatch] step $N: $SUMMARY testbed_cmds_lastep=$TB probe=$PROBE"
  bash scripts/prune_export_intermediates.sh "$TAG" >/dev/null 2>&1 || true
done
echo "[benchwatch] trainer exited — final checkpoint benching handled by the chain's eval"

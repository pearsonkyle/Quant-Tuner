#!/usr/bin/env bash
# Generate agent episodes on the harvest instances with a BARE model (no sampling
# penalties — loops are the harvest target). One llama-server, one episode per instance.
set -uo pipefail
cd /workspace/swe-mimic
Q=/workspace/Quant-Tuner
BIN=${BIN:-$Q/vendor/llama.cpp-prism/build/bin/llama-server}
GGUF=${GGUF:-$Q/out/exp-057/Ternary-Bonsai-8B-anchor9-Q2_0.gguf}
LABEL=${LABEL:-HARVEST-anchor9}
PORT=${PORT:-18092}
CTX=${CTX:-32768}

"$BIN" --model "$GGUF" --ctx-size "$CTX" --n-gpu-layers 999 --jinja \
       --flash-attn on --host 127.0.0.1 --port "$PORT" \
       > "logs_server_$LABEL.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
ok=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[gen] server died"; exit 1; }
  sleep 5
done
[ "$ok" = "1" ] || { echo "[gen] server never healthy"; exit 1; }

PY=/workspace/swe-mimic/.venv/bin/python   # the mimic harness env (has the agents SDK)
for w in work/*/; do
  iid=$(basename "$w")
  [ -f "$w/instance.json" ] || continue
  case "$iid" in dask__*) continue;; esac      # eval instance stays untouched
  echo "[gen] episode: $iid"
  timeout 3600 "$PY" run_agent.py --instance "$w/instance.json" \
      --base-url "http://127.0.0.1:$PORT/v1" --label "$LABEL" \
      --skip-gate --out harvest_results.csv 2>&1 | tail -3
done
echo "[gen] done"

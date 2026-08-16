#!/usr/bin/env bash
# MTP speculative decoding on the W4A16 checkpoint: acceptance and speed-up vs draft-n.
#
# The card currently says "shipped but not verified". This either earns that claim or
# withdraws it, and picks the right num_speculative_tokens while it is at it.
#
# Why sweeping n is a real question, not a formality
# --------------------------------------------------
# Qwen3.8 declares mtp_num_hidden_layers = 1 — ONE nextn layer. llama.cpp therefore caps
# the draft at 1. vLLM may either (a) refuse n > 1 outright, or (b) accept it by applying
# the single head autoregressively, in which case each extra position is drafted from the
# head's OWN previous guess and acceptance should fall off sharply with depth. Both are
# informative and neither is safe to assume, so we try each n and record what happened.
#
# What is measured per n:
#   started         — did vLLM come up at all with this speculative_config
#   acceptance      — accepted_tokens / draft_tokens  (per-token hit rate)
#   accept_per_step — accepted_tokens / drafts        (mean tokens gained per draft step)
#   decode tok/s    — single-stream, decode-bound
#   speedup         — vs the SAME checkpoint with no speculative decoding
#
# Counters come from vLLM's Prometheus endpoint, not a log grep: log formats change
# between releases and a silently-missing grep reads as "no problem".
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=$OUT/checkpoint-vllm-mtp
PLAIN=$OUT/checkpoint-vllm
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
PORT=18080
DRAFT_NS=${DRAFT_NS:-"1 2 3"}
RESULTS=$OUT/results/mtp_sweep.json
mkdir -p "$OUT/logs" "$OUT/results"
cd $REPO

stop_server() {
  [ -f "$OUT/vllm.pid" ] || return 0
  local pid; pid=$(cat "$OUT/vllm.pid")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$OUT/vllm.pid"
}

start_server() {   # start_server <model-dir> <logfile> [extra args…]
  local model=$1 log=$2; shift 2
  nohup $REPO/.venv-vllm/bin/vllm serve "$model" \
    --served-model-name local --max-model-len 32768 --port $PORT \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --chat-template "$TEMPLATE" --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 "$@" > "$log" 2>&1 &
  local pid=$!; echo "$pid" > "$OUT/vllm.pid"
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "  healthy"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  SERVER DIED"; grep -iE "error|assert|not supported|invalid|raise" "$log" | tail -8; return 1; }
    sleep 5
  done
  echo "  TIMEOUT"; tail -20 "$log"; return 1
}

# Single-stream long generation: decode-bound, so tok/s reflects the draft/verify loop
# rather than prefill or batching. Best-of-3 to shed scheduling noise.
bench_decode() {
  $REPO/.venv/bin/python - <<'PY'
import json, time, urllib.request
body = {"model": "local",
        "messages": [{"role": "user", "content": "Write a detailed technical explanation of how B-tree indexes work in relational databases, including node structure, splits, and range scans. Be thorough."}],
        "max_tokens": 600, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
best = 0.0
for _ in range(3):
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:18080/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"}), timeout=600).read())
    except Exception:
        continue
    dt = time.time() - t
    n = r.get("usage", {}).get("completion_tokens", 0)
    if n and dt > 0:
        best = max(best, n / dt)
print(f"{best:.2f}")
PY
}

# accepted, draft, drafts
spec_counters() {
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | \
    awk '/^vllm:spec_decode_num_accepted_tokens_total/ {a+=$2}
         /^vllm:spec_decode_num_draft_tokens_total/    {d+=$2}
         /^vllm:spec_decode_num_drafts_total/          {s+=$2}
         END {printf "%d %d %d\n", a+0, d+0, s+0}'
}

echo "=================================================================="
echo "BASELINE — same weights, no speculative decoding"
echo "=================================================================="
stop_server; sleep 10
if start_server "$PLAIN" "$OUT/logs/vllm_mtp_baseline.log"; then
  BASE_TPS=$(bench_decode)
else
  BASE_TPS=0
fi
echo "  baseline decode: ${BASE_TPS} tok/s"

ROWS="$OUT/results/.mtp_rows.jsonl"; : > "$ROWS"
for n in $DRAFT_NS; do
  echo
  echo "=================================================================="
  echo "num_speculative_tokens = $n"
  echo "=================================================================="
  stop_server; sleep 15
  log="$OUT/logs/vllm_mtp_n${n}.log"
  if start_server "$CKPT" "$log" \
       --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$n}"; then
    read -r A0 D0 S0 <<<"$(spec_counters)"
    TPS=$(bench_decode)
    read -r A1 D1 S1 <<<"$(spec_counters)"
    ACC=$((A1-A0)); DRAFT=$((D1-D0)); STEPS=$((S1-S0))
    echo "  decode ${TPS} tok/s | drafted ${DRAFT} | accepted ${ACC} | steps ${STEPS}"
    printf '{"n":%d,"started":true,"tok_s":%s,"accepted":%d,"draft":%d,"drafts":%d}\n' \
      "$n" "$TPS" "$ACC" "$DRAFT" "$STEPS" >> "$ROWS"
  else
    echo "  did not start at n=$n"
    printf '{"n":%d,"started":false,"tok_s":0,"accepted":0,"draft":0,"drafts":0}\n' "$n" >> "$ROWS"
  fi
done

stop_server

echo
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
$REPO/.venv/bin/python - "$BASE_TPS" "$ROWS" "$RESULTS" <<'PY'
import json, sys
base, rows_path, out = float(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(rows_path) if l.strip()]
for r in rows:
    r["acceptance"]      = round(r["accepted"] / r["draft"], 4) if r["draft"] else None
    r["accept_per_step"] = round(r["accepted"] / r["drafts"], 3) if r["drafts"] else None
    r["speedup"]         = round(r["tok_s"] / base, 3) if base else None

print(f"{'n':>3} {'started':>8} {'tok/s':>9} {'speedup':>8} {'accept':>8} {'acc/step':>9}")
print(f"{'--':>3} {'baseline':>8} {base:9.2f} {'1.000':>8} {'—':>8} {'—':>9}")
for r in rows:
    if not r["started"]:
        print(f"{r['n']:>3} {'NO':>8} {'—':>9} {'—':>8} {'—':>8} {'—':>9}")
        continue
    acc = f"{r['acceptance']:.1%}" if r["acceptance"] is not None else "—"
    aps = f"{r['accept_per_step']:.2f}" if r["accept_per_step"] is not None else "—"
    print(f"{r['n']:>3} {'yes':>8} {r['tok_s']:9.2f} {r['speedup']:8.3f} {acc:>8} {aps:>9}")

live = [r for r in rows if r["started"] and r["draft"]]
if not live:
    verdict = "MTP UNUSABLE — no configuration produced draft tokens"
    best = None
else:
    best = max(live, key=lambda r: r["speedup"] or 0)
    if (best["speedup"] or 0) < 1.05:
        verdict = (f"NO USEFUL GAIN — best is n={best['n']} at "
                   f"{best['speedup']:.2f}x ({best['acceptance']:.1%} acceptance)")
    else:
        verdict = (f"WORKS — best n={best['n']}: {best['speedup']:.2f}x decode at "
                   f"{best['acceptance']:.1%} acceptance")
res = {"baseline_tok_s": round(base, 2), "rows": rows,
       "best_n": best["n"] if best else None, "verdict": verdict}
json.dump(res, open(out, "w"), indent=2)
print("\n->", verdict)
print("wrote", out)
PY

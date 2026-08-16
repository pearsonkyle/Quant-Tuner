#!/usr/bin/env bash
# Validate the grafted multimodal + MTP checkpoint.
#
# Four questions, in dependency order. Each is a gate: there is no point measuring
# speculative acceptance on a server that will not start, or claiming vision works
# because the tensors are present.
#
#   1. Does it serve AT ALL with no rename step? (the multimodal mapper should accept
#      model.language_model.* / model.visual.* directly)
#   2. Does the vision tower actually see? (synthetic 3-shape image, colour+shape+position)
#   3. Does MTP speculative decoding start, draft, and get accepted? (sweep draft-n)
#   4. Did the text side regress? (same 174 turns, thinking off, vs 0.557)
#
# Nothing here is asserted from the checkpoint's contents — every claim is measured
# against the running server.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
CKPT=$OUT/checkpoint-mm-graft
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
HOLDOUT=$REPO/out/exp-060-32k/eval/toolcall_holdout.jsonl
PORT=18080
DRAFT_NS=${DRAFT_NS:-"1 2"}
mkdir -p "$OUT/logs" "$OUT/results"
cd $REPO

stop_server() {
  [ -f "$OUT/vllm.pid" ] || return 0
  local pid; pid=$(cat "$OUT/vllm.pid")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$OUT/vllm.pid"
}

start_server() {   # start_server <log> [extra args…]
  local log=$1; shift
  nohup $REPO/.venv-vllm/bin/vllm serve "$CKPT" \
    --served-model-name local --max-model-len 32768 --port $PORT \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --chat-template "$TEMPLATE" --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 "$@" > "$log" 2>&1 &
  local pid=$!; echo "$pid" > "$OUT/vllm.pid"
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "  healthy"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  SERVER DIED"; grep -iE "error|not supported|no module|missing|raise|assert" "$log" | tail -10; return 1; }
    sleep 5
  done
  echo "  TIMEOUT"; tail -20 "$log"; return 1
}

spec_counters() {
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | \
    awk '/^vllm:spec_decode_num_accepted_tokens_total/ {a+=$2}
         /^vllm:spec_decode_num_draft_tokens_total/    {d+=$2}
         /^vllm:spec_decode_num_drafts_total/          {s+=$2}
         END {printf "%d %d %d\n", a+0, d+0, s+0}'
}

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

echo "waiting for a free GPU …"
for _ in $(seq 1 240); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && { echo "GPU free (${used} MiB)"; break; }
  sleep 30
done

echo
echo "=================================================================="
echo "GATE 1/4 — does the grafted checkpoint serve, with no rename?"
echo "=================================================================="
stop_server; sleep 10
if ! start_server "$OUT/logs/vllm_graft.log"; then
  echo "GRAFT FAILED TO SERVE — stopping here."
  exit 1
fi
BASE_TPS=$(bench_decode)
echo "  baseline decode: ${BASE_TPS} tok/s"

echo
echo "=================================================================="
echo "GATE 2/4 — vision: does the encoder actually see?"
echo "=================================================================="
PYTHONPATH=src $REPO/.venv/bin/python scripts/vision_smoke.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$OUT/results/vision_smoke.json" 2>&1 | tee "$OUT/logs/vision_smoke.log"

echo
echo "=================================================================="
echo "GATE 3/4 — text side: 174 turns, thinking off (expect ~0.557)"
echo "=================================================================="
PYTHONPATH=src $REPO/.venv/bin/python scripts/eval_toolcall.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --holdout "$HOLDOUT" \
  --out "$OUT/results/toolcall_graft_off.csv" \
  --log-dir "$OUT/results/toolcall_graft_off_logs" \
  --temperature 0 --ctx 32768 --no-stop-on-fail \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "$OUT/logs/toolcall_graft_off.log" 2>&1
grep -E "Tool selection accuracy|Param accuracy|Schema-valid" "$OUT/logs/toolcall_graft_off.log" \
  || echo "   (no result — check log)"

echo
echo "=================================================================="
echo "GATE 4/4 — MTP speculative decoding, sweeping draft-n"
echo "=================================================================="
ROWS="$OUT/results/.graft_mtp_rows.jsonl"; : > "$ROWS"
for n in $DRAFT_NS; do
  echo "--- num_speculative_tokens = $n"
  stop_server; sleep 15
  if start_server "$OUT/logs/vllm_graft_mtp_n${n}.log" \
       --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$n}"; then
    read -r A0 D0 S0 <<<"$(spec_counters)"
    TPS=$(bench_decode)
    read -r A1 D1 S1 <<<"$(spec_counters)"
    echo "  decode ${TPS} tok/s | drafted $((D1-D0)) | accepted $((A1-A0)) | steps $((S1-S0))"
    printf '{"n":%d,"started":true,"tok_s":%s,"accepted":%d,"draft":%d,"drafts":%d}\n' \
      "$n" "$TPS" "$((A1-A0))" "$((D1-D0))" "$((S1-S0))" >> "$ROWS"
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
$REPO/.venv/bin/python - "$BASE_TPS" "$ROWS" "$OUT/results/graft_validation.json" "$OUT/results/vision_smoke.json" <<'PY'
import json, os, sys
base, rows_path, out, vision_path = sys.argv[1:5]
base = float(base)
rows = [json.loads(l) for l in open(rows_path) if l.strip()]
for r in rows:
    r["acceptance"]      = round(r["accepted"] / r["draft"], 4) if r["draft"] else None
    r["accept_per_step"] = round(r["accepted"] / r["drafts"], 3) if r["drafts"] else None
    r["speedup"]         = round(r["tok_s"] / base, 3) if base else None

vision = json.load(open(vision_path)) if os.path.exists(vision_path) else {}
print(f"serves without rename : YES")
print(f"vision                : {vision.get('verdict', 'NOT RUN')} ({vision.get('score','?')}/3)")
print(f"baseline decode       : {base:.2f} tok/s")
print(f"\n{'n':>3} {'started':>8} {'tok/s':>9} {'speedup':>8} {'accept':>8} {'acc/step':>9}")
for r in rows:
    if not r["started"]:
        print(f"{r['n']:>3} {'NO':>8} {'—':>9} {'—':>8} {'—':>8} {'—':>9}"); continue
    acc = f"{r['acceptance']:.1%}" if r["acceptance"] is not None else "—"
    aps = f"{r['accept_per_step']:.2f}" if r["accept_per_step"] is not None else "—"
    print(f"{r['n']:>3} {'yes':>8} {r['tok_s']:9.2f} {r['speedup']:8.3f} {acc:>8} {aps:>9}")

live = [r for r in rows if r["started"] and r["draft"]]
best = max(live, key=lambda r: r["speedup"] or 0) if live else None
if best:
    mtp = f"WORKS — best n={best['n']}: {best['speedup']:.2f}x at {best['acceptance']:.1%} acceptance"
elif any(r["started"] for r in rows):
    mtp = "STARTS BUT INACTIVE — zero draft tokens"
else:
    mtp = "STILL BROKEN — no configuration started"
json.dump({"baseline_tok_s": round(base, 2), "rows": rows,
           "vision": vision.get("verdict"), "vision_score": vision.get("score"),
           "mtp_verdict": mtp}, open(out, "w"), indent=2)
print(f"\nMTP -> {mtp}")
print("wrote", out)
PY

#!/usr/bin/env bash
# Rebuild the W4A16 export through the MULTIMODAL class, then prove vision survived.
#
# This is the build the original run should have been. Two problems collapse into one fix:
#
#   1. Vision. AutoModelForCausalLM resolves qwen3_5 to the TEXT-ONLY Qwen3_5ForCausalLM,
#      whose module tree has no `visual.*` — so all 333 vision tensors were absent from
#      the export. Declaring the class explicitly makes the tree match the checkpoint 1:1
#      (audited: only the 15 mtp.* tensors drop, and those cannot be kept at any setting).
#   2. The rename hack. vLLM's MULTIMODAL mapper expects exactly the naming transformers
#      writes (model.visual.* / model.language_model.*), so this checkpoint needs NO
#      post-hoc tensor rename to serve — unlike the text-only build.
#
# --ignore 're:.*visual.*' is REQUIRED and is the whole point of the audit. The default
# ignore list is gemma-shaped: its `vision_tower` pattern matches ZERO modules here, so
# without this flag the vision tower is silently quantized to int4 against a calibration
# corpus that contains no images at all. Verified by --dry-run-ignore: 281 modules matched.
set -uo pipefail

REPO=/workspace/Quant-Tuner
OUT=$REPO/out/exp-060-w4a16-32k
MM=$OUT/checkpoint-mm
REF=$REPO/out/exp-060/model_extracted
TEMPLATE=$REPO/data/chat_templates/qwen3_8_safe_v2.jinja
HOLDOUT=$REPO/out/exp-060-32k/eval/toolcall_holdout.jsonl
PORT=18080
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

# ---- 1. PTQ through the multimodal class (~3h) ---------------------------------------
if [ -f "$MM/config.json" ]; then
  echo "=== multimodal export already present at $MM — skipping PTQ ==="
else
  stop_server; sleep 15
  echo "=== PTQ (multimodal class, vision tower protected) — this takes ~3h ==="
  PYTHONPATH=src $REPO/.venv/bin/python scripts/run_vllm_ptq.py \
    --model "$REF" \
    --model-class Qwen3_5ForConditionalGeneration \
    --corpus "$REPO/out/exp-060-32k/corpora/corpus.cal.txt" \
    --out "$MM" \
    --ctx 32768 --scheme W4A16 \
    --ignore 're:.*visual.*' \
    > "$OUT/logs/ptq_mm.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { echo "PTQ FAILED (rc=$rc)"; tail -40 "$OUT/logs/ptq_mm.log"; exit 1; }
  echo "=== PTQ done ==="
fi

# ---- 2. audit the export before spending GPU time serving it -------------------------
$REPO/.venv/bin/python - "$MM" <<'PY'
import glob, json, os, struct, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    names = list(json.load(open(idx))["weight_map"])
else:
    names = []
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        h.pop("__metadata__", None); names += list(h)
vis = [n for n in names if "visual" in n]
print(f"total tensors      : {len(names)}")
print(f"visual.*           : {len(vis)}")
print(f"language_model.*   : {sum('language_model' in n for n in names)}")
# The vision tower must be UNQUANTIZED: a quantized module shows up as *_packed.
packed = [n for n in vis if "packed" in n or "weight_scale" in n]
print(f"visual quantized?  : {len(packed)} packed/scale tensors "
      f"({'BAD — tower was quantized' if packed else 'good — tower kept bf16'})")
cfg = json.load(open(os.path.join(d, "config.json")))
print(f"architectures      : {cfg.get('architectures')}")
assert cfg.get("quantization_config"), "no quantization_config — vLLM would serve bf16!"
assert vis, "NO VISION TENSORS — the export is text-only again"
assert not packed, "vision tower got quantized — --ignore did not take"
print("audit PASSED")
PY
[ $? -eq 0 ] || { echo "EXPORT AUDIT FAILED — not serving it"; exit 1; }

# ---- 3. serve WITHOUT any rename, which is the point ---------------------------------
echo "=== serving the multimodal export (no rename step) ==="
stop_server; sleep 15
nohup $REPO/.venv-vllm/bin/vllm serve "$MM" \
  --served-model-name local --max-model-len 32768 --port $PORT \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --chat-template "$TEMPLATE" --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 > "$OUT/logs/vllm_mm.log" 2>&1 &
pid=$!; echo "$pid" > "$OUT/vllm.pid"
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "healthy"; break; }
  kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$OUT/logs/vllm_mm.log"; exit 1; }
  sleep 5
done

# ---- 4. vision smoke test ------------------------------------------------------------
echo "=== vision smoke test ==="
PYTHONPATH=src $REPO/.venv/bin/python scripts/vision_smoke.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$OUT/results/vision_smoke.json" 2>&1 | tee "$OUT/logs/vision_smoke.log"

# ---- 5. confirm the text side did not regress ----------------------------------------
echo "=== tool-call re-validation (thinking off) on the multimodal build ==="
PYTHONPATH=src $REPO/.venv/bin/python scripts/eval_toolcall.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --holdout "$HOLDOUT" \
  --out "$OUT/results/toolcall_mm_off.csv" \
  --log-dir "$OUT/results/toolcall_mm_off_logs" \
  --temperature 0 --ctx 32768 --no-stop-on-fail \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "$OUT/logs/toolcall_mm_off.log" 2>&1
grep -E "Tool selection accuracy|Param accuracy|Schema-valid" "$OUT/logs/toolcall_mm_off.log" \
  || echo "   (no result — check log)"

echo "=== multimodal chain complete ==="

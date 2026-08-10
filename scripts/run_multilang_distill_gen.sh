#!/bin/bash
# Multi-language distillation data generation (SWE-rebench-V2).
#
# Harvests VERIFIED solutions across 8 languages (python, go, ts, js, rust, java, php,
# kotlin) from a strong solver, so the ternary student stops being trained on Python
# alone. Same contract as run_ornith_distill_gen.sh — one clean Docker container per
# instance, graded by actually running the gold tests — but the instances come from
# SWE-rebench-V2 and the grader dispatches on each instance's own log parser.
#
# The solver here is served by an ALREADY-RUNNING OpenAI-compatible server (LM Studio
# by default), so no llama-server is spawned and the GPU stays under your control:
#
#     BASE_URL=http://localhost:1234/v1  MODEL=ornith-1.0-35b
#
# Pool: out/external/swe-rebench/distill_train_multilang.jsonl, built --exclude-disjoint
# from the multilingual EVAL holdout, so the student is never graded on what it trained
# on. Rebuild both with scripts/build_swebench_holdout.py (see docs/ternary_qat.md).
#
# --resume skips any instance already graded, so a crash/restart never re-does work.
# Run scripts/docker_housekeep.sh alongside this: --cleanup-images untags each image and
# leaves <none> dangling layers that will otherwise fill the Docker VM disk.
#
# Usage: run_multilang_distill_gen.sh [N_INSTANCES]
set -u
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src
PY=.venv/bin/python

BASE_URL="${BASE_URL:-http://localhost:1234/v1}"
MODEL="${MODEL:-ornith-1.0-35b}"
POOL="${POOL:-out/external/swe-rebench/distill_train_multilang.jsonl}"
WS="${WS:-out/swe-rebench/multilang-distill-gen}"
N="${1:-0}"   # 0 = whole pool

if [ ! -f "$POOL" ]; then
  echo "ERROR: pool $POOL not found. Build it first:"
  echo "  $PY scripts/download_swebench_dataset.py --dataset nebius/SWE-rebench-V2"
  echo "  $PY scripts/build_swebench_holdout.py --from-local out/external/swe-rebench/v2_all.jsonl \\"
  echo "      --languages python,go,ts,js,rust,java,php,kotlin --difficulty medium \\"
  echo "      --max-f2p 25 --n 24 --seed 42 --out out/external/swe-rebench/holdout_multilang.jsonl"
  exit 1
fi

# Fail fast if the server isn't up or isn't serving MODEL — otherwise every instance
# burns a container before erroring.
if ! curl -sf --max-time 10 "${BASE_URL%/v1}/v1/models" -o /tmp/_models.json; then
  echo "ERROR: no OpenAI-compatible server at $BASE_URL"; exit 1
fi
if ! grep -q "\"$MODEL\"" /tmp/_models.json; then
  echo "ERROR: '$MODEL' is not served at $BASE_URL. Available:"
  $PY -c "import json;[print('  ',m['id']) for m in json.load(open('/tmp/_models.json'))['data']]"
  exit 1
fi

RUN_POOL="$POOL"
if [ "$N" -gt 0 ]; then
  RUN_POOL="$WS/pool_first_${N}.jsonl"
  mkdir -p "$WS"
  head -n "$N" "$POOL" > "$RUN_POOL"
fi

echo "=== [$(date)] multilang distill-gen START ==="
echo "    model : $MODEL @ $BASE_URL"
echo "    pool  : $RUN_POOL ($(wc -l < "$RUN_POOL") instances)"
$PY -c "
import json,collections
c=collections.Counter(json.loads(l).get('language','?') for l in open('$RUN_POOL'))
print('    langs :', ', '.join(f'{k}={v}' for k,v in sorted(c.items())))"

$PY -u scripts/run_swebench_eval.py \
  --base-url "$BASE_URL" \
  --target-model-name "$MODEL" \
  --holdout "$RUN_POOL" \
  --workspace "$WS" \
  --agent openai-agents \
  --temperature 0.25 \
  --max-steps 100 \
  --resume --cleanup-images --progress

echo "=== [$(date)] multilang distill-gen DONE ==="
TRAJ="$WS/trajectories/$MODEL"
$PY -c "
import json,glob,collections
res=[json.load(open(f)) for f in glob.glob('$TRAJ/*.result.json')]
ok=[r for r in res if r.get('resolved')]
print(f'graded={len(res)}  resolved={len(ok)}')
by=collections.Counter(r.get('language','?') for r in ok)
tot=collections.Counter(r.get('language','?') for r in res)
for k in sorted(tot): print(f'  {k:8s} {by[k]}/{tot[k]} resolved')" 2>/dev/null || true

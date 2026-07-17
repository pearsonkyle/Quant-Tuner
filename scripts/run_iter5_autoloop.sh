#!/bin/bash
# iter-5 AUTO-LOOP: grow verified data -> retrain -> dual-bench, repeat until the model
# generalizes (or a safety cap). One unattended driver for "keep going until it's better."
#
# Each round:
#   1. (round 1) use the trajectories already generated; (round >1) build a FRESH batch of
#      instances disjoint from the eval holdout AND everything already tried, append to the
#      pool, and generate Ornith trajectories over the new instances (--resume skips done).
#   2. Rebuild the distill corpus + in-distribution holdout from ALL resolved trajectories.
#   3. Retrain Ternary-Bonsai-8B at the fixed iter-5 sweet spot (5e-4, ~2.2 epochs, all-36),
#      export Q2_0, and dual-bench: generalization (disjoint 10-holdout) + in-distribution.
#   4. STOP if generalization pass_rate > 0 OR patch_rate >= 0.60 (a genuinely better model);
#      else loop and gather more data.
#
# Safety: MAX_ROUNDS cap; stops if no fresh instances can be sourced. Everything is logged;
# each round's artifacts are tagged iter5-rN so nothing collides. Kill the process to stop.
set -u
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
PY=.venv/bin/python

MAX_ROUNDS=6
BATCH=150                       # fresh instances to add per expansion round
TARGET_PATCH=0.60               # stop if generalization patch >= this ...
# (stop also if generalization pass_rate > 0 -- checked in python below)
POOL=out/external/swe-rebench/distill_train.jsonl
EVAL=out/external/swe-rebench/holdout.jsonl
ALL_LOCAL=out/external/swe-rebench/all_test.jsonl  # full split on disk (no datasets-server throttle)
GEN_WS=out/swe-rebench/ornith-distill-gen
TRAJ="$GEN_WS/trajectories/Ornith-1.0-9B-Q5_K_M"
ORNITH=uploads/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF/Ornith-1.0-9B-Q5_K_M.gguf

resolved_count() {
  $PY -c "import json,glob; print(sum(1 for f in glob.glob('$TRAJ/*.result.json') if json.load(open(f)).get('resolved')))"
}

log() { echo "=== [$(date '+%m-%d %H:%M:%S')] $*"; }

log "AUTO-LOOP start. Waiting for any in-flight generation to finish..."
while pgrep -f run_swebench_eval >/dev/null; do sleep 120; done
sleep 15

# Ensure the FULL split is on disk (one-time) so fresh-instance sourcing is a local read,
# never the rate-limited datasets-server.
if [ ! -s "$ALL_LOCAL" ]; then
  log "downloading full nebius/SWE-rebench test split to disk (one-time)..."
  $PY scripts/download_swebench_dataset.py --out "$ALL_LOCAL" || { log "dataset download failed"; exit 1; }
fi
log "local pool: $(grep -c . "$ALL_LOCAL") gradeable instances available"

round=0
while [ $round -lt $MAX_ROUNDS ]; do
  round=$((round + 1))
  TAG="iter5-r${round}"
  CORPUS="out/exp-058/distill_corpus_${TAG}.pt"
  INDIST="out/external/swe-rebench/indist_${TAG}.jsonl"
  TRAIN_OUT="out/exp-058/trained_${TAG}"
  GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"
  G_WS="out/swe-rebench/ternary-${TAG}-swe"
  I_WS="out/swe-rebench/ternary-${TAG}-indist-swe"

  # Resume: if this round's dual-eval already exists (e.g. a prior loop run), skip straight
  # to the verdict instead of re-training/re-evaluating.
  if [ -f "$G_WS/summary.json" ] && [ -f "$I_WS/summary.json" ]; then
    R=$(resolved_count)
    log "round $round ($TAG): already evaluated ($R traj) -> reusing results"
  else
  # ---- 1. expand data (rounds after the first) ------------------------------------------
  if [ $round -gt 1 ]; then
    log "round $round: sourcing $BATCH fresh instances from the local split (no throttling)"
    cat "$EVAL" "$POOL" > /tmp/exclude_all.jsonl
    if ! $PY scripts/build_swebench_holdout.py --n $BATCH --from-local "$ALL_LOCAL" \
         --exclude /tmp/exclude_all.jsonl --out /tmp/fresh_batch.jsonl; then
      log "round $round: fresh-batch build failed; stopping loop"; break
    fi
    nfresh=$(grep -c . /tmp/fresh_batch.jsonl || echo 0)
    if [ "$nfresh" -eq 0 ]; then log "round $round: no fresh instances left in the split; stopping loop"; break; fi
    cat /tmp/fresh_batch.jsonl >> "$POOL"
    log "round $round: added $nfresh fresh instances (pool now $(grep -c . "$POOL"))"
    log "round $round: generating Ornith trajectories over the new instances"
    $PY -u scripts/run_swebench_eval.py --models "$ORNITH" --holdout "$POOL" \
      --workspace "$GEN_WS" --agent openai-agents --temperature 0.25 --max-steps 100 \
      --resume --cleanup-images --progress || log "round $round: generation returned nonzero (continuing with what resolved)"
    docker image prune -f >/dev/null 2>&1 || true
  fi

  R=$(resolved_count)
  log "round $round ($TAG): $R resolved trajectories -> rebuild + train + eval"

  # ---- 2. rebuild corpus + in-dist holdout ----------------------------------------------
  $PY -u scripts/build_ornith_distill_corpus.py --traj-dir "$TRAJ" \
    --max-tool-tokens 1024 --out "$CORPUS" || { log "corpus build failed"; break; }
  $PY -c "
import json, glob, os
resolved={os.path.basename(f)[:-len('.result.json')] for f in glob.glob('$TRAJ/*.result.json') if json.load(open(f)).get('resolved')}
rows=[json.loads(l) for l in open('$POOL')]
picked=[r for r in rows if r['instance_id'] in resolved]
open('$INDIST','w').write(''.join(json.dumps(r)+'\n' for r in picked))
gen={json.loads(l)['instance_id'] for l in open('$EVAL')}
assert not (resolved & gen), 'in-dist overlaps generalization holdout!'
print('in-dist instances:', len(picked))
" || { log "in-dist build failed"; break; }

  # ---- 3. train -> export -> dual-bench --------------------------------------------------
  $PY -u scripts/exp058_qat_train_v2.py --corpus "$CORPUS" --layers 0-35 --optim adafactor \
    --epochs 2.2 --grad-accum 2 --lr 5e-4 --dtype fp32 --ckpt-every 20 --flip-sample 12 \
    --out "$TRAIN_OUT" || { log "train failed"; break; }
  $PY -u scripts/exp057_qat_export.py --latents "$TRAIN_OUT/trained_latents.pt" --tag "$TAG" \
    || { log "export failed"; break; }
  $PY -u scripts/run_swebench_eval.py --models "$GGUF" --holdout "$EVAL" --workspace "$G_WS" \
    --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images --progress
  $PY -u scripts/run_swebench_eval.py --models "$GGUF" --holdout "$INDIST" --workspace "$I_WS" \
    --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images --progress
  docker image prune -f >/dev/null 2>&1 || true
  fi  # end resume-guard (skip train/eval if this round was already evaluated)

  # ---- 4. verdict ------------------------------------------------------------------------
  log "round $round ($TAG, $R traj) RESULTS:"
  $PY -c "
import json
def row(p,label):
    m=list(json.load(open(p))['models'].values())[0]['aggregate']
    print(f'  {label:16s} patch={m[\"patch_rate\"]:.2f} pass={m[\"pass_rate\"]:.2f} resolved={m[\"n_resolved\"]}/{m[\"n_instances\"]} tool_err={m[\"tool_error_rate\"]:.2f} steps={m[\"mean_steps\"]:.1f}')
row('$G_WS/summary.json','generalization')
row('$I_WS/summary.json','in-distribution')
"
  better=$($PY -c "
import json
m=list(json.load(open('$G_WS/summary.json'))['models'].values())[0]['aggregate']
print('YES' if (m['pass_rate']>0 or m['patch_rate']>=$TARGET_PATCH) else 'NO')
")
  if [ "$better" = "YES" ]; then
    log "🎉 BETTER MODEL at round $round ($R trajectories). GGUF: $GGUF. Stopping loop."
    break
  fi
  log "round $round not yet better (need generalization pass>0 or patch>=$TARGET_PATCH); looping for more data"
done
log "AUTO-LOOP done (round $round of max $MAX_ROUNDS)."

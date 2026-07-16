#!/bin/bash
# iter-5b: SCALE UP the verified-trajectory set, same recipe. Waits for the --resume
# generation retry to finish, rebuilds the distill corpus + in-dist holdout from ALL resolved
# trajectories (now >12), retrains at the iter-5 sweet spot (5e-4, ~2 epochs, all-36), and
# dual-benches: generalization (disjoint 10-holdout) + in-distribution (the trained set).
#
# The bet: iter-5's in-distribution learning (first pass>0) turns into GENERALIZATION once
# there are 50-100 verified trajectories instead of 12. Recipe is fixed; only data grows.
set -e
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_DIR=vendor/llama.cpp-prism
PY=.venv/bin/python
TAG=iter5b
GEN=out/swe-rebench/ornith-distill-gen
TRAJ="$GEN/trajectories/Ornith-1.0-9B-Q5_K_M"
CORPUS=out/exp-058/distill_corpus_${TAG}_4096.pt
INDIST=out/external/swe-rebench/indist_${TAG}.jsonl
TRAIN_OUT=out/exp-058/trained_${TAG}
GGUF=out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf
GEN_WS=out/swe-rebench/ternary-${TAG}-swe
INDIST_WS=out/swe-rebench/ternary-${TAG}-indist-swe

echo "=== [$(date)] iter-5b waiting for the --resume generation retry to finish ==="
while pgrep -f run_swebench_eval >/dev/null; do sleep 120; done
sleep 20
N=$(ls "$TRAJ"/*.result.json 2>/dev/null | wc -l | tr -d ' ')
R=$($PY -c "import json,glob; print(sum(1 for f in glob.glob('$TRAJ/*.result.json') if json.load(open(f)).get('resolved')))")
echo "=== [$(date)] generation done: $N graded, $R resolved verified trajectories ==="

echo "=== [$(date)] rebuild distill corpus from ALL $R resolved trajectories ==="
$PY -u scripts/build_ornith_distill_corpus.py \
  --traj-dir "$TRAJ" --max-tool-tokens 1024 --out "$CORPUS"

echo "=== [$(date)] rebuild in-distribution holdout (all resolved instances) ==="
$PY -c "
import json, glob, os
resolved={os.path.basename(f)[:-len('.result.json')] for f in glob.glob('$TRAJ/*.result.json') if json.load(open(f)).get('resolved')}
rows=[json.loads(l) for l in open('out/external/swe-rebench/distill_train.jsonl')]
picked=[r for r in rows if r['instance_id'] in resolved]
open('$INDIST','w').write(''.join(json.dumps(r)+'\n' for r in picked))
# guard: must stay disjoint from the generalization holdout
gen={json.loads(l)['instance_id'] for l in open('out/external/swe-rebench/holdout.jsonl')}
assert not (resolved & gen), 'in-dist overlaps generalization holdout!'
print(f'wrote {len(picked)} in-dist instances -> $INDIST (disjoint from gen holdout: ok)')
"

echo "=== [$(date)] iter-5b TRAIN (all-36, adafactor, lr 5e-4, ~2.2 epochs) ==="
$PY -u scripts/exp058_qat_train_v2.py \
  --corpus "$CORPUS" --layers 0-35 --optim adafactor --epochs 2.2 --grad-accum 2 --lr 5e-4 \
  --dtype fp32 --ckpt-every 20 --flip-sample 12 --out "$TRAIN_OUT"

echo "=== [$(date)] iter-5b EXPORT -> Q2_0 (prism) ==="
$PY -u scripts/exp057_qat_export.py --latents "$TRAIN_OUT/trained_latents.pt" --tag "$TAG"

echo "=== [$(date)] iter-5b GENERALIZATION eval (disjoint 10-holdout) ==="
$PY -u scripts/run_swebench_eval.py --models "$GGUF" \
  --holdout out/external/swe-rebench/holdout.jsonl --workspace "$GEN_WS" \
  --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images --progress

echo "=== [$(date)] iter-5b IN-DISTRIBUTION eval (trained instances) ==="
$PY -u scripts/run_swebench_eval.py --models "$GGUF" \
  --holdout "$INDIST" --workspace "$INDIST_WS" \
  --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images --progress

echo "=== [$(date)] iter-5b DUAL-EVAL SUMMARY ($R trajectories) ==="
$PY -c "
import json
def row(p,label):
    m=list(json.load(open(p))['models'].values())[0]['aggregate']
    print(f'  {label:16s} patch={m[\"patch_rate\"]:.2f} pass={m[\"pass_rate\"]:.2f} '
          f'resolved={m[\"n_resolved\"]}/{m[\"n_instances\"]} tool_err={m[\"tool_error_rate\"]:.2f} steps={m[\"mean_steps\"]:.1f}')
row('$GEN_WS/summary.json','generalization')
row('$INDIST_WS/summary.json','in-distribution')
print('  (compare iter-5 @12 traj: generalization 40%/0%, in-dist 25%/8%)')
"

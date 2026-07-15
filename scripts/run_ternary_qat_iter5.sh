#!/bin/bash
# iter-5: distill VERIFIED SWE-rebench solutions from Ornith-9B into Ternary-Bonsai-8B.
#
# The data-quality lever the iter-4 verdict + the optimization audit both point at. Unlike
# iter-2/3/4 (scraped Claude-Code/Gemini-CLI logs = imitation data), this trains on Ornith's
# graded-PASS agentic trajectories re-rendered through Bonsai's tokenizer, on the CORRECTED
# corpus (stop token labeled, real bash schema). Plain masked-CE; same-vocab logit-KD from a
# SWE-tuned Qwen3-8B teacher is deferred to iter-6 (see project_iter6_kd_teachers memory).
#
# Run AFTER scripts/run_ornith_distill_gen.sh finishes harvesting resolved trajectories.
# Two phases: (1) build the corpus, (2) a 3-point LR PROBE — the audit's mandated first step,
# because at lr 5e-5 the flip math predicts ~zero code flips (the iter-2/3 loss drop was
# likely pure scale drift). Read the "flips X%" telemetry from each probe, pick the LR that
# actually flips codes without diverging, then launch the full run (command at the bottom).
cd /Users/kpearson/Programs/ai/llm/quant-tuner
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
GEN=out/swe-rebench/ornith-distill-gen
CORPUS=out/exp-058/distill_corpus_4096.pt

# --- phase 1: build the distill corpus from RESOLVED trajectories ------------------------
echo "=== [$(date)] iter-5 build distill corpus ==="
$PY -u scripts/build_ornith_distill_corpus.py \
  --traj-dir "$GEN/trajectories/Ornith-1.0-9B-Q5_K_M" \
  --results "$GEN/results.csv" \
  --max-tool-tokens 1024 \
  --out "$CORPUS" || exit 1

# --- phase 2: 3-point LR probe (short runs; watch the flip telemetry) --------------------
# all-36 layers, adafactor, fp32; ~1 short pass with a checkpoint at step 20 so flip_report
# fires. We want the smallest LR whose "flips" telemetry is clearly non-zero and whose loss
# is stable (not NaN-skipping). 5e-5 is the old default that flipped ~nothing.
for LR in 5e-5 3e-4 1e-3; do
  echo "=== [$(date)] iter-5 LR PROBE lr=$LR ==="
  $PY -u scripts/exp058_qat_train_v2.py \
    --corpus "$CORPUS" \
    --layers 0-35 --optim adafactor --epochs 0.4 --grad-accum 4 --lr "$LR" \
    --dtype fp32 --ckpt-every 20 --flip-sample 12 \
    --out "out/exp-058/probe_iter5_lr${LR}" 2>&1 | tee "out/exp-058/iter5_probe_lr${LR}.log"
done

echo "=== [$(date)] iter-5 LR probe DONE — inspect 'flips %' in the logs, then run the full pass ==="
cat <<'NEXT'
# --- phase 3: full run (edit LR to the probe winner, scale epochs to the corpus size) ---
#   $PY -u scripts/exp058_qat_train_v2.py \
#     --corpus out/exp-058/distill_corpus_4096.pt \
#     --layers 0-35 --optim adafactor --epochs 3 --grad-accum 4 --lr <PROBE_WINNER> \
#     --dtype fp32 --ckpt-every 40 --flip-sample 12 \
#     --out out/exp-058/trained_iter5
# --- then export (writes out/exp-057/Ternary-Bonsai-8B-iter5-Q2_0.gguf) + eval ---
#   $PY -u scripts/exp057_qat_export.py --latents out/exp-058/trained_iter5/trained_latents.pt \
#     --tag iter5
#   $PY -u scripts/run_swebench_eval.py \
#     --models out/exp-057/Ternary-Bonsai-8B-iter5-Q2_0.gguf \
#     --holdout out/external/swe-rebench/holdout.jsonl \
#     --workspace out/swe-rebench/ternary-iter5-swe --agent openai-agents --temperature 0.25 \
#     --max-steps 100 --resume --progress
NEXT

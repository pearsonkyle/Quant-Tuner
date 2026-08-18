#!/usr/bin/env bash
# One tagged checkpoint -> Q2_0 GGUF -> GGUF stop probe -> in-distribution stop
# measurement -> Docker-free SWE mimic -> prune the ~50 GB of export intermediates.
#
#     bash scripts/run_kd_export_bench.sh anchor3
#     bash scripts/run_kd_export_bench.sh anchor4 out/exp-058/kd8b-full-anchor4/trained_latents.pt
#
# Read the three measurements TOGETHER (each is blind somewhere): the GGUF probe is the
# serving-numerics endpoint of the in-training series; measure_indist_stop is real
# corpus stop positions (catches a bimodal weak tail the probe's medians hide); the
# mimic episode is ground truth for "does it patch and stop".
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${1:?usage: run_kd_export_bench.sh TAG [latents.pt]}"
CK="${2:-out/exp-058/kd8b-full-${TAG}/trained_latents.pt}"
[ -f "$CK" ] || { echo "[bench] no checkpoint at $CK"; exit 1; }
export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-vendor/llama.cpp-prism}"

echo "[bench] 1/4 export $CK"
PYTHONPATH=src .venv/bin/python scripts/exp057_qat_export.py \
    --latents "$CK" --tag "$TAG" 2>&1 | grep -vE "it/s\]|%\|" | tail -3
GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"
[ -f "$GGUF" ] || { echo "[bench] export produced no $GGUF"; exit 1; }

echo "[bench] 2/4 GGUF stop probe"
PYTHONPATH=src .venv/bin/python scripts/probe_stop_prob.py \
    --model "$GGUF" --label "${TAG}-Q2_0" 2>&1 | tail -7

echo "[bench] 3/4 in-distribution stop measurement"
PYTHONPATH=src .venv/bin/python scripts/measure_indist_stop.py \
    --corpus out/exp-058/fixed/corpus_ourssft_val_32768.pt \
    --latents "$CK" --windows 8 2>&1 | grep -vE "it/s\]|%\|" | grep indist

echo "[bench] 4/4 SWE mimic"
bash scripts/run_swe_mimic.sh "$TAG" "$GGUF" 2>&1 | tail -12
bash scripts/prune_export_intermediates.sh "$TAG" 2>/dev/null | tail -1

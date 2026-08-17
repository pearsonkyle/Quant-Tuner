#!/usr/bin/env bash
# Everything that has to happen after the sft32k_sw1 run ends, in dependency order.
#
#     bash scripts/run_sw1_postflight.sh [RUN_DIR] [TAG]
#
# Cheap, reversible, and CPU-only apart from the export, so it is safe to fire
# automatically when the trainer exits. It deliberately stops BEFORE launching the
# curriculum: that is a ~33 h job whose stop-weight depends on the answer this postflight
# produces, so it stays a human decision.
#
# Order matters. The P(im_end) probe is the primary endpoint and the cheapest, so it runs
# first — if termination is still broken there is no reason to spend an hour on tool-call
# reps before saying so.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-out/exp-058/trained_sft32k_sw1}"
TAG="${2:-sft32k_sw1}"
PY=.venv/bin/python
GGUF="out/exp-057/Ternary-Bonsai-8B-${TAG}-Q2_0.gguf"
STOP_CSV="out/exp-058/eval/stop_prob.csv"

[ -f "${RUN}/trained_latents.pt" ] || {
    echo "[postflight] no trained_latents.pt in ${RUN} — the run did not finish"; exit 1; }

echo "[postflight] 1/4 export -> Q2_0 (ftype 41 is fork-only)"
if [ ! -f "$GGUF" ]; then
    LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY scripts/exp057_qat_export.py \
        --latents "${RUN}/trained_latents.pt" --tag "$TAG" \
        2>&1 | tee "out/exp-058/export_${TAG}.log" | tail -3
else
    echo "  already exported: $GGUF"
fi
[ -f "$GGUF" ] || { echo "[postflight] export produced no GGUF"; exit 1; }

# The primary endpoint. CPU (--ngl 0) so it never contends with anything on the card.
echo "[postflight] 2/4 P(im_end) probe — the primary endpoint"
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY scripts/probe_stop_prob.py \
    --model "$GGUF" --label "$TAG" --out "$STOP_CSV" \
    --json-out "out/exp-058/eval/stop_prob_${TAG}.json" --ngl 0

echo "[postflight] 3/4 tool-call eval (n is tiny — read it with the probe, not alone)"
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY scripts/eval_toolcall.py \
    --model "$GGUF" \
    --holdout out/exp-060-32k/eval/toolcall_holdout.jsonl \
    --out "out/exp-058/eval/toolcall_${TAG}.csv" \
    --temperature 0 --seed 1234 --ctx 8192 --ngl 99 2>&1 | tail -12 || \
    echo "[postflight] tool-call eval failed — not fatal, the probe is the endpoint"

echo "[postflight] 4/4 refresh the report (now carries the ${TAG} probe column)"
bash scripts/qat_report_refresh.sh "$RUN" "${TAG} — ternary QAT (complete)" || true

echo
echo "[postflight] done. Termination endpoint:"
$PY - <<PY
import csv, pathlib
p = pathlib.Path("$STOP_CSV")
rows = [r for r in csv.DictReader(p.open())] if p.exists() else []
probes = list(dict.fromkeys(r["probe"] for r in rows))
labels = list(dict.fromkeys(r["label"] for r in rows))
w = max((len(x) for x in labels), default=8) + 2
print("  " + "probe".ljust(18) + "".join(l.ljust(w) for l in labels))
for pr in probes:
    cells = []
    for l in labels:
        m = next((r for r in rows if r["label"] == l and r["probe"] == pr), None)
        if not m: cells.append("-".ljust(w)); continue
        v = m["stop_prob"]
        cells.append((f"{float(v):.5f}" if v else "<tail").ljust(w))
    print("  " + pr.ljust(18) + "".join(cells))
print()
print("  sentence_period is the diagnostic: vanilla 0.0092, sft32k(6.0) 0.9743.")
print("  Near vanilla => the 32K window alone fixed termination.")
print("  Near sft32k  => try --stop-weight 2.0 before adding data (docs/ternary_qat_curriculum.md).")
PY
echo
echo "The curriculum is NOT started automatically. When you have read the numbers above:"
echo "  bash scripts/run_curriculum_qat.sh curriculum 5e-4 1.0"

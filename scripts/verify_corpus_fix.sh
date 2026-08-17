#!/usr/bin/env bash
# Does the corpus fix actually prevent the termination collapse? A/B, ~60 steps each.
#
#     bash scripts/verify_corpus_fix.sh [STEPS]
#
# Two short runs from the SAME shipped weights with identical hyper-parameters, differing
# only in the corpus:
#
#   old  out/exp-058/corpus_ourssft_32768.pt        (split assistant turns: 18.5% of
#                                                    "Let me..." turns end at sentence 1)
#   new  out/exp-058/fixed/corpus_ourssft_32768.pt  (merged: 0.0%)
#
# The endpoint is the in-training stop probe, not the loss. P(<|im_end|> | completed
# sentence) starts at ~0.002 on the shipped weights and reached 0.95 in every run trained
# on the old corpus; if the fix works, the new run holds near its starting value while the
# old one climbs. Masked-CE cannot decide this — sft32k's validation was flat for 225
# steps while exactly this collapse happened.
#
# ~60 steps at ~45 s/step is ~45 min per arm. That is the price of not discovering the
# answer 23 h into the curriculum.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-60}"
PY=.venv/bin/python
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OLD="${OLD:-out/exp-058/corpus_ourssft_32768.pt}"
NEW="${NEW:-out/exp-058/fixed/corpus_ourssft_32768.pt}"
VAL_OLD="${VAL_OLD:-out/exp-058/corpus_ourssft_val_32768.pt}"
VAL_NEW="${VAL_NEW:-out/exp-058/fixed/corpus_ourssft_val_32768.pt}"

arm() {  # tag corpus valcorpus
    local tag="$1" corpus="$2" val="$3"
    local out="out/exp-058/verify-${tag}"
    if [ -f "${out}/.done" ]; then echo "[verify] ${tag} already run"; return 0; fi
    [ -f "$corpus" ] || { echo "[verify] missing corpus $corpus"; return 1; }
    local nwin
    nwin=$($PY -c "import torch;print(torch.load('$corpus',weights_only=False)['ids'].shape[0])")
    # epochs chosen so the run is exactly STEPS optimizer steps at grad-accum 1
    local ep
    ep=$($PY -c "print(f'{$STEPS/$nwin:.6f}')")
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${busy:-0}" -gt 2048 ] && { echo "[verify] GPU busy (${busy} MiB)"; return 1; }

    mkdir -p "$out"
    echo "[verify] ${tag}: ${nwin} windows -> ${STEPS} steps (epochs ${ep})"
    $PY -m quant_tuner.qat.train \
        --corpus "$corpus" --val-corpus "$val" \
        --train-layers 36 --optim adafactor --dtype fp32 \
        --compute-dtype fp32 --matmul-precision high \
        --grad-accum 1 --epochs "$ep" --lr 5e-4 --warmup-frac 0.05 \
        --stop-weight 1.0 --grad-spike-factor 0 \
        --val-every 0 --probe-every 10 \
        --ckpt-every 100000 --ckpt-keep 1 \
        --out "$out" > "${out}/train.log" 2>&1
    local rc=$?
    touch "${out}/.done"
    # The final checkpoint is 27.8 GB and this run exists only for its probe series.
    rm -f "${out}"/trained_latents*.pt
    echo "[verify] ${tag} finished rc=$rc"
}

arm old "$OLD" "$VAL_OLD"
arm new "$NEW" "$VAL_NEW"

echo
echo "=============== TERMINATION PROBE: old corpus vs fixed corpus ==============="
$PY - <<'PYEOF'
import pathlib, re
STEP = re.compile(r"^\[qat\] step (\d+) STOPPROBE (.*)$")
KV = re.compile(r"([a-z_]+)=([\d.eE+-]+)")
series = {}
for tag in ("old", "new"):
    p = pathlib.Path(f"out/exp-058/verify-{tag}/train.log")
    rows = []
    if p.exists():
        for line in p.read_text(errors="replace").replace("\r", "\n").splitlines():
            m = STEP.match(line)
            if m:
                body = m.group(2).split("[")[0]
                rows.append((int(m.group(1)),
                             {k: float(v) for k, v in KV.findall(body)}))
    series[tag] = rows
steps = sorted({s for r in series.values() for s, _ in r})
print(f"{'step':>6}{'OLD sentence_period':>22}{'NEW sentence_period':>22}"
      f"{'OLD after_tool':>17}{'NEW after_tool':>17}")
print("-" * 84)
for s in steps:
    cells = []
    for probe in ("sentence_period", "after_tool_call"):
        for tag in ("old", "new"):
            v = next((d.get(probe) for st, d in series[tag] if st == s), None)
            cells.append(f"{v:.5f}" if v is not None else "-")
    print(f"{s:>6}{cells[0]:>22}{cells[1]:>22}{cells[2]:>17}{cells[3]:>17}")
print()
print("  shipped weights (torch fp32): sentence_period 0.0017   after_tool_call 0.99996")
print("  every run on the OLD corpus ended at        0.95        and              0.81")
print()
print("  Fix works if NEW stays near 0.002 while OLD climbs.")
print("  If BOTH climb, the corpus was not sufficient and lr is the next lever.")
PYEOF

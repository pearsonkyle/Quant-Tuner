#!/usr/bin/env bash
# Does the optimizer drive the termination drift? N steps on the FIXED corpus, one arm
# per optimizer, compared on the in-training stop probe.
#
#     bash scripts/verify_optimizer.sh adamw8bit [STEPS]
#     bash scripts/verify_optimizer.sh lion8bit 60
#
# The reference arm already exists: `verify-new` in verify_corpus_fix.sh is the same
# corpus, same lr, same everything, with adafactor and NO MOMENTUM (beta1 defaults to
# None). Its series is
#     step 10 0.0043 · 20 0.0053 · 30 0.0095 · 40 0.0092 · 50 0.0102
# against the shipped model's 0.0017.
#
# WHY THIS IS A FAIR COMPARISON AT A FIXED LR. Adafactor here runs with
# scale_parameter=False and relative_step=False, which makes it "Adam with a rank-1
# second moment and no momentum" — both optimizers normalize the step by a gradient RMS
# estimate, so holding lr constant holds the effective step size roughly constant and the
# arms differ in the two things we want to test: MOMENTUM and full-vs-factored second
# moment. That is not true of an optimizer swap in general (SGD at 5e-4 is a different
# universe), so do not reuse this harness for a non-RMS-normalized optimizer without
# rescaling lr.
#
# THE HYPOTHESIS. A ternary latent only changes anything when it crosses the ternarization
# threshold, and crossing needs pressure accumulated over many steps. Without momentum a
# latent near the threshold jitters on instantaneous gradients, while a coarse signal
# present in nearly every batch ("a sentence end is often followed by <|im_end|>") is
# reinforced every step regardless. That would preferentially learn the always-on coarse
# rule and fail to learn the fine context-dependent one — which is the observed failure.
set -uo pipefail
cd "$(dirname "$0")/.."

OPTIM="${1:?usage: verify_optimizer.sh OPTIM [STEPS]}"
STEPS="${2:-60}"
LR="${LR:-5e-4}"
BETA1="${BETA1:-0.9}"
PY=.venv/bin/python
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CORPUS="${CORPUS:-out/exp-058/fixed/corpus_ourssft_32768.pt}"
VAL="${VAL:-out/exp-058/fixed/corpus_ourssft_val_32768.pt}"
OUT="out/exp-058/verify-opt-${OPTIM}"

[ -f "${OUT}/.done" ] && { echo "[verify-opt] ${OPTIM} already run"; }
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${busy:-0}" -gt 2048 ] && { echo "[verify-opt] GPU busy (${busy} MiB)"; exit 1; }

nwin=$($PY -c "import torch;print(torch.load('$CORPUS',weights_only=False)['ids'].shape[0])")
ep=$($PY -c "print(f'{$STEPS/$nwin:.6f}')")

beta_args=()
case "$OPTIM" in
    adafactor|adamw8bit|lion8bit) beta_args=(--beta1 "$BETA1") ;;
esac

mkdir -p "$OUT"
echo "[verify-opt] ${OPTIM} lr=${LR} beta1=${BETA1} -> ${STEPS} steps (epochs ${ep})"
$PY -m quant_tuner.qat.train \
    --corpus "$CORPUS" --val-corpus "$VAL" \
    --train-layers 36 --optim "$OPTIM" --dtype fp32 \
    --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs "$ep" --lr "$LR" --warmup-frac 0.05 \
    "${beta_args[@]}" \
    --stop-weight 1.0 --grad-spike-factor 0 \
    --val-every 0 --probe-every 10 \
    --ckpt-every 100000 --ckpt-keep 1 \
    --out "$OUT" > "${OUT}/train.log" 2>&1
rc=$?
touch "${OUT}/.done"
rm -f "${OUT}"/trained_latents*.pt      # this run exists only for its probe series
echo "[verify-opt] ${OPTIM} finished rc=$rc"

echo
echo "=============== TERMINATION PROBE by optimizer ==============="
$PY - <<'PYEOF'
import pathlib, re
STEP = re.compile(r"^\[qat\] step (\d+) STOPPROBE (.*)$")
KV = re.compile(r"([a-z_]+)=([\d.eE+-]+)")
arms = {"adafactor (no momentum)": "out/exp-058/verify-new"}
for d in sorted(pathlib.Path("out/exp-058").glob("verify-opt-*")):
    arms[d.name.replace("verify-opt-", "")] = str(d)
series = {}
for name, d in arms.items():
    p = pathlib.Path(d) / "train.log"
    rows = []
    if p.exists():
        for line in p.read_text(errors="replace").replace("\r", "\n").splitlines():
            m = STEP.match(line)
            if m:
                body = m.group(2).split("[")[0]
                rows.append((int(m.group(1)), {k: float(v) for k, v in KV.findall(body)}))
    series[name] = rows
steps = sorted({s for r in series.values() for s, _ in r})
names = list(series)
w = max((len(n) for n in names), default=10) + 2
print("sentence_period (the diagnostic; shipped weights = 0.0017)")
print("  step " + "".join(n.ljust(w) for n in names))
for s in steps:
    cells = []
    for n in names:
        v = next((d.get("sentence_period") for st, d in series[n] if st == s), None)
        cells.append((f"{v:.5f}" if v is not None else "-").ljust(w))
    print(f"  {s:>4} " + "".join(cells))
print()
print("after_tool_call (the CONTROL; shipped weights = 0.99996, high is CORRECT)")
print("  step " + "".join(n.ljust(w) for n in names))
for s in steps:
    cells = []
    for n in names:
        v = next((d.get("after_tool_call") for st, d in series[n] if st == s), None)
        cells.append((f"{v:.5f}" if v is not None else "-").ljust(w))
    print(f"  {s:>4} " + "".join(cells))
PYEOF

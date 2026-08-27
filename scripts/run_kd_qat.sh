#!/usr/bin/env bash
# Offline-KD training: precompute a teacher's top-K distribution over a corpus, then train
# the ternary student against CE + KL.
#
#     bash scripts/run_kd_qat.sh                                   # 8B teacher, 60-step A/B
#     STEPS=613 bash scripts/run_kd_qat.sh                         # full run
#     TEACHER=SWE-Lego/SWE-Lego-Qwen3-32B TAG=kd32b bash scripts/run_kd_qat.sh
#
# WHY KD AT ALL. Four levers were tested against the termination collapse and all four
# failed or traded against learning (docs/ternary_qat_curriculum.md):
#   --stop-weight   a 6x change moved the diagnostic by 0.02
#   the corpus      a real defect, worth 3-6x, but a structurally clean corpus still drifts
#   the optimizer   momentum makes it 3.2x WORSE
#   the lr          5e-4 learns and breaks termination; 2.5e-4 preserves it and learns nothing
# Hard CE gives one target per position and says nothing about the SHAPE of the
# distribution, so the model is free to collapse P(stop) anywhere the argmax survives. The
# KL term constrains exactly that, while leaving the argmax free to move — it attacks the
# mechanism instead of moving along the trade.
#
# TEACHER REQUIREMENT — THE ONE THAT SILENTLY RUINS A RUN. Per-token KD needs the teacher
# and student to share a tokenizer id->string map. `kd_precompute.tokenizer_compatibility`
# refuses a mismatch, which is why it must not be bypassed:
#   SWE-Lego/SWE-Lego-Qwen3-8B    OK   agrees on all 151,669 ids (hidden 4096, 36 layers)
#   SWE-Lego/SWE-Lego-Qwen3-32B   OK   agrees on all 151,669 ids (hidden 5120, 64 layers)
#   Qwen/Qwen3.8-27B              NO   vocab 248,320, a different tokenizer family
# The 27B solves our SWE instance at IQ2_M and is the obvious teacher to reach for; it is
# the one that cannot be used.
set -uo pipefail
cd "$(dirname "$0")/.."

TEACHER="${TEACHER:-SWE-Lego/SWE-Lego-Qwen3-8B}"
TAG="${TAG:-kd8b}"
STEPS="${STEPS:-60}"
LR="${LR:-5e-4}"
ALPHA="${ALPHA:-0.5}"
TEMP="${TEMP:-1.0}"
TOPK="${TOPK:-64}"
PY=.venv/bin/python
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CORPUS="${CORPUS:-out/exp-058/fixed/corpus_ourssft_32768.pt}"
VAL="${VAL:-out/exp-058/fixed/corpus_ourssft_val_32768.pt}"
TABLE="${TABLE:-out/exp-058/kd/$(basename "$TEACHER")_topk${TOPK}.pt}"
OUT="out/exp-058/verify-opt-${TAG}-a${ALPHA}"

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${busy:-0}" -gt 2048 ] && { echo "[kd] GPU busy (${busy} MiB)"; exit 1; }

# ---- 1. precompute (idempotent; ~2.8 s/window for the 8B) ---------------------------
if [ ! -f "$TABLE" ]; then
    echo "[kd] precomputing $TEACHER top-$TOPK over $(basename "$CORPUS")"
    mkdir -p "$(dirname "$TABLE")"
    $PY scripts/kd_precompute.py --teacher "$TEACHER" --corpus "$CORPUS" \
        --out "$TABLE" --topk "$TOPK" --dtype bf16 \
        --student-model out/exp-057/model \
        --chat-template out/exp-057/chat_template.jinja 2>&1 | grep -vE "it/s\]|%\|" | tail -6
    [ -f "$TABLE" ] || { echo "[kd] precompute produced no table"; exit 1; }
else
    echo "[kd] reusing table $TABLE"
fi

# ---- 2. train ------------------------------------------------------------------------
nwin=$($PY -c "import torch;print(torch.load('$CORPUS',weights_only=False)['ids'].shape[0])")
ep=$($PY -c "print(f'{$STEPS/$nwin:.6f}')")
mkdir -p "$OUT"
echo "[kd] training tag=$TAG lr=$LR alpha=$ALPHA T=$TEMP -> $STEPS steps (epochs $ep)"
$PY -m quant_tuner.qat.train \
    --corpus "$CORPUS" --val-corpus "$VAL" \
    --kd-table "$TABLE" --kd-alpha "$ALPHA" --kd-temp "$TEMP" \
    --train-layers 36 --optim adafactor --dtype fp32 \
    --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs "$ep" --lr "$LR" --warmup-frac 0.05 \
    --stop-weight 1.0 --grad-spike-factor 0 \
    --val-every 0 --probe-every 10 \
    --ckpt-every "${CKPT_EVERY:-100000}" --ckpt-keep 1 \
    --out "$OUT" > "${OUT}/train.log" 2>&1
rc=$?
echo "[kd] training finished rc=$rc"
# A short A/B run exists only for its probe series; a full run's latents are the artifact.
[ "$STEPS" -lt 200 ] && rm -f "${OUT}"/trained_latents*.pt

echo
echo "=============== TERMINATION PROBE: KD vs the non-KD arms ==============="
$PY - <<'PYEOF'
import pathlib, re
STEP = re.compile(r"^\[qat\] step (\d+) STOPPROBE (.*)$")
KV = re.compile(r"([a-z_]+)=([\d.eE+-]+)")
arms = {"CE only lr5e-4": "out/exp-058/verify-new",
        "CE only lr2.5e-4": "out/exp-058/verify-opt-adafactor-lr2.5e-4"}
for d in sorted(pathlib.Path("out/exp-058").glob("verify-opt-kd*")):
    arms[d.name.replace("verify-opt-", "")] = str(d)
series, kls = {}, {}
for name, d in arms.items():
    p = pathlib.Path(d) / "train.log"
    rows = []
    if p.exists():
        txt = p.read_text(errors="replace").replace("\r", "\n")
        for line in txt.splitlines():
            m = STEP.match(line)
            if m:
                rows.append((int(m.group(1)),
                             {k: float(v) for k, v in KV.findall(m.group(2).split("[")[0])}))
        kl = re.findall(r"kl=([\d.]+)", txt)
        kls[name] = (kl[0], kl[-1]) if kl else None
    series[name] = rows
names = list(series)
w = max((len(n) for n in names), default=12) + 2
for probe, gloss in (("sentence_period", "the diagnostic; shipped = 0.0017, broken = 0.95"),
                     ("after_tool_call", "the CONTROL; shipped = 0.99996, high is CORRECT")):
    print(f"{probe}  ({gloss})")
    print("  step " + "".join(n.ljust(w) for n in names))
    for s in sorted({s for r in series.values() for s, _ in r}):
        cells = [(f"{v:.5f}" if (v := next((d.get(probe) for st, d in series[n] if st == s),
                                           None)) is not None else "-").ljust(w)
                 for n in names]
        print(f"  {s:>4} " + "".join(cells))
    print()
for n, k in kls.items():
    if k:
        print(f"  {n}: KL {k[0]} -> {k[1]}")
PYEOF
echo
echo "Code flips (does it still LEARN? 59 steps is too short to judge — at lr 5e-4 the"
echo "non-KD arm shows only 0.0194% here against 4.22% over a full 613-step run):"
awk '/code flips vs run start/{f=1;next} f&&/^  /{print} f&&!/^  /{f=0}' "${OUT}/train.log" \
    | tail -8 | sed 's/self_attn\.//; s/mlp\.//; s/model\.layers\.//; s/ (0->.*scale-drift/ | drift/'

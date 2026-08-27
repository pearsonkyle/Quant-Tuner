#!/usr/bin/env bash
# The control arm notes.md declared before launch: teacher = the dense E4B itself.
#
# Removes both confounds the 31B arms carried at once —
#   * termination: the teacher's stop policy IS the target policy. The 31B's own probe
#     reads 0.0000 at every point where E4B reads 0.0703, so its anchor pulled toward
#     never stopping; the self teacher's reads 0.0703 by construction.
#   * metric: KD's target and the damage metric's reference become the same model, so
#     "moved toward the teacher" stops being scored as damage.
#
# Table (verified): coverage 0.9917, P(stop)=0.530 at real stop targets vs 0.000534
# elsewhere, median 0.605 — more decisive about stopping than the 31B's 0.387.
set -uo pipefail
cd "$(dirname "$0")/.."
R=out/gemma4-ternary
LR="${LR:-2e-4}"
OUT="${OUT:-$R/selfkd-lr${LR}}"
TABLE="$R/kd/e4bself_topk64_fs106.pt"

say() { echo "[selfkd $(date +%H:%M:%S)] $*"; }
[ -f "$TABLE" ] || { say "FATAL: no self-KD table"; exit 1; }

# Gate the teacher's POLICY, not just its table. Every table-level check passed on the
# 31B (tokenizer 262,144/262,144, coverage 0.9993, a 4,090x stop-signal ratio) and it was
# still unusable, because none of them asks whether the teacher's stop policy is the one
# the student should adopt. That reading was in the log before training started.
say "gating the teacher's own stop policy"
.venv/bin/python - <<'PY' || exit 1
import json, sys
from pathlib import Path
base = json.loads(Path("out/gemma4-ternary/stop_baseline.json").read_text())["probs"]
p = Path("out/gemma4-ternary/kd/teacher_probe_e4bself.json")
if not p.exists():
    print("[gate] no teacher probe yet — measuring")
    sys.exit(0)
t = json.loads(p.read_text())
t = t.get("probs", t)      # teacher_stop_probe writes flat; measure_stop_baseline nests
ctrl = "answer_after_tool"
if ctrl not in t:
    print(f"[gate] REFUSED: teacher probe has no {ctrl} — cannot judge its stop policy")
    sys.exit(1)
if t[ctrl] < 0.25 * base[ctrl]:
    print(f"[gate] REFUSED: teacher control {t[ctrl]:.5f} is under a quarter of the "
          f"student's {base[ctrl]:.5f} — distilling it teaches the student not to stop")
    sys.exit(1)
print(f"[gate] OK: teacher control {t[ctrl]:.5f} vs student {base[ctrl]:.5f}")
PY

say "waiting for the gpu"
while pgrep -f "quant_tuner[.]qat[.]train" > /dev/null; do sleep 30; done
while pgrep -f "kd_precomput[e].py" > /dev/null; do sleep 60; done

mkdir -p "$OUT"
say "training 0,1,2,3,7,8 ternary + self-KD at lr=$LR"
.venv/bin/python -m quant_tuner.qat.train \
    --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --corpus "$R/corpus_sft_gemma4_32768.pt" --val-corpus "$R/corpus_sft_gemma4_val_32768.pt" \
    --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --kd-table "$TABLE" --kd-alpha 0.5 --kd-temp 1.0 \
    --stop-anchor 0.2 --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1 \
    --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
    --optim adafactor --dtype fp32 --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs 0.0922 --lr "$LR" --warmup-frac 0.05 \
    --val-every 25 --probe-every 25 \
    --probe-abort 0.03 --probe-abort-control 0.01 --probe-abort-patience 2 \
    --ckpt-every 25 --ckpt-keep 2 \
    --out "$OUT" > "$OUT/train.log" 2>&1
say "rc=$?"
grep -a "STOPPROBE\|PROBE-ABORT" "$OUT/train.log" | tail -3

[ -f "$OUT/trained_latents.pt" ] && .venv/bin/python scripts/gemma4_stage_damage.py --probe \
    --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --ckpt "$OUT/trained_latents.pt" --out "$OUT/stage_damage_trained.json" 2>&1 | tail -6
say "done"

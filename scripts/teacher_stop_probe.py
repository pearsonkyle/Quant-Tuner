"""Measure a dense KD teacher's own stop probe -> teacher_probe.json.

The report ("Termination policy over training" panel) draws the teacher's values as
dotted asymptotes: under KD the student's probe series should converge toward them, so
the teacher must be measured with the SAME prompts and the SAME chat template the
student is probed with (the student's — SWE-Lego repos ship no chat template, and KD
feeds the teacher student-templated windows anyway).

    PYTHONPATH=src .venv/bin/python scripts/teacher_stop_probe.py \
        --teacher SWE-Lego/SWE-Lego-Qwen3-32B --out out/exp-058/kd/teacher_probe_32b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.kd_precompute import load_teacher, load_tokenizer_tolerant  # noqa: E402
from quant_tuner.qat.stop_probe import StopProbe, format_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="HF id or local path of the dense teacher")
    ap.add_argument("--student-model", default="out/exp-057/model",
                    help="tokenizer + chat template source (must match the training probe)")
    ap.add_argument("--out", required=True, help="teacher_probe.json path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    tok = load_tokenizer_tolerant(args.student_model)
    probe = StopProbe.build(tok)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = load_teacher(args.teacher, device=args.device, dtype=dtype)
    model.eval()
    probs = probe.measure(model, args.device)
    print(f"[teacher-probe] {args.teacher}: {format_line(probs, probe.dialect)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probs, indent=2) + "\n")
    print(f"[teacher-probe] -> {out}")


if __name__ == "__main__":
    main()

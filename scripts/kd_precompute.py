"""CLI shim for offline top-K teacher logits — logic in ``quant_tuner.qat.kd_precompute``.

Smoke test (a few windows, no 16 GB resident during training later):

    .venv/bin/python scripts/kd_precompute.py \
        --teacher SWE-Lego/SWE-Lego-Qwen3-8B \
        --corpus out/exp-058/distill_corpus_iter5-r2.pt \
        --max-windows 4 --out out/exp-058/kd_topk_smoke.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from quant_tuner.qat.kd_precompute import DEFAULT_TOPK, precompute_topk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    help="HF id or local path of a same-tokenizer dense teacher")
    ap.add_argument("--corpus", type=Path, required=True, help="masked corpus .pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="only process the first N windows (smoke test)")
    ap.add_argument("--student-model", type=Path, default=REPO / "out" / "exp-057" / "model")
    ap.add_argument("--chat-template", type=Path,
                    default=REPO / "out" / "exp-057" / "chat_template.jinja")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None)
    ap.add_argument("--include-ids", default="",
                    help="comma-separated token ids forced into every support row at "
                         "their true teacher logprob (pass the stop id, 151645, so the "
                         "KL constrains P(stop) per-position, not just via the tail "
                         "bucket)")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}.get(args.dtype or "")
    precompute_topk(
        corpus=args.corpus, teacher=args.teacher, out=args.out,
        student_model_dir=args.student_model, topk=args.topk,
        max_windows=args.max_windows, device=args.device, dtype=dtype or None,
        student_chat_template=args.chat_template,
        include_ids=[int(t) for t in args.include_ids.split(",") if t.strip()],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

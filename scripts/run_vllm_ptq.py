#!/usr/bin/env python3
"""Quantize an HF checkpoint to a vLLM-servable compressed-tensors dir (W4A16),
calibrated on our corpora at long context.

Typical use (gemma-4-E4B for the HomeLab GPU-1 vLLM deployment):

    uv sync --extra vllm-ptq
    uv run python scripts/run_vllm_ptq.py \
        --model ~/Programs/llm/hf/gemma-4-E4B-it \
        --corpus out/<run>/corpus/corpus.cal.txt \
        --out out/e4b-w4a16-logs8k \
        --ctx 8192

Then serve:  vllm serve out/e4b-w4a16-logs8k --max-model-len 131072 ...

Corpora come from scripts/build_corpora.py (calibration = logtrain train slice
+ wiki, interleaved). Passing several --corpus files splits the token budget
across them proportionally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_tuner.vllm_export import PTQConfig, run_ptq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        type=Path,
        help="calibration text file (repeatable; budget split proportionally)",
    )
    ap.add_argument("--out", required=True, type=Path, help="output checkpoint dir")
    ap.add_argument("--ctx", type=int, default=8192, help="calibration seq length")
    ap.add_argument("--budget-tokens", type=int, default=524_288)
    ap.add_argument(
        "--scheme",
        default="W4A16",
        choices=["W4A16", "W8A8", "W8A16", "FP8_DYNAMIC"],
    )
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument(
        "--pipeline",
        default="sequential",
        choices=["sequential", "basic", "independent"],
        help="use 'basic' for models with cross-layer state (gemma-4 shared KV)",
    )
    args = ap.parse_args()

    cfg = PTQConfig(
        model_id=args.model,
        out_dir=args.out,
        corpus_files=args.corpus,
        ctx=args.ctx,
        budget_tokens=args.budget_tokens,
        scheme=args.scheme,
        group_size=args.group_size,
        device_map=args.device_map,
        pipeline=args.pipeline,
    )
    out = run_ptq(cfg)
    print(f"quantized checkpoint written to {out}")


if __name__ == "__main__":
    main()

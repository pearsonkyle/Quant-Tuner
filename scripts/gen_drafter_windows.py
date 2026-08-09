#!/usr/bin/env python3
"""Generate long-context training windows for MTP-drafter fine-tuning.

    python scripts/gen_drafter_windows.py --logs logtrain.jsonl \
        --model ~/Programs/llm/hf/gemma-4-E4B-it \
        --out out/drafter/windows.jsonl --max-len 32768
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from quant_tuner.drafter.windows import WindowConfig, write_windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--model", required=True, help="tokenizer source (target model dir)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-len", type=int, default=32_768)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = WindowConfig(
        logs=args.logs, out=args.out, max_len=args.max_len,
        stride=args.stride, split=args.split,
    )
    audit = write_windows(cfg, tok)
    print(f"wrote {args.out}: {audit}")


if __name__ == "__main__":
    main()

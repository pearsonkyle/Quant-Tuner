#!/usr/bin/env python3
"""Fine-tune the Gemma-4 MTP drafter on long-context log windows.

    python scripts/train_drafter.py \
        --target ~/Programs/llm/hf/gemma-4-E4B-it \
        --drafter ~/Programs/llm/hf/drafter-ft-e4b \
        --windows out/drafter/windows-32k.jsonl \
        --out out/drafter/ft-logs --max-len 8192
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_tuner.drafter.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--windows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--target-device", default="cuda:0")
    ap.add_argument("--drafter-device", default="cuda:0")
    ap.add_argument("--no-4bit", action="store_true", help="load target in bf16 (needs ~16GB free)")
    args = ap.parse_args()

    cfg = TrainConfig(
        target_model=args.target,
        drafter_model=args.drafter,
        windows=args.windows,
        out_dir=args.out,
        max_len=args.max_len,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        target_device=args.target_device,
        drafter_device=args.drafter_device,
        load_target_4bit=not args.no_4bit,
    )
    out = train(cfg)
    print(f"drafter fine-tuned -> {out}")


if __name__ == "__main__":
    main()

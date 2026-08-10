#!/usr/bin/env python3
"""Online on-policy distillation: generate target outputs into a buffer, train the
drafter on them, and log an acceptance-vs-steps curve (eval every N steps).

    python scripts/train_drafter_online.py \
        --target ~/Programs/llm/hf/gemma-4-E4B-it --drafter out/drafter/ft-logs \
        --prompt-windows out/drafter/logs-tooldense-2k.jsonl \
        --eval-windows out/drafter/windows-test-4k.jsonl \
        --seed-windows out/drafter/onpolicy-logs.jsonl \
        --out out/drafter/online-onpolicy --eval-every 500 --max-steps 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_tuner.drafter.online import OnlineConfig, train_online


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--prompt-windows", required=True, type=Path)
    ap.add_argument("--eval-windows", required=True, type=Path)
    ap.add_argument("--seed-windows", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gen-base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--gen-model", default="gemma-4-e4b-w4a16-logs")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--target-acceptance", type=float, default=None)
    ap.add_argument("--max-gen", type=int, default=4000)
    args = ap.parse_args()

    cfg = OnlineConfig(
        target_model=args.target, drafter_model=args.drafter, out_dir=args.out,
        prompt_windows=args.prompt_windows, eval_windows=args.eval_windows,
        seed_windows=args.seed_windows, gen_base_url=args.gen_base_url, gen_model=args.gen_model,
        prompt_len=args.prompt_len, gen_len=args.gen_len, lr=args.lr, grad_accum=args.grad_accum,
        eval_every=args.eval_every, max_steps=args.max_steps,
        target_acceptance=args.target_acceptance, max_gen=args.max_gen,
    )
    out = train_online(cfg)
    print(f"online distillation done -> {out}")


if __name__ == "__main__":
    main()

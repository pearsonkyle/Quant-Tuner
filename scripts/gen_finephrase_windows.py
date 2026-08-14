#!/usr/bin/env python3
"""Stream a longest-first FinePhrase subset into drafter training windows.

    python scripts/gen_finephrase_windows.py --model ~/Programs/llm/hf/gemma-4-E4B-it \
        --out out/drafter/finephrase-200m.jsonl --token-budget 200000000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from quant_tuner.drafter.finephrase import FinePhraseConfig, write_windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="tokenizer source (target model dir)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--token-budget", type=int, default=200_000_000)
    ap.add_argument("--min-gen-tokens", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--configs", nargs="+", default=["faq", "math", "table", "tutorial"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = FinePhraseConfig(
        out=args.out, token_budget=args.token_budget,
        min_gen_tokens=args.min_gen_tokens, max_len=args.max_len,
        configs=tuple(args.configs),
    )
    stats = write_windows(cfg, tok)
    print(f"wrote {args.out}: {stats}")


if __name__ == "__main__":
    main()

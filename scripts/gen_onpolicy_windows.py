#!/usr/bin/env python3
"""Generate on-policy distillation windows from a deployed target.

Queries an OpenAI-compatible target for GREEDY continuations over prompts taken
from an existing windows file, and writes windows of prompt_ids + target's own
generated token ids (with gen_start). Train the drafter on these so it learns the
target's actual output distribution — the standard EAGLE fix.

    python scripts/gen_onpolicy_windows.py \
        --base-url http://127.0.0.1:1234/v1 --model gemma-4-e4b-w4a16-logs \
        --prompt-windows out/drafter/logs-tooldense-2k.jsonl \
        --out out/drafter/onpolicy-logs.jsonl --max-prompts 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_tuner.drafter.onpolicy import OnPolicyConfig, generate_windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-windows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=512)
    ap.add_argument("--max-prompts", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    cfg = OnPolicyConfig(
        base_url=args.base_url, model=args.model,
        out=args.out, prompt_windows=args.prompt_windows,
        prompt_len=args.prompt_len, gen_len=args.gen_len,
        max_prompts=args.max_prompts, concurrency=args.concurrency,
    )
    stats = generate_windows(cfg)
    print(f"wrote {args.out}: {stats}")


if __name__ == "__main__":
    main()

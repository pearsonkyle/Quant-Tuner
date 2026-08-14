#!/usr/bin/env python3
"""Mix several drafter-window JSONLs into one shuffled training file.

Each --source is 'path[:repeat[:cap_tokens]]':
  - repeat: emit the file's windows this many times (upsample a small
    in-distribution set relative to a big general one).
  - cap_tokens: stop taking from this source after ~this many tokens.

Example (agentic logs primary, finephrase for breadth):
  python scripts/mix_windows.py --out out/drafter/mix.jsonl --seed 42 \
    --source out/drafter/logs-tooldense-2k.jsonl:2 \
    --source out/drafter/finephrase-30m.jsonl:1:12000000

Deterministic given (--seed, sources). Prints a per-source token breakdown.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", action="append", required=True, dest="sources")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool: list[str] = []
    breakdown = {}
    for spec in args.sources:
        parts = spec.split(":")
        path = parts[0]
        repeat = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        cap = int(parts[2]) if len(parts) > 2 and parts[2] else None
        lines = Path(path).read_text().splitlines()
        taken, toks = 0, 0
        for _ in range(repeat):
            for ln in lines:
                if cap is not None and toks >= cap:
                    break
                n = len(json.loads(ln)["input_ids"])
                pool.append(ln)
                taken += 1
                toks += n
            if cap is not None and toks >= cap:
                break
        breakdown[Path(path).name] = {"windows": taken, "tokens": toks, "repeat": repeat}

    rng.shuffle(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ln in pool:
            f.write(ln + "\n")
    total = sum(v["tokens"] for v in breakdown.values())
    print(f"wrote {args.out}: {len(pool)} windows, {total/1e6:.1f}M tokens")
    for name, v in breakdown.items():
        print(f"  {name}: {v['windows']} windows, {v['tokens']/1e6:.1f}M tokens (x{v['repeat']}) "
              f"= {100*v['tokens']/total:.0f}%")


if __name__ == "__main__":
    main()

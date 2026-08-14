#!/usr/bin/env python3
"""Build a combined SFT messages JSONL for E4B target training.

Sources (all normalized to strict user/assistant alternation, system stubbed):
  - our agentic Claude-Code logs (in-distribution agentic)
  - pearsonkyle/swe-agentic-trajectories [resolved] (real solved SWE tool-use)
  - pearsonkyle/broad-domain-supplement [instruct] (9-domain breadth; mtp half)
  - a light dusting of FinePhrase (kept small — it's simple)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def to_alt(msgs, sys_stub_chars=600):
    seq = []
    sys_txt = ""
    for m in msgs:
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):  # multimodal/typed content -> join text parts
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        c = (c or "").strip()
        if not c:
            continue
        if role == "system":
            sys_txt += c + "\n"
            continue
        r = "assistant" if role == "assistant" else "user"  # user/tool -> user
        seq.append((r, c))
    if not seq:
        return None
    if sys_txt:
        stub = sys_txt[:sys_stub_chars].rsplit(" ", 1)[0]
        if seq[0][0] == "user":
            seq[0] = ("user", stub + "\n\n" + seq[0][1])
        else:
            seq = [("user", stub)] + seq
    merged = []
    for r, c in seq:
        if merged and merged[-1][0] == r:
            merged[-1] = (r, merged[-1][1] + "\n\n" + c)
        else:
            merged.append((r, c))
    while merged and merged[0][0] == "assistant":
        merged.pop(0)
    if len(merged) < 2 or not any(r == "assistant" for r, _ in merged):
        return None
    return [{"role": r, "content": c} for r, c in merged]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agentic-logs", type=Path, default=Path("out/sft/agentic-logs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("out/sft/combined.jsonl"))
    ap.add_argument("--n-swe", type=int, default=500)
    ap.add_argument("--n-broad", type=int, default=1800)
    ap.add_argument("--n-finephrase", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import urllib.request

    def iter_hub_jsonl(url):
        with urllib.request.urlopen(url, timeout=120) as r:
            for line in r:
                line = line.strip()
                if line:
                    yield json.loads(line)

    SWE = "https://huggingface.co/datasets/pearsonkyle/swe-agentic-trajectories/resolve/main/data/resolved.jsonl"
    BROAD = "https://huggingface.co/datasets/pearsonkyle/broad-domain-supplement/resolve/main/data/instruct.jsonl"

    rng = random.Random(args.seed)
    pool = []
    counts = {}

    # 1) our agentic logs (already normalized)
    if args.agentic_logs.is_file():
        n = 0
        with open(args.agentic_logs) as f:
            for line in f:
                pool.append(json.loads(line))
                n += 1
        counts["agentic-logs"] = n

    # 2) swe-agentic-trajectories [resolved] — real solved SWE tool-use
    n = 0
    for row in iter_hub_jsonl(SWE):
        if n >= args.n_swe:
            break
        conv = to_alt(row.get("messages", []))
        if conv:
            pool.append({"messages": conv})
            n += 1
    counts["swe-resolved"] = n

    # 3) broad-domain-supplement [instruct] — 9-domain breadth, prefer mtp half
    n = 0
    for row in iter_hub_jsonl(BROAD):
        if n >= args.n_broad:
            break
        if row.get("half") not in (None, "mtp"):
            continue
        conv = to_alt(row.get("messages", []))
        if conv:
            pool.append({"messages": conv})
            n += 1
    counts["broad-domain"] = n

    # 4) light FinePhrase (optional; via our windows decoded back is lossy, so skip
    #    unless a prepared messages file is supplied)
    counts["finephrase"] = 0

    rng.shuffle(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in pool:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.out}: {len(pool)} conversations")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

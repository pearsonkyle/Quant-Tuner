#!/usr/bin/env python3
"""Build a knowledge-probe set for the Jacobian-lens knowledge-loss study.

A probe is ``{id, prompt, gold, domain}``: a short prompt with a one-token-ish
gold continuation whose lens rank we track across layers (see
``quant_tuner.lens.probes``). Two sources:

  1. a static seed set of factual/commonsense probes (the paper's currency-of-boot
     example and friends), spanning a few domains;
  2. tool-log-mined probes — entity/fact continuations pulled from the tool-call
     holdout so the set includes in-distribution knowledge the calibration
     corpus is meant to preserve.

Writes ``artifacts/probes.jsonl``. Deterministic (seed 42).

Usage::

    PYTHONPATH=src python scripts/build_probe_set.py \
        --holdout out/omnicoder_q4_k_m/toolcall_holdout.jsonl \
        --out artifacts/probes.jsonl --n-mined 150
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

# Static seed probes: (prompt, gold, domain). Golds chosen to be a natural
# single-word continuation. Kept small and hand-checked; the mined set adds bulk.
SEED_PROBES: list[tuple[str, str, str]] = [
    ("Fact: The capital of Japan is Tokyo.\nFact: The currency used in the country shaped like a boot is", "Euro", "geography"),
    ("The chemical symbol for gold is", "Au", "science"),
    ("The largest planet in our solar system is", "Jupiter", "science"),
    ("The author of Romeo and Juliet is William", "Shakespeare", "literature"),
    ("The speed of light is approximately 300,000 kilometers per", "second", "science"),
    ("The powerhouse of the cell is the", "mitochondria", "science"),
    ("In Python, the keyword used to define a function is", "def", "code"),
    ("The HTTP status code for 'Not Found' is", "404", "code"),
    ("The command to list files in a Unix shell is", "ls", "code"),
    ("Git's command to record changes to the repository is git", "commit", "code"),
    ("The square root of 144 is", "12", "math"),
    ("Water boils at 100 degrees", "Celsius", "science"),
    ("The first president of the United States was George", "Washington", "history"),
    ("The tallest mountain on Earth is Mount", "Everest", "geography"),
    ("The programming language created by Guido van Rossum is", "Python", "code"),
    ("JSON stands for JavaScript Object", "Notation", "code"),
    ("The opposite of 'true' in boolean logic is", "false", "code"),
    ("The default port for HTTPS is", "443", "code"),
    ("The number of bits in a byte is", "8", "code"),
    ("The SQL keyword to retrieve data is", "SELECT", "code"),
]


def _mine_from_holdout(holdout: Path, n: int, rng: random.Random) -> list[dict]:
    """Pull short prompt->next-word probes from tool-call user/assistant text.

    Cheap heuristic: take sentences from user turns, cut at a content word near
    the end, and use the following word as the gold. Filters to lines that read
    like factual statements (avoid code blocks / tool JSON).
    """
    if not holdout.exists():
        return []
    sessions = [json.loads(line) for line in holdout.read_text().splitlines() if line.strip()]
    candidates: list[dict] = []
    for s in sessions:
        for m in s.get("messages", []):
            if m.get("role") not in ("user", "assistant"):
                continue
            content = m.get("content") or ""
            for sent in re.split(r"(?<=[.!?])\s+", content):
                words = sent.strip().split()
                if not (8 <= len(words) <= 30):
                    continue
                if any(c in sent for c in "{}[]()<>`|") or "http" in sent:
                    continue
                cut = len(words) - 1
                gold = words[cut].strip(".,!?;:\"'")
                if not gold or not gold.isalnum() or len(gold) < 3:
                    continue
                prompt = " ".join(words[:cut])
                candidates.append({"prompt": prompt, "gold": gold, "domain": "tools"})
    rng.shuffle(candidates)
    # de-dup by prompt tail
    seen = set()
    out = []
    for c in candidates:
        key = c["prompt"][-40:]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", type=Path, default=None, help="tool-call holdout JSONL to mine")
    ap.add_argument("--out", type=Path, default=_REPO / "artifacts" / "probes.jsonl")
    ap.add_argument("--n-mined", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probes: list[dict] = []
    for i, (prompt, gold, domain) in enumerate(SEED_PROBES):
        probes.append({"id": f"seed_{i:03d}", "prompt": prompt, "gold": gold, "domain": domain})
    if args.holdout is not None:
        for i, p in enumerate(_mine_from_holdout(args.holdout, args.n_mined, rng)):
            probes.append({"id": f"mined_{i:03d}", **p})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")
    domains = {}
    for p in probes:
        domains[p["domain"]] = domains.get(p["domain"], 0) + 1
    print(f"wrote {len(probes)} probes -> {args.out}")
    print("  by domain:", domains)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""How big is the proposed super-dataset, and what would it cost to train?

    python scripts/size_super_dataset.py --ultrachat 75000 --distill 50000

Sizing only — it builds nothing and writes no corpus. The point is to answer, before
committing a multi-day run, three questions the conversation-count proposal leaves open:

1. **How many tokens is that really?** Conversation counts are not comparable across
   these sources. Tokens per conversation ranges from ~650 (`iterative_instruction`) to
   ~8,950 (`grounded_long_context`) — a 14x spread — so "50K conversations" can mean
   32M tokens or 200M depending only on which domains it lands in.
2. **Is the ask even available?** The distillation repo's benchmark-free deduped pool is
   finite, and a request above it silently becomes "all of it" rather than failing.
3. **What does it cost?** At the measured 66 s/step, wall-clock is the number that
   decides whether this is a weekend or a fortnight.

Reads the local parquet caches; no network unless they are missing.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_external_sft import UPSTREAM_SPLITS, fetch_config, fetch_ultrachat  # noqa: E402

from quant_tuner.data import external_sft as ext  # noqa: E402
from quant_tuner.qat.corpus import load_tokenizer  # noqa: E402

# Measured on this box for the 32768-window, all-36-layer, fp32 adafactor config.
S_PER_STEP = 66.0
WINDOW = 32768
# Fraction of raw conversation tokens that survive packing into training windows, from the
# three corpora already built: 20.00M -> 610 windows, 20.21M -> 604, 20.09M -> 613.
# ~0.98 of raw tokens land in a window; the rest is tail that cannot fill one.
PACK_EFFICIENCY = 0.98


def sample_tokens(recs: list[dict], tok, n: int = 400, seed: int = 42) -> float:
    """Mean tokens/conversation from a deterministic sample — tokenizing 125K
    conversations to answer a sizing question is not worth 40 minutes."""
    import random
    if not recs:
        return 0.0
    rng = random.Random(seed)
    pick = recs if len(recs) <= n else rng.sample(recs, n)
    total = 0
    for r in pick:
        text = tok.apply_chat_template(r["messages"], tools=r.get("tools"),
                                       tokenize=False)
        total += len(tok.encode(text, add_special_tokens=False))
    return total / len(pick)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ultrachat", type=int, default=75_000)
    ap.add_argument("--distill", type=int, default=50_000)
    ap.add_argument("--our-sft", type=Path,
                    default=Path("out/corpora/qwen3-universal-v2/sft.jsonl.gz"))
    ap.add_argument("--cache", type=Path, default=Path("out/external/hf-sft"))
    ap.add_argument("--distill-cache", type=Path, default=Path("out/external/distill"))
    ap.add_argument("--keep-benchmarks", action="store_true")
    ap.add_argument("--sample", type=int, default=400)
    a = ap.parse_args()

    tok = load_tokenizer()
    parts = []

    # ---- ultrachat -----------------------------------------------------------
    uc = list(ext.convert_ultrachat_rows(
        fetch_ultrachat(ext.ULTRACHAT_SPLIT, a.cache, None)))
    uc_train = [r for r in uc if r["split"] == "train"]
    uc_mean = sample_tokens(uc_train, tok, a.sample)
    uc_take = min(a.ultrachat, len(uc_train))
    parts.append(("ultrachat", uc_take, len(uc_train), uc_take * uc_mean, uc_mean))

    # ---- distillation --------------------------------------------------------
    raw: list[dict] = []
    for usplit in UPSTREAM_SPLITS:
        rows = fetch_config(ext.DISTILL_REPO, "canonical", usplit, a.distill_cache)
        if rows:
            raw += list(ext.convert_distill_rows(
                ext.dedup_rows(rows), source="distill", split=usplit,
                drop_benchmarks=not a.keep_benchmarks))
    d_train = [r for r in raw if r["split"] == "train"]
    doms = collections.Counter((r.get("meta") or {}).get("domain") for r in d_train)
    d_mean = sample_tokens(d_train, tok, a.sample)
    d_take = min(a.distill, len(d_train))
    parts.append(("distillation", d_take, len(d_train), d_take * d_mean, d_mean))

    # ---- our SFT (taken whole) ----------------------------------------------
    import gzip
    import json
    ours = []
    if a.our_sft.exists():
        op = gzip.open if a.our_sft.suffix == ".gz" else open
        with op(a.our_sft, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ours.append(json.loads(line))
    ours_train = [r for r in ours if r.get("split", "train") == "train"]
    o_mean = sample_tokens(ours_train, tok, a.sample)
    parts.append(("our SFT (all)", len(ours_train), len(ours_train),
                  len(ours_train) * o_mean, o_mean))

    # ---- report --------------------------------------------------------------
    print()
    print(f"{'source':<18}{'take':>9}{'available':>12}{'tok/conv':>11}{'tokens':>16}")
    print("-" * 66)
    total = 0.0
    for name, take, avail, toks, mean in parts:
        short = "  (ALL — asked for more than exists)" if take < 10**9 and take == avail \
                and take < (a.ultrachat if name == "ultrachat" else
                            a.distill if name == "distillation" else 10**9) else ""
        print(f"{name:<18}{take:>9,}{avail:>12,}{mean:>11,.0f}{toks:>16,.0f}{short}")
        total += toks
    print("-" * 66)
    print(f"{'TOTAL':<18}{'':>9}{'':>12}{'':>11}{total:>16,.0f}")

    windows = total * PACK_EFFICIENCY / WINDOW
    hours = windows * S_PER_STEP / 3600
    print()
    print(f"at window {WINDOW}, ~{windows:,.0f} windows = {windows:,.0f} steps "
          f"(1 epoch, grad-accum 1)")
    print(f"at {S_PER_STEP:.0f} s/step that is {hours:,.1f} h = {hours/24:,.1f} days "
          f"for ONE epoch")
    print(f"with --matmul-precision high (TF32, ~1.38x): {hours/1.38:,.1f} h "
          f"= {hours/1.38/24:,.1f} days")
    print()
    print("For scale: each curriculum round is ~610 windows / ~11 h.")

    print()
    print("distillation domains available (benchmark-free):")
    for dom, n in doms.most_common():
        mark = "  (agentic)" if dom in ext.AGENTIC_DOMAINS else ""
        print(f"  {str(dom):<32}{n:>8,}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI shim for the universal-SFT QAT corpus — logic in ``quant_tuner.qat.corpus``.

Reads the ``sft.jsonl.gz`` that ``quant_tuner.data.universal`` writes next to the
calibration corpus (full conversations, real tool schemas, scrubbed system prompts,
a ``split`` field that matches the calibration corpus) and renders it through the
STUDENT's chat template with the same assistant/tool masking every other QAT corpus
uses. EVERY source is taken whole by default — unlike the calibration corpus, which is
token-budgeted because the quantizers sample a fixed slice of it. QAT spends its budget
in epochs, so the corpus on disk should hold everything.

    PYTHONPATH=src .venv/bin/python scripts/build_sft_qat_corpus.py \
        --sft out/corpora/qwen3-universal/sft.jsonl.gz \
        --window 8064 --max-tool-tokens 3072 --min-density 0.05 \
        --out out/exp-058/sft_corpus_universal_8064.pt

Cap a source with ``--budget SOURCE=TOKENS`` (``none`` = uncapped, ``0`` = drop it).

Scale ``--max-tool-tokens`` with the window (3072 at 8064, 4096 at 12288, 8192 at 32768):
1024 was only ever right at a 4096 window and drops 28% of all conversation content.

For a long window, pair this with the trainer's ``--trained-tail``: at 32768 only the last
N tokens carry gradient, so the window buys *context* for the part being trained. That is
the right shape for agentic data, where the terminal ``<|im_end|>`` — the stop decision —
sits at the end of the window anyway.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.qat.corpus import (  # noqa: E402
    SFT_DEFAULT_BUDGETS,
    build_sft_corpus,
)


def _parse_budgets(items: list[str] | None) -> dict[str, int | None]:
    budgets = dict(SFT_DEFAULT_BUDGETS)
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--budget expects SOURCE=TOKENS, got {item!r}")
        src, _, val = item.partition("=")
        budgets[src.strip()] = None if val.strip().lower() in {"none", ""} else int(val)
    return budgets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True, help="sft.jsonl(.gz) from data.universal")
    ap.add_argument("--split", default="train",
                    help="'train' (default) keeps the test/holdout eval slices held out; "
                         "'all' uses every row — only for a throwaway diagnostic")
    ap.add_argument("--source", action="append", default=None,
                    help="restrict to these sources (repeatable); default = all present")
    ap.add_argument("--budget", action="append", default=None,
                    help="SOURCE=TOKENS per-source cap; 'none' = uncapped, 0 = drop")
    ap.add_argument("--window", type=int, default=8064,
                    help="chunked SDPA removed the old n_heads*S^2 < 2^31 kernel cap "
                         "(S <= 8191 at 32 heads); the limit is now memory, and the "
                         "trainer's --trained-tail is what makes 32768 affordable. "
                         "8064 remains the largest full-gradient window that trains clean.")
    ap.add_argument("--max-tool-tokens", type=int, default=1024)
    ap.add_argument("--min-density", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=REPO / "out" / "exp-058" / "sft_corpus_universal.pt")
    args = ap.parse_args()

    build_sft_corpus(
        sft_path=args.sft,
        data_split=None if args.split == "all" else args.split,
        sources=args.source,
        budgets=_parse_budgets(args.budget),
        window=args.window,
        max_tool_tokens=args.max_tool_tokens,
        min_density=args.min_density,
        seed=args.seed,
        out=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

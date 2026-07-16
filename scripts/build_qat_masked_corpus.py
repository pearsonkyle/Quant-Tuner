"""CLI shim for the log-based masked QAT corpus — logic in ``quant_tuner.qat.corpus``.

    PYTHONPATH=src .venv/bin/python scripts/build_qat_masked_corpus.py \
        --window 4096 --wiki-tokens 300000 --max-tool-tokens 1024 \
        --out out/exp-058/masked_corpus_4096_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.qat.corpus import build_log_corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4096,
                    help="tokens per window; 4096 is the MPS hard max")
    ap.add_argument("--wiki-tokens", type=int, default=300_000)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="logtrain slice; 'test' builds the --val-corpus set")
    ap.add_argument("--max-tool-tokens", type=int, default=0,
                    help="head+tail truncate role=tool contents to N tokens (0 = off)")
    ap.add_argument("--min-density", type=float, default=0.0,
                    help="drop windows below this trainable-token fraction")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "masked_corpus.pt")
    args = ap.parse_args()
    build_log_corpus(window=args.window, wiki_tokens=args.wiki_tokens,
                     data_split=args.split, max_tool_tokens=args.max_tool_tokens,
                     min_density=args.min_density, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

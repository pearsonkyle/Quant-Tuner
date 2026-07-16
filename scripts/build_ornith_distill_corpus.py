"""CLI shim for the strong-solver distillation corpus — logic in ``quant_tuner.qat.corpus``.

Takes a strong solver's agentic SWE-rebench trajectories, keeps the RESOLVED ones (verified
by the hidden tests), and re-renders them through the student's chat template + tokenizer
with the assistant/tool spans masked. See ``quant_tuner.qat.corpus.build_distill_corpus``.

    PYTHONPATH=src .venv/bin/python scripts/build_ornith_distill_corpus.py \
        --traj-dir out/swe-rebench/ornith-distill-gen/trajectories/Ornith-1.0-9B-Q5_K_M \
        --results out/swe-rebench/ornith-distill-gen/results.csv \
        --max-tool-tokens 1024 --out out/exp-058/distill_corpus_4096.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.qat.corpus import build_distill_corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True, action="append",
                    help="a <model> trajectory dir (repeatable to pool several runs)")
    ap.add_argument("--results", type=Path, default=None, action="append",
                    help="results.csv paired with each --traj-dir (same order); optional — "
                         "if omitted, resolved status is read from per-instance result.json")
    ap.add_argument("--all-patched", action="store_true",
                    help="train on all patched trajectories, not just resolved (mimicry risk)")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--max-tool-tokens", type=int, default=1024,
                    help="head+tail truncate tool outputs (container dumps are huge)")
    ap.add_argument("--min-density", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "distill_corpus.pt")
    args = ap.parse_args()
    build_distill_corpus(traj_dirs=args.traj_dir, results=args.results,
                         all_patched=args.all_patched, window=args.window,
                         max_tool_tokens=args.max_tool_tokens,
                         min_density=args.min_density, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

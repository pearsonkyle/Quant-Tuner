#!/usr/bin/env python3
"""Re-run the README headline table's task-level evals consistently.

For each of the six models in the published leaderboard row set:

  - baseline/fp16              (model-f16.gguf)
  - hybrid / mixed             (Q4_K_M-hybrid_mixed.gguf)
  - hybrid / custom            (Q4_K_M-hybrid_custom.gguf)
  - stock  / custom            (Q4_K_M-stock_custom.gguf)
  - stock  / wiki              (Q4_K_M-stock_wiki.gguf)
  - baseline/Q4_K_M-none       (Q4_K_M-none.gguf)

run two benchmarks back-to-back, each spawning one ``llama-server`` per
model with ``--chat-template-kwargs '{"enable_thinking":false}'``:

  1. tool-call: ``--reps 5`` at T=0.6 / top_p=0.95 / top_k=20
     (seeds 1000–1004), ``--ctx 32768`` so rollouts don't overflow
     after several tool-call rounds. ~3–4 h per model.
  2. MMLU-Pro:  ``--reps 1`` at T=0 over the 5-shot 50-question holdout.
     ~5–10 min per model.

Outputs (overwritten on re-run):

  out/omnicoder_q4_k_m/table_toolcall_{results,aggregated}.csv
  out/mmlu_pro/table_mmlu_{results,aggregated}.csv

Each sub-step is delegated to the existing rep runners so behavior stays
in lockstep with the rest of the codebase. Pass ``--skip-toolcall`` /
``--skip-mmlu`` to scope down. Total wall time at defaults (5 reps × 6
models + MMLU): ~25 h.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WORK = _REPO / "out" / "omnicoder_q4_k_m"
_MMLU = _REPO / "out" / "mmlu_pro"

TABLE_MODELS = [
    _WORK / "model-f16.gguf",
    _WORK / "Q4_K_M-hybrid_mixed.gguf",
    _WORK / "Q4_K_M-hybrid_custom.gguf",
    _WORK / "Q4_K_M-stock_custom.gguf",
    _WORK / "Q4_K_M-stock_wiki.gguf",
    _WORK / "Q4_K_M-none.gguf",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps-toolcall", type=int, default=5)
    p.add_argument("--reps-mmlu", type=int, default=1)
    p.add_argument(
        "--mmlu-holdout",
        type=Path,
        default=_MMLU / "holdout_5shot.json",
        help="5-shot holdout JSON (default: out/mmlu_pro/holdout_5shot.json)",
    )
    p.add_argument("--skip-toolcall", action="store_true")
    p.add_argument("--skip-mmlu", action="store_true")
    return p


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=_REPO)
    if res.returncode != 0:
        raise SystemExit(
            f"stage failed (rc={res.returncode}) after {(time.time() - t0) / 60:.1f} min"
        )


def main() -> int:
    args = _build_arg_parser().parse_args()

    missing = [m for m in TABLE_MODELS if not m.exists()]
    if missing:
        print("ERROR: missing models:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    model_args = [str(m) for m in TABLE_MODELS]

    if not args.skip_toolcall:
        _run([
            "uv", "run", "python", "scripts/run_toolcall_reps.py",
            "--models", *model_args,
            "--reps", str(args.reps_toolcall),
            "--results", str(_WORK / "table_toolcall_results.csv"),
            "--aggregated", str(_WORK / "table_toolcall_aggregated.csv"),
        ])

    if not args.skip_mmlu:
        if not args.mmlu_holdout.exists():
            print(
                f"ERROR: MMLU holdout missing at {args.mmlu_holdout}. "
                f"Build it with: uv run python scripts/build_mmlu_pro_holdout.py "
                f"--n-shot 5 --out {args.mmlu_holdout}",
                file=sys.stderr,
            )
            return 1
        _run([
            "uv", "run", "python", "scripts/run_mmlu_pro_reps.py",
            "--models", *model_args,
            "--holdout", str(args.mmlu_holdout),
            "--reps", str(args.reps_mmlu),
            "--temperature", "0.0",
            "--max-tokens", "128",
            "--results", str(_MMLU / "table_mmlu_results.csv"),
            "--aggregated", str(_MMLU / "table_mmlu_aggregated.csv"),
        ])

    print("\n=== refresh_table_evals: DONE ===", flush=True)
    print("  tool-call:  out/omnicoder_q4_k_m/table_toolcall_aggregated.csv")
    print("  MMLU-Pro:   out/mmlu_pro/table_mmlu_aggregated.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

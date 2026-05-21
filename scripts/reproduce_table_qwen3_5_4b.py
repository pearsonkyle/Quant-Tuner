"""Run reproduce_table.py twice for Qwen/Qwen3.5-4B: Q4_K_M, then IQ4_NL.

Each pass is fully idempotent and lives in its own workspace
(``out/qwen_qwen3_5_4b_q4_k_m/`` and ``out/qwen_qwen3_5_4b_iq4_nl/``).

    uv run python scripts/reproduce_table_qwen3_5_4b.py \\
        --logs logtrain.jsonl --wiki-test-raw /path/to/wiki.test.raw

Pass any flag through verbatim — e.g. ``--skip-toolcall --skip-mmlu`` to do
just the calibration + KLD bench rows for both quant types.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODEL = "Qwen/Qwen3.5-4B"
QUANT_TYPES = ("Q4_K_M", "IQ4_NL")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl")
    p.add_argument("--wiki-test-raw", type=Path, default=None)
    p.add_argument("--supplement", type=Path,
                   default=REPO / "calibration_supplement.txt")
    p.add_argument("--skip-speed-rebench", action="store_true")
    p.add_argument("--skip-toolcall", action="store_true")
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--toolcall-reps", type=int, default=5)
    p.add_argument("--mmlu-reps", type=int, default=1)
    p.add_argument("--mmlu-n-shot", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runner = REPO / "scripts" / "reproduce_table.py"

    for qt in QUANT_TYPES:
        cmd: list[str] = [
            sys.executable, str(runner),
            "--model", MODEL,
            "--quant-type", qt,
            "--logs", str(args.logs),
            "--supplement", str(args.supplement),
            "--toolcall-reps", str(args.toolcall_reps),
            "--mmlu-reps", str(args.mmlu_reps),
            "--mmlu-n-shot", str(args.mmlu_n_shot),
        ]
        if args.wiki_test_raw is not None:
            cmd += ["--wiki-test-raw", str(args.wiki_test_raw)]
        for flag in ("skip_speed_rebench", "skip_toolcall", "skip_mmlu", "dry_run"):
            if getattr(args, flag):
                cmd.append("--" + flag.replace("_", "-"))

        print(f"\n{'=' * 60}\n>>> {qt}\n{'=' * 60}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[abort] {qt} pass failed (exit {rc})", flush=True)
            return rc

    print("\n=== BOTH PASSES DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

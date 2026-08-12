"""Run MMLU-Pro reps across all 14 MMLU-Pro domains.

Convenience wrapper that (a) builds a holdout covering every MMLU-Pro
subject if one isn't already present, then (b) invokes
``run_mmlu_pro_reps.py`` with the given models, reps, and sampling.

Example:

    PYTHONPATH=src .venv/bin/python scripts/run_mmlu_pro_all_domains.py \\
        --models out/exp-001/.../model-f16.gguf out/exp-001/.../Q4_K_M-custom.gguf \\
        --reps 5 --n-per-subject 25 \\
        --label jackrong-all-domains

The label controls where outputs land:
    out/mmlu_pro/<label>/{holdout.json, reps_per.csv, reps_agg.csv, reps_logs/, reps_run.log}
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Standard MMLU-Pro categories (TIGER-Lab/MMLU-Pro). 14 subjects.
ALL_SUBJECTS = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, type=Path,
                   help="GGUF paths to evaluate")
    p.add_argument("--label", required=True,
                   help="Subdir under out/mmlu_pro/ for outputs")
    p.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS,
                   help=f"Subjects to include (default: all {len(ALL_SUBJECTS)})")
    p.add_argument("--n-per-subject", type=int, default=25)
    p.add_argument("--n-shot", type=int, default=2)
    p.add_argument("--holdout-seed", type=int, default=42)
    p.add_argument("--holdout", type=Path, default=None,
                   help="Reuse an existing holdout (skip build step)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--chat-template-kwargs", default='{"enable_thinking":false}',
                   help="Forwarded to run_mmlu_pro_reps (default disables "
                        "Qwen3-family thinking). Pass '' to omit (e.g. for Gemma).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them")
    return p


def _build_holdout(args, out_dir: Path) -> Path:
    if args.holdout is not None:
        if not args.holdout.exists():
            print(f"--holdout points at missing file: {args.holdout}", file=sys.stderr)
            sys.exit(2)
        return args.holdout

    holdout = out_dir / "holdout.json"
    if holdout.exists():
        print(f"reusing existing holdout: {holdout}")
        return holdout

    cmd = [
        sys.executable, str(REPO / "scripts" / "build_mmlu_pro_holdout.py"),
        "--subjects", *args.subjects,
        "--n-per-subject", str(args.n_per_subject),
        "--n-shot", str(args.n_shot),
        "--seed", str(args.holdout_seed),
        "--out", str(holdout),
    ]
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    if args.dry_run:
        return holdout
    subprocess.run(cmd, check=True, env={**__import__('os').environ,
                                         "PYTHONPATH": f"{REPO}/src"})
    return holdout


def _run_reps(args, holdout: Path, out_dir: Path) -> int:
    per_csv = out_dir / "reps_per.csv"
    agg_csv = out_dir / "reps_agg.csv"
    log_dir = out_dir / "reps_logs"
    run_log = out_dir / "reps_run.log"

    cmd = [
        sys.executable, str(REPO / "scripts" / "run_mmlu_pro_reps.py"),
        "--models", *(str(m) for m in args.models),
        "--holdout", str(holdout),
        "--reps", str(args.reps),
        "--base-seed", str(args.base_seed),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--ctx", str(args.ctx),
        "--ngl", str(args.ngl),
        "--max-tokens", str(args.max_tokens),
        "--chat-template-kwargs", args.chat_template_kwargs,
        "--results", str(per_csv),
        "--aggregated", str(agg_csv),
        "--log-dir", str(log_dir),
    ]
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)} > {run_log} 2>&1")
    if args.dry_run:
        return 0

    n_samples = args.n_per_subject * len(args.subjects)
    n_total = n_samples * args.reps * len(args.models)
    print(f"\nTotal: {len(args.models)} model(s) × {args.reps} rep(s) × "
          f"{n_samples} sample(s) = {n_total} llama-server completions\n")

    with run_log.open("w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            env={**__import__('os').environ, "PYTHONPATH": f"{REPO}/src"},
        )
    return proc.returncode


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = REPO / "out" / "mmlu_pro" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = _build_holdout(args, out_dir)
    rc = _run_reps(args, holdout, out_dir)
    if rc != 0:
        print(f"run_mmlu_pro_reps.py exited {rc}", file=sys.stderr)
        return rc
    print(f"\nDone. Outputs under {out_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

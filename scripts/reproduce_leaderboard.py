"""End-to-end reproducer for the OmniCoder-9B Q4_K_M leaderboard in the README.

Chains the existing per-stage drivers. Each stage's outputs are detected and
skipped on re-run, so this script is safe to launch repeatedly: a clean
machine runs the whole pipeline (~17-20 h), a populated machine just verifies
state and rebuilds the leaderboard.

Pipeline:

    Stage  Inputs                                  Outputs / artifacts
    ─────  ──────────────────────────────────────  ─────────────────────────────────
    1      Tesslate/OmniCoder-9B (HF)              model_extracted/, model-f16.gguf,
                                                   corpus.train.txt, corpus.eval.txt,
                                                   imatrix-custom.gguf,
                                                   imatrix-hybrid_custom.gguf,
                                                   baseline.kld,
                                                   Q4_K_M-none.gguf,
                                                   Q4_K_M-hybrid_custom.gguf,
                                                   results.csv (3 rows)
    2      stage 1                                 imatrix-wiki.gguf,
                                                   imatrix-hybrid_wiki.gguf,
                                                   Q4_K_M-stock_{custom,wiki}.gguf,
                                                   Q4_K_M-hybrid_wiki.gguf,
                                                   results.csv (6 rows)
    3      stage 1                                 corpus.mixed.txt,
                                                   imatrix-mixed.gguf,
                                                   imatrix-hybrid_mixed.gguf,
                                                   Q4_K_M-stock_mixed.gguf,
                                                   Q4_K_M-hybrid_mixed.gguf,
                                                   results.csv (8 rows)
    4      logtrain holdout + test slices          eval_pool_sessions.jsonl,
                                                   toolcall_holdout.jsonl (n=25)
    5      stage 1-3 GGUFs                         results.csv with refreshed
                                                   prefill/decode/TTFT + stdev
                                                   (one model at a time, 10 reps)
    6      stage 4 holdout + stage 1-3 GGUFs       toolcall_reps_results.csv,
                                                   toolcall_reps_aggregated.csv
                                                   (10 reps × 8 models)
    7      stage 5 + stage 6                       LEADERBOARD.md, README updated

Stage 6 is by far the longest (~14 h at default 10 reps × 8 models × n=25).
Pass --skip-toolcall to leave it out; pass --quick-toolcall to do 2 reps
instead of 10 for a smoke-test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log
from quant_tuner.experiments import phase as stage

WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"
SCRIPTS = REPO / "scripts"
RESULTS_CSV = WORK / "results.csv"


def run(cmd: list[str], log_path: Path | None = None) -> None:
    """Run a subprocess, redirecting output to ``log_path`` if given. Raises on non-zero exit."""
    log(f"  running: {' '.join(str(c) for c in cmd[:6])}…")
    if log_path is None:
        rc = subprocess.run(cmd).returncode
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}")


# ---------------------------------------------------------------------------
# Stage 4: tool-call holdout
# ---------------------------------------------------------------------------

def build_toolcall_holdout(force: bool = False) -> None:
    """Combine test + holdout splits into a 25-session eval pool, then sample n=25."""
    from quant_tuner.data.ingest import filter_sessions, load_sessions
    from quant_tuner.data.split import split_sessions

    pool = WORK / "eval_pool_sessions.jsonl"
    holdout = WORK / "toolcall_holdout.jsonl"

    if holdout.exists() and not force:
        log(f"  ✓ {holdout.name} already present")
        return

    sessions = load_sessions(REPO / "logtrain.jsonl")
    sessions = filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split_sessions(sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42)
    combined = splits["test"] + splits["holdout"]
    log(f"  eval pool: {len(combined)} sessions ({dict(Counter(s.get('source') for s in combined))})")
    with pool.open("w") as f:
        for s in combined:
            f.write(json.dumps(s) + "\n")

    cmd = [
        sys.executable, str(SCRIPTS / "build_toolcall_holdout.py"),
        "--input", str(pool),
        "--output", str(holdout),
        "--no-require-anchor",
        "--n", "25",
        "--min-tool-calls", "2",
        "--min-score", "0.5",
    ]
    run(cmd, log_path=LOGS / "build_toolcall_holdout.log")


# ---------------------------------------------------------------------------
# Stage 6 toggle: override REPETITIONS in scripts/run_toolcall_reps.py
# ---------------------------------------------------------------------------

def run_toolcall_reps(reps: int) -> None:
    """Run scripts/run_toolcall_reps.py with REPETITIONS overridden via env."""
    cmd = [sys.executable, "-c",
           f"import scripts.run_toolcall_reps as m; m.REPETITIONS={reps}; m.main()"]
    run(cmd, log_path=LOGS / f"run_toolcall_reps_n{reps}.log")


# ---------------------------------------------------------------------------
# Stage 7: render leaderboard markdown
# ---------------------------------------------------------------------------

def render_leaderboard(toolcall_reps_csv: Path | None) -> None:
    """Render LEADERBOARD.md from results.csv, merging in toolcall data if available.

    If `toolcall_reps_csv` exists, use it (multi-rep mean values). Otherwise fall
    back to single-rep `toolcall_results.csv` if that exists.
    """
    from quant_tuner.leaderboard.aggregate import aggregate

    # Pick whichever toolcall data is present, preferring the multi-rep one.
    legacy_single = WORK / "toolcall_results.csv"
    tc = None
    if toolcall_reps_csv and toolcall_reps_csv.exists():
        tc = toolcall_reps_csv
        log(f"  using tool-call source: {tc.name} (multi-rep)")
    elif legacy_single.exists():
        tc = legacy_single
        log(f"  using tool-call source: {tc.name} (single-rep)")
    else:
        log("  no tool-call data — leaderboard will omit those columns")

    md = aggregate(
        RESULTS_CSV,
        weights=(1.0, 2.0, 1.0),
        sort_by="mean_kld",
        toolcall_csv=tc,
    )
    (WORK / "LEADERBOARD.md").write_text(md)
    log(f"  wrote {WORK / 'LEADERBOARD.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-toolcall", action="store_true",
                   help="Skip the multi-rep tool-call benchmark (saves ~14 h)")
    p.add_argument("--quick-toolcall", action="store_true",
                   help="Run only 2 reps × 8 models for tool-call (smoke test, ~3 h)")
    p.add_argument("--skip-speed-rebench", action="store_true",
                   help="Skip the 10-rep speed rebench (keep speed numbers from initial run)")
    args = p.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    with stage("stage 1: foundation + custom imatrix (~25 min cold)"):
        run([sys.executable, str(SCRIPTS / "run_omnicoder_q4_k_m.py")],
            log_path=LOGS / "reproduce_stage1.log")

    with stage("stage 2: wiki imatrix + 3 quants (~15 min cold)"):
        run([sys.executable, str(SCRIPTS / "run_omnicoder_wiki_vs_custom.py")],
            log_path=LOGS / "reproduce_stage2.log")

    with stage("stage 3: mixed-corpus imatrix + 2 quants (~22 min cold)"):
        run([sys.executable, str(SCRIPTS / "run_omnicoder_mixed_corpus.py")],
            log_path=LOGS / "reproduce_stage3.log")

    with stage("stage 4: build n=25 tool-call holdout"):
        build_toolcall_holdout()

    if not args.skip_speed_rebench:
        with stage("stage 5: 10-rep speed rebench × 8 (one model at a time, ~10 min)"):
            run([sys.executable, str(SCRIPTS / "rebench_speed.py")],
                log_path=LOGS / "reproduce_stage5.log")
    else:
        log("[skip] stage 5: --skip-speed-rebench")

    if args.skip_toolcall:
        log("[skip] stage 6: --skip-toolcall")
        tc_csv = None
    else:
        reps = 2 if args.quick_toolcall else 10
        with stage(f"stage 6: {reps}-rep × 8 tool-call (~{reps * 1.5:.0f} h)"):
            run_toolcall_reps(reps)
        tc_csv = WORK / "toolcall_reps_aggregated.csv"

    with stage("stage 7: render LEADERBOARD.md"):
        render_leaderboard(tc_csv)

    total_h = (time.time() - t_total) / 3600
    log(f"=== ALL DONE in {total_h:.2f} h ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run toolcall + MMLU-Pro evals across all MTP variants for each 9B model.

Evaluates 9 combinations (3 models × 3 MTP variants) at IQ3_S with the
hybrid_custom imatrix+ calibration, producing per-rep and aggregated CSVs
plus a Markdown snippet ready to paste into EXPERIMENT.md.

GGUF mapping per slug/variant:
  vanilla       → IQ3_S-custom.gguf       (donor Qwen parent MTP)
  trained       → IQ3_S-trained.gguf      (custom-trained MTP)
  trained-noisy → IQ3_S-trained-noisy.gguf (custom-trained + IQ3_S noise)

Usage::

    # Full suite, 5 reps each
    uv run python scripts/run_mtp_eval_suite.py \\
        --holdout artifacts/toolcall_holdout.jsonl \\
        --mmlu-holdout out/mmlu_pro/holdout.json

    # Toolcall only, single model
    uv run python scripts/run_mtp_eval_suite.py \\
        --holdout artifacts/toolcall_holdout.jsonl \\
        --models tesslate --skip-mmlu

    # Dry run
    uv run python scripts/run_mtp_eval_suite.py \\
        --holdout artifacts/toolcall_holdout.jsonl \\
        --mmlu-holdout out/mmlu_pro/holdout.json \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase  # noqa: E402

WORKSPACE = REPO / "out" / "benchmark_9b_iq3s"

GGUF_NAMES = {
    "vanilla":        "IQ3_S-custom.gguf",
    "trained":        "IQ3_S-trained.gguf",
    "trained-noisy":  "IQ3_S-trained-noisy.gguf",
    "trained-mixed":  "IQ3_S-trained-mixed.gguf",
    "trained-iq3s":   "IQ3_S-trained-iq3s.gguf",
}

TOOLCALL_SAMPLING = dict(
    temperature=0.6, top_p=0.95, top_k=20,
    min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
    max_tokens=512,
)

MMLU_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gguf(workspace: Path, slug: str, variant: str) -> Path:
    return workspace / slug / GGUF_NAMES[variant]


def _run_toolcall(
    gguf: Path,
    holdout: Path,
    out_dir: Path,
    label: str,
    reps: int,
    dry_run: bool,
) -> Path:
    """Run toolcall reps for one GGUF; return path to aggregated CSV."""
    per_rep    = out_dir / f"toolcall_{label}_reps.csv"
    aggregated = out_dir / f"toolcall_{label}_agg.csv"
    log_dir    = out_dir / "logs"

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/run_toolcall_reps.py"),
        "--models",     str(gguf),
        "--holdout",    str(holdout),
        "--reps",       str(reps),
        "--results",    str(per_rep),
        "--aggregated", str(aggregated),
        "--log-dir",    str(log_dir),
    ]
    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        return aggregated

    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / f"toolcall_{label}.log"
    log(f"  toolcall reps → {run_log}")
    with open(run_log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    while proc.poll() is None:
        time.sleep(60)
        try:
            lines = run_log.read_text().splitlines()
            recent = [l for l in lines[-10:] if "rep" in l or "aggregate" in l or "DONE" in l]
            if recent:
                log(f"    {recent[-1].strip()}")
        except Exception:
            pass
    if proc.returncode != 0:
        raise RuntimeError(f"toolcall reps failed for {label} (exit {proc.returncode})")
    return aggregated


def _run_mmlu(
    gguf: Path,
    holdout: Path,
    out_dir: Path,
    label: str,
    reps: int,
    dry_run: bool,
) -> Path:
    """Run MMLU-Pro reps for one GGUF; return path to aggregated CSV."""
    per_rep    = out_dir / f"mmlu_{label}_reps.csv"
    aggregated = out_dir / f"mmlu_{label}_agg.csv"
    log_dir    = out_dir / "logs"

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/run_mmlu_pro_reps.py"),
        "--models",     str(gguf),
        "--holdout",    str(holdout),
        "--reps",       str(reps),
        "--results",    str(per_rep),
        "--aggregated", str(aggregated),
        "--log-dir",    str(log_dir),
    ]
    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        return aggregated

    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / f"mmlu_{label}.log"
    log(f"  mmlu-pro reps → {run_log}")
    with open(run_log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    while proc.poll() is None:
        time.sleep(60)
        try:
            lines = run_log.read_text().splitlines()
            recent = [l for l in lines[-10:] if "rep" in l or "aggregate" in l or "DONE" in l]
            if recent:
                log(f"    {recent[-1].strip()}")
        except Exception:
            pass
    if proc.returncode != 0:
        raise RuntimeError(f"mmlu reps failed for {label} (exit {proc.returncode})")
    return aggregated


def _read_agg_row(csv_path: Path, key: str) -> dict[str, float]:
    """Read the first data row from an aggregated CSV; return {metric_mean: value}."""
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[0]
    return {k: float(v) for k, v in row.items() if v not in ("", None) and k != "model"}


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def _render_general_table(summary: list[dict]) -> str:
    """Format results as EXPERIMENT.md General Performance rows."""
    lines = [
        "| Model | Variant | Tool Sel % | Param Acc % | Schema % | Rollout % | MMLU-Pro % |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary:
        sel   = r.get("tool_selection_acc_mean",        float("nan"))
        param = r.get("param_acc_mean_mean",             float("nan"))
        sch   = r.get("schema_valid_rate_mean",          float("nan"))
        roll  = r.get("rollout_complete_rate_mean",      float("nan"))
        mmlu  = r.get("accuracy_mean",                   float("nan"))

        def fmt(v: float) -> str:
            return f"{v*100:.1f}" if v == v else "—"

        lines.append(
            f"| {r['slug']} | {r['variant']} "
            f"| {fmt(sel)} | {fmt(param)} | {fmt(sch)} | {fmt(roll)} | {fmt(mmlu)} |"
        )
    return "\n".join(lines)


def _render_mtp_tables(mtp_json: Path, n_max_values: list[int]) -> str:
    """Format mtp_acceptance.json as EXPERIMENT.md MTP tables."""
    if not mtp_json.exists():
        return "(mtp_acceptance.json not found — run run_mtp_pipeline.py first)"

    data = json.loads(mtp_json.read_text())

    acc_header = "| Model | Variant | " + " | ".join(f"n={n} acc%" for n in n_max_values) + " |"
    tps_header = "| Model | Variant | " + " | ".join(f"n={n} tok/s" for n in n_max_values) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in n_max_values) + " |"

    acc_lines = [acc_header, sep]
    tps_lines = [tps_header, sep]

    for row in data:
        slug    = row["slug"]
        variant = row["variant"]
        acc_cols = []
        tps_cols = []
        for n in n_max_values:
            match = next((r for r in row.get("sweep", []) if r["n_max"] == n), None)
            if match and match.get("accept_rate") is not None:
                acc_cols.append(f"{match['accept_rate']*100:.1f}")
                tps_cols.append(f"{match['mean_tps']:.1f}")
            else:
                acc_cols.append("—")
                tps_cols.append("—")
        acc_lines.append(f"| {slug} | {variant} | " + " | ".join(acc_cols) + " |")
        tps_lines.append(f"| {slug} | {variant} | " + " | ".join(tps_cols) + " |")

    return (
        "MTP Performance Acceptance:\n"
        + "\n".join(acc_lines)
        + "\n\nMTP Throughput:\n"
        + "\n".join(tps_lines)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace",    type=Path, default=WORKSPACE)
    parser.add_argument("--models",       type=str,  default="qwen,jackrong,tesslate")
    parser.add_argument("--variants",     type=str,  default="vanilla,trained,trained-noisy")
    parser.add_argument("--holdout",      type=Path, required=False,
                        default=REPO / "artifacts/toolcall_holdout.jsonl",
                        help="Toolcall holdout JSONL (25 sessions)")
    parser.add_argument("--mmlu-holdout", type=Path, required=False,
                        default=REPO / "out/mmlu_pro/holdout.json",
                        help="MMLU-Pro holdout JSON")
    parser.add_argument("--reps",         type=int,  default=5)
    parser.add_argument("--skip-toolcall", action="store_true")
    parser.add_argument("--skip-mmlu",     action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    slugs    = [s.strip() for s in args.models.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    out_dir  = args.workspace / "eval" / "mtp_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log(f"MTP eval suite: {len(slugs)} models × {len(variants)} variants")
    log(f"workspace:      {args.workspace}")
    log(f"reps:           {args.reps}")
    log("=" * 60)

    # Validate GGUFs before starting
    missing = []
    for slug in slugs:
        for variant in variants:
            p = _gguf(args.workspace, slug, variant)
            status = "✓" if p.exists() else "MISSING"
            log(f"  {slug:10s} / {variant:14s}  {status}  ({p.name})")
            if not p.exists():
                missing.append(str(p))

    if missing and not args.dry_run:
        log(f"\nERROR: {len(missing)} GGUFs missing — run run_mtp_pipeline.py first:")
        for m in missing:
            log(f"  {m}")
        return 1

    summary: list[dict] = []

    for slug in slugs:
        for variant in variants:
            label = f"{slug}_{variant.replace('-', '_')}"
            g     = _gguf(args.workspace, slug, variant)

            log(f"\n{'='*60}")
            log(f"  {slug} / {variant}")
            log(f"{'='*60}")

            row: dict = {"slug": slug, "variant": variant}

            if not args.skip_toolcall:
                if not args.holdout.exists() and not args.dry_run:
                    log(f"  WARNING: holdout not found at {args.holdout} — skipping toolcall")
                else:
                    with phase(f"{label} toolcall reps"):
                        agg = _run_toolcall(g, args.holdout, out_dir, label,
                                            args.reps, args.dry_run)
                    row.update(_read_agg_row(agg, label))

            if not args.skip_mmlu:
                if not args.mmlu_holdout.exists() and not args.dry_run:
                    log(f"  WARNING: mmlu holdout not found at {args.mmlu_holdout} — skipping mmlu")
                else:
                    with phase(f"{label} mmlu-pro reps"):
                        agg = _run_mmlu(g, args.mmlu_holdout, out_dir, label,
                                        args.reps, args.dry_run)
                    row.update(_read_agg_row(agg, label))

            summary.append(row)

    # Write summary JSON
    summary_json = out_dir / "mtp_eval_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {summary_json}")

    # Render and print table snippets
    log("\n" + "=" * 60)
    log("GENERAL PERFORMANCE TABLE (paste into EXPERIMENT.md):")
    log("=" * 60)
    print(_render_general_table(summary))

    mtp_json = args.workspace / "mtp_acceptance.json"
    log("\n" + "=" * 60)
    log("MTP ACCEPTANCE + THROUGHPUT TABLES (paste into EXPERIMENT.md):")
    log("=" * 60)
    print(_render_mtp_tables(mtp_json, [1, 2, 3]))

    return 0


if __name__ == "__main__":
    sys.exit(main())

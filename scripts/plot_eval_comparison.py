"""Plot toolcall + MMLU-Pro comparison across (model, MTP variant) pairs.

Reads the aggregated CSVs from out/benchmark_9b_iq3s/eval/mtp_suite/ and
renders a grouped bar chart per metric so every variant for a given model
sits side-by-side. Missing combinations show up as blanks rather than
errors, so this can be re-run as the eval suite fills in.

Usage::

    uv run python scripts/plot_eval_comparison.py
    uv run python scripts/plot_eval_comparison.py --out out/.../eval_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "out" / "benchmark_9b_iq3s"
SUITE = WORKSPACE / "eval" / "mtp_suite"

MODELS = ["qwen", "jackrong", "tesslate"]
VARIANTS = ["vanilla", "trained", "trained-noisy", "trained-mixed", "trained-iq3s"]
COLORS = {
    "vanilla":        "#6c757d",
    "trained":        "#4e79a7",
    "trained-noisy":  "#e15759",
    "trained-mixed":  "#f28e2b",
    "trained-iq3s":   "#59a14f",
}

TOOLCALL_METRICS = [
    ("tool_selection_acc_mean",        "Tool selection %"),
    ("param_acc_mean_mean",            "Param accuracy %"),
    ("schema_valid_rate_mean",         "Schema valid %"),
    ("rollout_complete_rate_mean",     "Rollout complete %"),
]
MMLU_METRIC = ("accuracy_mean", "MMLU-Pro %")


def _read_agg(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def _collect(workspace: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for slug in MODELS:
        for variant in VARIANTS:
            label = f"{slug}_{variant.replace('-', '_')}"
            row: dict = {}
            tc = _read_agg(SUITE / f"toolcall_{label}_agg.csv")
            mm = _read_agg(SUITE / f"mmlu_{label}_agg.csv")
            if tc:
                for key, _ in TOOLCALL_METRICS:
                    try:
                        row[key] = float(tc[key])
                    except (KeyError, ValueError):
                        pass
            if mm:
                try:
                    row[MMLU_METRIC[0]] = float(mm[MMLU_METRIC[0]])
                except (KeyError, ValueError):
                    pass
            if row:
                out[(slug, variant)] = row
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, default=WORKSPACE)
    p.add_argument("--out", type=Path, default=WORKSPACE / "eval_comparison.png")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib + numpy required", file=sys.stderr)
        return 1

    data = _collect(args.workspace)
    if not data:
        print("no aggregated CSVs found", file=sys.stderr)
        return 1

    metrics = TOOLCALL_METRICS + [MMLU_METRIC]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 14), sharex=True)

    x = np.arange(len(MODELS))
    width = 0.85 / len(VARIANTS)

    for ax, (key, title) in zip(axes, metrics):
        for vi, variant in enumerate(VARIANTS):
            vals = []
            for slug in MODELS:
                v = data.get((slug, variant), {}).get(key)
                vals.append(v * 100 if v is not None else 0.0)
            present = [data.get((slug, variant), {}).get(key) is not None for slug in MODELS]
            bars = ax.bar(
                x + (vi - (len(VARIANTS) - 1) / 2) * width, vals, width,
                color=COLORS[variant], label=variant if ax is axes[0] else None,
                edgecolor="black", linewidth=0.4,
            )
            for bar, ok, val in zip(bars, present, vals):
                if ok:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)

    axes[0].legend(loc="upper right", fontsize=9, ncol=len(VARIANTS))
    fig.suptitle("9B / IQ3_S — MTP-variant comparison (toolcall + MMLU-Pro)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

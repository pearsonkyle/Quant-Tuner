"""Correlate per-rep task accuracy (MMLU-Pro + tool-call) with per-cell
quant-quality metrics on Jackrong/Qwopus3.5-9B-Coder.

Produces a 3×3 grid of scatter plots:
  rows    = task metric (MMLU avg, tool_selection_acc, schema_valid_rate)
  columns = quant predictor (PPL, KLD, same_top_p)

Per-cell quant metrics come from exp-008/exp-001 results.csv. Per-rep
task accuracies come from out/mmlu_pro/jackrong/reps_per.csv and
out/toolcall/jackrong/reps_per.csv.

F16 is excluded from regression for KLD/same_top_p panels (anchor
values); it's plotted for context.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MMLU_CSV = REPO / "out" / "mmlu_pro" / "jackrong" / "reps_per.csv"
TC_CSV = REPO / "out" / "toolcall" / "jackrong" / "reps_per.csv"
OUT_PNG = REPO / "out" / "mmlu_pro" / "jackrong" / "correlation_grid.png"

CELL_METRICS: dict[str, dict[str, float]] = {
    "Q4_K_M-wiki.gguf":     {"ppl": 2.9200, "kld": 0.53350, "stp": 90.4180},
    "Q4_K_M-custom.gguf":   {"ppl": 3.3549, "kld": 0.51300, "stp": 90.4960},
    "Q4_K_M-mixed512.gguf": {"ppl": 3.0949, "kld": 0.50676, "stp": 90.5450},
    "Q4_K_M-mixed2k.gguf":  {"ppl": 3.1488, "kld": 0.52010, "stp": 90.4080},
    "Q4_K_M-mixed8k.gguf":  {"ppl": 3.1578, "kld": 0.51996, "stp": 90.3440},
    "model-f16.gguf":       {"ppl": 3.8035, "kld": 0.0,     "stp": 100.0},
}
CELL_STYLES = {
    "Q4_K_M-wiki.gguf":     ("wiki",     "tab:blue"),
    "Q4_K_M-custom.gguf":   ("custom",   "tab:orange"),
    "Q4_K_M-mixed512.gguf": ("mixed512", "tab:green"),
    "Q4_K_M-mixed2k.gguf":  ("mixed2k",  "tab:purple"),
    "Q4_K_M-mixed8k.gguf":  ("mixed8k",  "tab:brown"),
}
PREDICTORS = [
    ("ppl", "PPL on eval set"),
    ("kld", "Mean KLD vs F16"),
    ("stp", "same_top_p vs F16 (%)"),
]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(x)).astype(float),
                   np.argsort(np.argsort(y)).astype(float))


def load_reps(csv_path: Path, acc_col: str) -> list[dict]:
    out: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m not in CELL_METRICS:
                continue
            try:
                acc = float(row[acc_col])
            except (KeyError, ValueError):
                continue
            out.append({"model": m, "acc": acc, **CELL_METRICS[m],
                        "is_f16": m == "model-f16.gguf"})
    return out


def main() -> int:
    if not MMLU_CSV.exists() or not TC_CSV.exists():
        print(f"missing input csv (need {MMLU_CSV} and {TC_CSV})", file=sys.stderr)
        return 1

    tasks = [
        ("MMLU-Pro Avg",     load_reps(MMLU_CSV, "accuracy")),
        ("tool_selection_acc", load_reps(TC_CSV, "tool_selection_acc")),
        ("schema_valid_rate",  load_reps(TC_CSV, "schema_valid_rate")),
    ]

    fig, axes = plt.subplots(len(tasks), len(PREDICTORS),
                             figsize=(16, 4.6 * len(tasks)),
                             sharex="col")
    summary_rows: list[tuple] = []

    for r, (task_name, pts) in enumerate(tasks):
        quant_pts = [p for p in pts if not p["is_f16"]]
        f16_pts = [p for p in pts if p["is_f16"]]
        y = np.array([p["acc"] for p in quant_pts])

        for c, (key, xlabel) in enumerate(PREDICTORS):
            ax = axes[r, c]
            x = np.array([p[key] for p in quant_pts])
            r_p = pearson(x, y); r_s = spearman(x, y)
            summary_rows.append((task_name, key, r_p, r_s))

            for model, (label, color) in CELL_STYLES.items():
                xs = [p[key] for p in quant_pts if p["model"] == model]
                ys = [p["acc"] for p in quant_pts if p["model"] == model]
                if xs:
                    ax.scatter(xs, ys, s=60, alpha=0.75, color=color,
                               edgecolors="black", linewidths=0.4,
                               label=label if r == 0 and c == 0 else None)

            if f16_pts:
                xf = [p[key] for p in f16_pts]
                yf = [p["acc"] for p in f16_pts]
                ax.scatter(xf, yf, s=90, marker="*", color="tab:red",
                           edgecolors="black", linewidths=0.4,
                           label="FP16 (excl.)" if r == 0 and c == 0 else None)

            if len(np.unique(x)) > 1:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 50)
                ax.plot(xline, m * xline + b, color="gray", linestyle="--",
                        alpha=0.7)

            ax.set_title(f"r={r_p:+.3f}  ρ={r_s:+.3f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{task_name}\n(per rep)", fontsize=10)
            if r == len(tasks) - 1:
                ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)

    # One legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)

    fig.suptitle("Jackrong/Qwopus3.5-9B-Coder — task accuracy vs quant-quality "
                 "(per-rep, 5 reps × 5 cells)", y=1.02, fontsize=12)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"plot -> {OUT_PNG.relative_to(REPO)}\n")

    print(f"{'task':<22} {'predictor':<6} {'Pearson r':>12} {'Spearman ρ':>14}")
    print("-" * 58)
    for task, pred, rp, rs in summary_rows:
        print(f"{task:<22} {pred:<6} {rp:>+12.4f} {rs:>+14.4f}")

    # Per-task winners
    print("\nStrongest |correlation| per task:")
    by_task: dict[str, list] = {}
    for task, pred, rp, rs in summary_rows:
        by_task.setdefault(task, []).append((pred, rp, rs))
    for task, rows in by_task.items():
        bp = max(rows, key=lambda r: abs(r[1]))
        bs = max(rows, key=lambda r: abs(r[2]))
        print(f"  {task:<22} Pearson: {bp[0]} ({bp[1]:+.3f})  "
              f"Spearman: {bs[0]} ({bs[2]:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

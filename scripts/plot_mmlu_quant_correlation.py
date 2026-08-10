"""Correlate per-rep MMLU-Pro accuracy with per-cell quant-quality metrics.

Reads `out/mmlu_pro/jackrong/reps_per.csv` (one row per (model, rep)),
joins per-cell PPL / KLD / same_top_p from exp-001 results, then
produces a 3-panel scatter (one panel per predictor) with the best-fit
line and prints Pearson + Spearman correlations.

F16 is excluded from the regression because its KLD=0 and
same_top_p=100 are by-definition values, not measurements — including
it anchors the correlation artificially. We still plot it (as a
separate marker) so it's visible.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
REPS_CSV = REPO / "out" / "mmlu_pro" / "jackrong" / "reps_per.csv"
OUT_PNG = REPO / "out" / "mmlu_pro" / "jackrong" / "correlation.png"

# Per-cell quant-quality (from exp-008 README / out/exp-001 results.csv)
CELL_METRICS: dict[str, dict[str, float]] = {
    "Q4_K_M-wiki.gguf":     {"ppl": 2.9200, "kld": 0.53350, "stp": 90.4180},
    "Q4_K_M-custom.gguf":   {"ppl": 3.3549, "kld": 0.51300, "stp": 90.4960},
    "Q4_K_M-mixed512.gguf": {"ppl": 3.0949, "kld": 0.50676, "stp": 90.5450},
    "Q4_K_M-mixed2k.gguf":  {"ppl": 3.1488, "kld": 0.52010, "stp": 90.4080},
    "Q4_K_M-mixed8k.gguf":  {"ppl": 3.1578, "kld": 0.51996, "stp": 90.3440},
    "model-f16.gguf":       {"ppl": 3.8035, "kld": 0.0,     "stp": 100.0},
}

PREDICTORS = [
    ("ppl", "PPL on eval set (lower = lower avg log-loss)"),
    ("kld", "Mean KLD vs F16 (lower = closer to F16)"),
    ("stp", "same_top_p vs F16, % (higher = closer to F16)"),
]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(x)).astype(float),
                   np.argsort(np.argsort(y)).astype(float))


def main() -> int:
    if not REPS_CSV.exists():
        print(f"missing {REPS_CSV}", file=sys.stderr)
        return 1

    points: list[dict] = []
    with REPS_CSV.open() as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in CELL_METRICS:
                continue
            try:
                acc = float(row["accuracy"])
            except (KeyError, ValueError):
                continue
            entry = {"model": model, "rep": int(row["rep"]), "acc": acc,
                     **CELL_METRICS[model]}
            entry["is_f16"] = (model == "model-f16.gguf")
            points.append(entry)

    quant_pts = [p for p in points if not p["is_f16"]]
    f16_pts = [p for p in points if p["is_f16"]]
    if not quant_pts:
        print("no quant rows found in reps CSV", file=sys.stderr)
        return 1

    print(f"loaded {len(quant_pts)} quant points + {len(f16_pts)} F16 points "
          f"across {len({p['model'] for p in quant_pts})} cells\n")

    y = np.array([p["acc"] for p in quant_pts])

    # Per-cell color + short label for the legend
    cell_styles = {
        "Q4_K_M-wiki.gguf":     ("wiki",     "tab:blue"),
        "Q4_K_M-custom.gguf":   ("custom",   "tab:orange"),
        "Q4_K_M-mixed512.gguf": ("mixed512", "tab:green"),
        "Q4_K_M-mixed2k.gguf":  ("mixed2k",  "tab:purple"),
        "Q4_K_M-mixed8k.gguf":  ("mixed8k",  "tab:brown"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    rows = []
    for ax, (key, xlabel) in zip(axes, PREDICTORS):
        x = np.array([p[key] for p in quant_pts])
        r_pearson = pearson(x, y)
        r_spearman = spearman(x, y)
        rows.append((key, r_pearson, r_spearman))

        # Scatter per cell so reps from the same dataset are color-coded
        for model, (label, color) in cell_styles.items():
            xs = [p[key] for p in quant_pts if p["model"] == model]
            ys = [p["acc"] for p in quant_pts if p["model"] == model]
            if not xs:
                continue
            ax.scatter(xs, ys, s=70, alpha=0.75, color=color,
                       edgecolors="black", linewidths=0.5, label=label)

        # F16 plotted but excluded from regression
        if f16_pts:
            xf = np.array([p[key] for p in f16_pts])
            yf = np.array([p["acc"] for p in f16_pts])
            ax.scatter(xf, yf, s=110, marker="*", color="tab:red",
                       edgecolors="black", linewidths=0.5,
                       label="FP16 (excluded)")

        # Best-fit line over quant points only
        if len(np.unique(x)) > 1:
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 50)
            ax.plot(xline, m * xline + b, color="gray", linestyle="--",
                    alpha=0.7,
                    label=f"fit  r={r_pearson:+.3f}  ρ={r_spearman:+.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("MMLU-Pro accuracy (per rep)")
        ax.set_title(f"{key.upper()} vs MMLU")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, framealpha=0.9)

    fig.suptitle("Jackrong/Qwopus3.5-9B-Coder — quant-quality vs MMLU-Pro "
                 "(5 reps × 5 Q4_K_M cells)", fontsize=12)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"plot -> {OUT_PNG.relative_to(REPO)}")

    print("\nCorrelations (Q4_K_M cells only, F16 excluded):")
    print(f"  {'predictor':<14} {'Pearson r':>12} {'Spearman ρ':>14}")
    for key, rp, rs in rows:
        print(f"  {key:<14} {rp:>+12.4f} {rs:>+14.4f}")

    best_pearson = max(rows, key=lambda r: abs(r[1]))
    best_spearman = max(rows, key=lambda r: abs(r[2]))
    print(f"\n  strongest |Pearson|:  {best_pearson[0]} ({best_pearson[1]:+.3f})")
    print(f"  strongest |Spearman|: {best_spearman[0]} ({best_spearman[2]:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

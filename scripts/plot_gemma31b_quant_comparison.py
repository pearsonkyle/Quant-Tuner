"""Plot Gemma-4-31B-it quant sweep: PPL, KLD, same_top_p vs size (GiB).

Three subplots, one per metric. One line per technique:
  - plain                    (black)         exp-011
  - imatrix                  (blue)          exp-009
  - awq+imatrix (naive)      (green)         exp-010
  - awq+outproj+imatrix      (orange)        exp-014
  - awq-q2kproxy+imatrix     (red)           exp-015
  - awq-pertensor+imatrix    (purple)        exp-016

Sources whose results.csv does not yet exist are silently skipped, so this
script can be run incrementally as experiments complete. Each point is
labelled with its quant name on the PPL plot only (others kept clean). PPL
uses a log scale because sub-3-bpw plain quants span 4 orders of magnitude.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SLUG = "google__gemma-4-31B-it"

SOURCES = {
    "plain":             (REPO / "out" / "exp-011" / SLUG / "results.csv", "|plain|",                    "black"),
    "imatrix":           (REPO / "out" / "exp-009" / SLUG / "results.csv", "|imatrix|",                  "tab:blue"),
    "awq (naive)":       (REPO / "out" / "exp-010" / SLUG / "results.csv", "|awq+imatrix|",              "tab:green"),
    "awq + o/down_proj": (REPO / "out" / "exp-014" / SLUG / "results.csv", "|awq+outproj+imatrix|",      "tab:orange"),
    "awq q2k-proxy":     (REPO / "out" / "exp-015" / SLUG / "results.csv", "|awq-q2kproxy+imatrix|",     "tab:red"),
    "awq per-tensor α":  (REPO / "out" / "exp-016" / SLUG / "results.csv", "|awq-pertensor+imatrix|",    "tab:purple"),
    "awq cv-gate":       (REPO / "out" / "exp-017" / SLUG / "results.csv", "|awq-cv-gate+imatrix|",      "tab:brown"),
    "awq cv-mixed":      (REPO / "out" / "exp-018" / SLUG / "results.csv", "|awq-cv-mixed+imatrix|",     "tab:pink"),
}

OUT = REPO / "out" / "figures" / "gemma31b_quant_comparison.png"


def load(csv_path: Path, key: str):
    points = []
    if not csv_path.exists():
        return points
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if key not in row["model"]:
                continue
            quant = row["model"].split("|")[1]
            try:
                points.append((
                    quant,
                    float(row["size_gib"]),
                    float(row["ppl"]),
                    float(row["mean_kld"]),
                    float(row["same_top_p"]),
                ))
            except (KeyError, ValueError):
                continue
    points.sort(key=lambda r: r[1])
    return points


def main() -> int:
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    ax_ppl, ax_kld, ax_top = axes

    for label, (csv_path, key, color) in SOURCES.items():
        pts = load(csv_path, key)
        if not pts:
            print(f"WARN: no points for {label}", file=sys.stderr)
            continue
        names = [p[0] for p in pts]
        sizes = [p[1] for p in pts]
        ppls = [p[2] for p in pts]
        klds = [p[3] for p in pts]
        tops = [p[4] for p in pts]
        ax_ppl.plot(sizes, ppls, "-o", color=color, label=label, linewidth=2, markersize=6)
        ax_kld.plot(sizes, klds, "-o", color=color, label=label, linewidth=2, markersize=6)
        ax_top.plot(sizes, tops, "-o", color=color, label=label, linewidth=2, markersize=6)
        # Annotate points with quant names (only on the PPL plot to keep the others clean).
        for name, x, y in zip(names, sizes, ppls):
            ax_ppl.annotate(
                name, (x, y), textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=color, alpha=0.85,
            )

    # FP16 reference lines (computed from exp-009 baseline; same fp16 across all three).
    fp16_size = 57.20
    fp16_ppl = 302.26
    for ax in (ax_ppl,):
        ax.axhline(fp16_ppl, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label=f"FP16 PPL = {fp16_ppl:.1f}")
    ax_top.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="FP16 top_p = 100")
    ax_kld.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="FP16 KLD = 0")

    ax_ppl.set_yscale("log")
    ax_ppl.set_ylabel("Perplexity (log scale, lower is better)")
    ax_ppl.set_title("Gemma-4-31B-it: quantization technique comparison")
    ax_ppl.grid(True, which="both", alpha=0.3)
    ax_ppl.legend(loc="upper right", fontsize=9)

    ax_kld.set_ylabel("Mean KLD vs FP16 (lower is better)")
    ax_kld.grid(True, alpha=0.3)
    ax_kld.legend(loc="upper right", fontsize=9)

    ax_top.set_ylabel("same_top_p (%, higher is better)")
    ax_top.set_xlabel("File size (GiB)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

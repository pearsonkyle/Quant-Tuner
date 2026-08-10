"""Plot exp-019 quants only: PPL / KLD / same_top_p vs file size.

Three vertically-stacked subplots, one per metric. Lines:
  - awq-cv-gate  (exp-019, clean corpora)   tab:blue
  - awq-cv-mixed (exp-019, clean corpora)   tab:orange
  - plain Q2_K   (exp-019, no imatrix)      tab:gray (single point)

FP16 reference lines (PPL=277.89 on new eval corpus, KLD=0, same_top_p=100)
drawn for context. PPL uses a log scale — plain Q2_K is ~24× the calibrated
quants. Each point annotated with its quant name on the PPL plot.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SLUG = "google__gemma-4-31B-it"
EXP19 = REPO / "out" / "exp-019" / SLUG

# (label, csv_path, color, marker)
SOURCES = [
    ("awq-cv-gate (exp-019)",  EXP19 / "gate"  / "results.csv", "tab:blue",   "o"),
    ("awq-cv-mixed (exp-019)", EXP19 / "mixed" / "results.csv", "tab:orange", "s"),
    ("plain Q2_K (no imatrix)", EXP19 / "plain" / "results.csv", "tab:gray",   "D"),
]

# FP16 anchor (computed by exp-019 baseline.kld on the new code+math+tools eval).
FP16_SIZE_GIB = 57.20
FP16_PPL = 277.89

OUT = REPO / "out" / "figures" / "exp019_quants.png"


def load(csv_path: Path) -> list[tuple[str, float, float, float, float]]:
    pts: list[tuple[str, float, float, float, float]] = []
    if not csv_path.exists():
        return pts
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            quant = row["model"].split("|")[1]
            try:
                pts.append((
                    quant,
                    float(row["size_gib"]),
                    float(row["ppl"]),
                    float(row["mean_kld"]),
                    float(row["same_top_p"]),
                ))
            except (KeyError, ValueError):
                continue
    pts.sort(key=lambda r: r[1])
    return pts


def main() -> int:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax_ppl, ax_kld, ax_top = axes

    for label, csv_path, color, marker in SOURCES:
        pts = load(csv_path)
        if not pts:
            print(f"WARN: no rows in {csv_path}", file=sys.stderr)
            continue
        names = [p[0] for p in pts]
        sizes = [p[1] for p in pts]
        ppls = [p[2] for p in pts]
        klds = [p[3] for p in pts]
        tops = [p[4] for p in pts]

        line_kw = dict(color=color, label=label, linewidth=2, markersize=8, marker=marker)
        # Single-point series (plain) → bigger marker, no line, dark edge.
        if len(pts) == 1:
            line_kw["linestyle"] = ""
            line_kw["markersize"] = 13
            line_kw["markeredgecolor"] = "black"
            line_kw["markeredgewidth"] = 1.2
        ax_ppl.plot(sizes, ppls, **line_kw)
        ax_kld.plot(sizes, klds, **line_kw)
        ax_top.plot(sizes, tops, **line_kw)

        for name, x, y in zip(names, sizes, ppls):
            ax_ppl.annotate(
                name, (x, y), textcoords="offset points", xytext=(7, 6),
                fontsize=8, color=color, alpha=0.9,
            )

    ax_ppl.axhline(FP16_PPL, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label=f"FP16 PPL = {FP16_PPL:.2f}")
    ax_kld.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="FP16 KLD = 0")
    ax_top.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="FP16 same_top_p = 100")

    ax_ppl.set_yscale("log")
    ax_ppl.set_ylabel("Perplexity (log scale, lower is better)")
    ax_ppl.set_title("Gemma-4-31B-it exp-019 quants (eval: code+math+tools)")
    ax_ppl.grid(True, which="both", alpha=0.3)
    ax_ppl.legend(loc="best", fontsize=9)

    ax_kld.set_ylabel("Mean KLD vs FP16 (lower is better)")
    ax_kld.grid(True, alpha=0.3)
    ax_kld.legend(loc="best", fontsize=9)

    ax_top.set_ylabel("same_top_p (%, higher is better)")
    ax_top.set_xlabel("File size (GiB)")
    ax_top.set_ylim(20, 102)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

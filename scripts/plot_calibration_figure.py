"""Figure: how 2-bit quality changes with calibration technique.

Small multiples, one metric per panel (no dual axes). x = the ordinal
progression Q2_K(none) -> imatrix -> AWQ. Two series = the two models.
Top row = static fidelity (what KLD/PPL see), bottom row = agentic ability
(what actually matters). Data from exp-051/052/053.

    PYTHONPATH=src .venv/bin/python scripts/plot_calibration_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "reddit_kld_figure.png"

X = ["Q2_K\n(none)", "IQ2_M\n(imatrix)", "IQ2_M\n(AWQ)"]
XI = [0, 1, 2]

# validated categorical slots 1 (blue) + 2 (aqua)
GEMMA = "#2a78d6"
ORNITH = "#0f8a5f"  # aqua slot darkened for line contrast on white
INK = "#0b0b0b"
MUTE = "#52514e"
GRID = "#e6e5e2"

# gemma-4-31B / Ornith-9B, per technique
D = {
    "gemma": {
        "kld": [5.21, 1.57, 1.80],
        "topp": [25.4, 46.6, 43.9],
        "tool": [0.0, 45.4, 49.2],
        "arg": [0.0, 17.1, 26.3],
        "valid": [0.0, 80.5, 82.3],
    },
    "ornith": {
        "kld": [2.03, 0.11, 0.12],
        "topp": [37.9, 80.6, 79.7],
        "tool": [2.6, 30.6, 53.6],
        "arg": [0.0, 5.4, 33.3],
        "valid": [2.6, 85.1, 93.0],
    },
}

# (row, col): key, title, is_pct, log, fmt  -- portrait 3x2 grid
PANELS = [
    (0, 0, "kld", "Median KLD vs fp16  (lower better)", False, True, "{:.2f}"),
    (0, 1, "topp", "Top-token match rate", True, False, "{:.0f}%"),
    (1, 0, "tool", "Correct tool selection rate", True, False, "{:.0f}%"),
    (1, 1, "arg", "Correct argument rate", True, False, "{:.0f}%"),
    (2, 0, "valid", "Valid tool-call rate", True, False, "{:.0f}%"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#cfcec9", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
})

fig, axes = plt.subplots(3, 2, figsize=(8.4, 11.6), facecolor="white")
for ax in axes.flat:
    ax.set_facecolor("white")

_HALO = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72)

for row, col, key, title, is_pct, log, fmt in PANELS:
    ax = axes[row, col]
    g, o = D["gemma"][key], D["ornith"][key]
    for model, color, marker in (("gemma", GEMMA, "o"), ("ornith", ORNITH, "s")):
        y = D[model][key]
        ax.plot(XI, y, color=color, lw=2.2, marker=marker, ms=8,
                mfc=color, mec="white", mew=1.4, zorder=3, clip_on=False)
        for i, v in enumerate(y):
            other = (o if model == "gemma" else g)[i]
            near_floor = is_pct and v <= 6        # keep near-0 labels off the x-ticks
            both_low = near_floor and other <= 6
            if log or near_floor:
                above = True
            elif v != other:                      # side away from the other line
                above = v > other
            else:
                above = model == "gemma"
            dy, va = (11, "bottom") if above else (-12, "top")
            dx = (-11 if model == "gemma" else 11) if both_low else 0
            ha = "right" if (both_low and model == "gemma") else \
                 "left" if both_low else "center"
            ax.annotate(fmt.format(v), (XI[i], v), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va, fontsize=8.5,
                        color=color, fontweight="bold", zorder=5, bbox=_HALO)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=10, loc="left")
    ax.set_xticks(XI)
    ax.set_xticklabels(X, fontsize=9)
    ax.set_xlim(-0.28, 2.28)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.07, 9)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if is_pct:
        ax.set_ylim(-8, 108)
        ax.set_yticks([0, 25, 50, 75, 100])

# group captions on the left
axes[0, 0].set_ylabel("STATIC\nwhat KLD/PPL see", fontsize=10, fontweight="bold",
                      color=MUTE, labelpad=12)
axes[1, 0].set_ylabel("AGENTIC\nwhat actually matters", fontsize=10, fontweight="bold",
                      color=MUTE, labelpad=12)

# legend + takeaway in the spare bottom-right cell
leg = axes[2, 1]
leg.axis("off")
handles = [
    Line2D([0], [0], color=GEMMA, lw=2.4, marker="o", ms=8, mec="white",
           label="gemma-4-31B  (10 GB)"),
    Line2D([0], [0], color=ORNITH, lw=2.4, marker="s", ms=8, mec="white",
           label="Ornith-9B  (3.4 GB)"),
]
leg.legend(handles=handles, loc="upper center", frameon=False, fontsize=11,
           handlelength=1.8, bbox_to_anchor=(0.5, 1.0))
leg.text(0.5, 0.62,
         "Same story on both models:\n\n"
         "no calibration = dead\n(the bigger Q2_K can't tool-call)\n\n"
         "imatrix wins the static row,\nbut AGENTIC keeps\nclimbing to AWQ.\n\n"
         "KLD/PPL don't\npredict tool use.",
         transform=leg.transAxes, ha="center", va="top", fontsize=10.5,
         color=MUTE, linespacing=1.5)

fig.suptitle("How a 2-bit quant changes with\ncalibration technique",
             fontsize=16, fontweight="bold", color=INK, x=0.5, y=0.988)
fig.text(0.5, 0.918,
         "same model, same size, three ways to spend the bits:\nno calibration  >  imatrix  >  AWQ",
         ha="center", fontsize=10.5, color=MUTE, linespacing=1.4)

fig.tight_layout(rect=[0.02, 0.0, 1, 0.895], h_pad=3.0, w_pad=2.0)
fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")

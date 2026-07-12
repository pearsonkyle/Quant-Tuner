"""Figures: how 2-bit quality changes with calibration technique.

ONE figure per model (mixing the two models in one panel was confusing).
Each figure is small multiples, one metric per panel; the three bars in a
panel are the three techniques Q2_K(none) -> imatrix -> AWQ. Top row = static
fidelity (what KLD/PPL see), rest = agentic ability (what actually matters).
Data from exp-051/052/053.

    PYTHONPATH=src .venv/bin/python scripts/plot_calibration_figure.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

TECHS = ["Q2_K\n(none)", "IQ2_M\n(imatrix)", "IQ2_M\n(AWQ)"]
XI = [0, 1, 2]
# none = muted gray (the dead baseline); imatrix = blue; AWQ = green (the winner)
TECH_COLORS = ["#9a9995", "#2a78d6", "#0f8a5f"]

INK = "#0b0b0b"
MUTE = "#52514e"
GRID = "#e6e5e2"

MODELS = {
    "gemma": {
        "name": "gemma-4-31B  (10 GB, IQ2_M)",
        "kld": [5.21, 1.57, 1.80], "topp": [25.4, 46.6, 43.9],
        "tool": [0.0, 45.4, 49.2], "arg": [0.0, 17.1, 26.3], "valid": [0.0, 80.5, 82.3],
    },
    "ornith": {
        "name": "Ornith-9B  (3.4 GB, IQ2_M)",
        "kld": [2.03, 0.11, 0.12], "topp": [37.9, 80.6, 79.7],
        "tool": [2.6, 30.6, 53.6], "arg": [0.0, 5.4, 33.3], "valid": [2.6, 85.1, 93.0],
    },
}

# key, title, group, is_pct, fmt
METRICS = [
    ("kld", "Median KLD vs fp16  (lower better)", "static", False, "{:.2f}"),
    ("topp", "Top-token match rate", "static", True, "{:.0f}%"),
    ("tool", "Correct tool selection rate", "agentic", True, "{:.0f}%"),
    ("arg", "Correct argument rate", "agentic", True, "{:.0f}%"),
    ("valid", "Valid tool-call rate", "agentic", True, "{:.0f}%"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#cfcec9", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
})


def make_figure(mkey: str, m: dict) -> Path:
    ncells = len(METRICS) + 1                 # + legend/takeaway cell
    nrows = math.ceil(ncells / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.4, 3.7 * nrows), facecolor="white")
    axf = axes.flat
    for ax in axf:
        ax.set_facecolor("white")

    static_ax = agentic_ax = None
    for idx, (key, title, group, is_pct, fmt) in enumerate(METRICS):
        ax = axes.flat[idx]
        y = m[key]
        ax.bar(XI, y, width=0.62, color=TECH_COLORS, zorder=3,
               edgecolor="white", linewidth=0.8)
        for xi, v in zip(XI, y):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom", fontsize=9.5,
                        color=INK, fontweight="bold", zorder=5)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=10, loc="left")
        ax.set_xticks(XI)
        ax.set_xticklabels(TECHS, fontsize=9.5)
        ax.set_xlim(-0.62, 2.62)
        ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if is_pct:
            ax.set_ylim(0, 116)
            ax.set_yticks([0, 25, 50, 75, 100])
        else:
            ax.set_ylim(0, max(y) * 1.18)
        if group == "static" and static_ax is None:
            static_ax = ax
        if group == "agentic" and agentic_ax is None:
            agentic_ax = ax

    if static_ax is not None:
        static_ax.set_ylabel("STATIC\nwhat KLD/PPL see", fontsize=10,
                             fontweight="bold", color=MUTE, labelpad=12)
    if agentic_ax is not None:
        agentic_ax.set_ylabel("AGENTIC\nwhat actually matters", fontsize=10,
                              fontweight="bold", color=MUTE, labelpad=12)

    # legend + takeaway in the trailing empty cell(s)
    for j in range(len(METRICS), nrows * 2):
        axes.flat[j].axis("off")
    leg = axes.flat[len(METRICS)]
    handles = [Patch(facecolor=c, edgecolor="white", label=t.replace("\n", " "))
               for t, c in zip(TECHS, TECH_COLORS)]
    leg.legend(handles=handles, loc="upper center", frameon=False, fontsize=11,
               title="calibration", title_fontsize=10, bbox_to_anchor=(0.5, 1.0))
    leg.text(0.5, 0.5,
             "no calibration = dead\n(Q2_K can't tool-call)\n\n"
             "imatrix wins the static row,\nbut AGENTIC keeps\nclimbing to AWQ.\n\n"
             "KLD/PPL don't\npredict tool use.",
             transform=leg.transAxes, ha="center", va="top", fontsize=10.5,
             color=MUTE, linespacing=1.5)

    fig.suptitle(m["name"], fontsize=16, fontweight="bold", color=INK, x=0.5, y=0.992)
    fig.text(0.5, 0.945,
             "one 2-bit model, three ways to spend the bits:  "
             "no calibration  >  imatrix  >  AWQ",
             ha="center", fontsize=10.5, color=MUTE)

    fig.tight_layout(rect=[0.02, 0.0, 1, 0.925], h_pad=3.0, w_pad=2.0)
    out = ROOT / f"reddit_kld_{mkey}.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


for mkey, m in MODELS.items():
    print(f"wrote {make_figure(mkey, m)}")

#!/usr/bin/env python3
"""Figures for the Ornith-1.0-9B IQ2_M Jacobian-lens analysis.

Writes two PNGs under out/exp-051/lens/:

  jlens_story.png    2x2: (A) 2-bit sharpens the explore>>act bias at the commit
                     decision; (B) the multi-direction bake shifts that readout;
                     (C) ...but the real tool-call holdout shows it breaks the
                     model; (D) the repetition penalty is the safe loop fix.
  jlens_layers.png   layer x token rank heatmap — the "inside the network" view
                     of where explore/act tokens sit across the stack.

Data are the measured results of exp-105 / exp-107 (commit-decision token ranks
F16-vs-IQ2_M and control-vs-baked), the tool-call holdout eval, the
repetition-penalty sweep, and the captured layer matrix (out/exp-051/lens/
layer_ranks.json). Lower rank = the model is more disposed to say that token.

    PYTHONPATH=src python scripts/plot_jlens_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "out" / "exp-051" / "lens"

INK, MUTE, GRID = "#26262b", "#68687a", "#e8e8ee"
EXPLORE_C, ACT_C = "#2a78d6", "#e1575a"      # CVD-safe blue / warm red (by job)
BEFORE_C = "#b8b4ad"                          # neutral "before" dot
GOOD_C, BAD_C = "#0f8a5f", "#e1575a"

EXPLORE = [" Let", " inspect", " look", " understand"]
ACT = [" edit", " fix", " modify", " change", " apply"]

# exp-105: commit-decision ranks, FP16 reference vs IQ2_M-AWQ (shared lens)
RANKS_105 = {
    " Let": (697, 2), " inspect": (149, 64), " look": (844, 841), " understand": (23692, 2500),
    " edit": (8750, 30177), " fix": (167620, 126508), " modify": (20077, 155040),
    " change": (76177, 91820), " apply": (95959, 21523),
}
# exp-107: commit-decision ranks, matched control vs multi-direction baked
RANKS_107 = {
    " Let": (4, 46), " inspect": (350, 7892), " look": (993, 3287), " understand": (7621, 29011),
    " edit": (7874, 15699), " fix": (77193, 30261), " modify": (167418, 4817),
    " change": (15646, 8491), " apply": (14738, 10144),
}
# exp-107: real 25-session tool-call holdout accuracy
TOOLCALL = {"tool selection": (0.40, 0.20), "param accuracy": (0.193, 0.013), "schema valid": (0.88, 0.75)}
# repetition-penalty sweep: unique tokens of 320 greedy tokens on a loop-prone prompt
PENALTY = [(1.0, 101 / 320), (1.15, 191 / 320), (1.30, 275 / 320)]


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcec9")
    ax.tick_params(colors=MUTE, labelsize=8.5)


def _dumbbell(ax, ranks, before_lbl, after_lbl, title, subtitle):
    tokens = EXPLORE + ["__sep__"] + ACT
    ypos = list(range(len(tokens)))[::-1]
    for y, tok in zip(ypos, tokens, strict=False):
        if tok == "__sep__":
            ax.axhline(y, color=GRID, lw=1, ls=(0, (2, 2)))
            continue
        b, a = ranks[tok]
        c = EXPLORE_C if tok in EXPLORE else ACT_C
        ax.plot([b, a], [y, y], color=c, lw=1.6, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.scatter([b], [y], s=34, color=BEFORE_C, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([a], [y], s=46, color=c, zorder=4, edgecolor="white", linewidth=0.8)
        ax.annotate(f"{a:,}", (a, y), xytext=(6 if a >= b else -6, 0), textcoords="offset points",
                    va="center", ha="left" if a >= b else "right", fontsize=7.2, color=INK)
    ax.set_yticks([y for y, t in zip(ypos, tokens, strict=False) if t != "__sep__"])
    ax.set_yticklabels([t.strip() for t in tokens if t != "__sep__"], fontsize=8.5)
    for lbl, y in zip(ax.get_yticklabels(), [t for t in tokens if t != "__sep__"], strict=False):
        lbl.set_color(EXPLORE_C if y in EXPLORE else ACT_C)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e5)
    ax.set_xlabel("lens rank of next token  (log — lower = more likely)", fontsize=8, color=MUTE)
    ax.text(0.0, 1.12, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, fontsize=8.6, color=MUTE)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _style(ax)
    # legend above the plot (top-right), clear of the left-aligned title/subtitle
    # and of the data, which spans the panel
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=BEFORE_C, mec="white", label=before_lbl),
        Line2D([], [], marker="o", ls="", mfc=INK, mec="white", label=after_lbl),
        Line2D([], [], marker="s", ls="", mfc=EXPLORE_C, mec="white", label="explore"),
        Line2D([], [], marker="s", ls="", mfc=ACT_C, mec="white", label="act"),
    ], fontsize=7.4, loc="lower right", bbox_to_anchor=(1.0, 1.135), ncol=4,
        frameon=False, handletextpad=0.25, columnspacing=1.0)


def _toolcall(ax):
    labels = list(TOOLCALL)
    x = np.arange(len(labels))
    ctrl = [TOOLCALL[k][0] for k in labels]
    bake = [TOOLCALL[k][1] for k in labels]
    ax.bar(x - 0.2, ctrl, 0.38, color=GOOD_C, zorder=3, edgecolor="white", label="control quant")
    ax.bar(x + 0.2, bake, 0.38, color=BAD_C, zorder=3, edgecolor="white", label="multi-dir baked")
    for xi, (c, b) in zip(x, zip(ctrl, bake, strict=False), strict=False):
        ax.annotate(f"{c:.2f}", (xi - 0.2, c), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.4, color=INK)
        ax.annotate(f"{b:.3f}", (xi + 0.2, b), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.4, color=BAD_C)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score on real 25-session holdout", fontsize=8, color=MUTE)
    ax.text(0.0, 1.12, "C · The bake broke the model", transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0.0, 1.045, "readout shifted (B), but real tool-calling collapsed — param acc 0.193→0.013",
            transform=ax.transAxes, fontsize=8.6, color=MUTE)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(fontsize=7.6, loc="upper left", frameon=False)


def _penalty(ax):
    xs = [p for p, _ in PENALTY]
    ys = [u for _, u in PENALTY]
    ax.plot(xs, ys, "-o", color=EXPLORE_C, lw=2, ms=8, mec="white", zorder=3)
    for x, y in PENALTY:
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xlim(0.95, 1.35)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("repeat_penalty", fontsize=8, color=MUTE)
    ax.set_ylabel("unique-token ratio (of 320 greedy)", fontsize=8, color=MUTE)
    ax.text(0.0, 1.12, "D · Repetition penalty: the safe loop fix", transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0.0, 1.045, "a sampling knob — breaks loops, no weight edit, no capability loss",
            transform=ax.transAxes, fontsize=8.6, color=MUTE)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _style(ax)


def story():
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.4), facecolor="white")
    _dumbbell(axes[0, 0], RANKS_105, "FP16", "IQ2_M",
              "A · 2-bit sharpens the explore ≫ act bias",
              "at the commit decision, IQ2_M pushes action verbs down and 'Let me…' up vs FP16")
    _dumbbell(axes[0, 1], RANKS_107, "control", "baked",
              "B · The jlens bake shifts the readout",
              "ablating the explore-vs-act subspace: explore ranks ↑ (worse), act ranks ↓ (modify 167k→4.8k)")
    _toolcall(axes[1, 0])
    _penalty(axes[1, 1])
    fig.suptitle("Inside a 2-bit agent: why Ornith-9B IQ2_M patches but never resolves",
                 fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.965, "Jacobian-lens readout of the 'keep exploring vs commit to an edit' decision "
             "· lens fit on FP16 over the calibration corpus",
             fontsize=9, color=MUTE, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "jlens_story.png"
    fig.savefig(p, dpi=150, facecolor="white")
    print("wrote", p)


def layers():
    data = json.loads((OUT / "layer_ranks.json").read_text())
    tokens = data["tokens"]
    layers = [int(x) for x in data["layers"]]
    M = np.array([[np.log10(data["ranks"][str(ly)][t] + 1) for t in tokens] for ly in layers])
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
    im = ax.imshow(M, aspect="auto", cmap="magma_r", origin="lower")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([t.strip() for t in tokens], fontsize=9)
    for lbl, t in zip(ax.get_xticklabels(), tokens, strict=False):
        lbl.set_color(EXPLORE_C if t in EXPLORE else ACT_C)
        lbl.set_fontweight("bold")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_ylabel("layer (0 = input · 31 = output)", fontsize=9, color=MUTE)
    for i in range(len(layers)):
        for j in range(len(tokens)):
            r = data["ranks"][str(layers[i])][tokens[j]]
            txt = f"{r//1000}k" if r >= 1000 else str(r)
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="white" if M[i, j] > M.max() * 0.55 else "#333")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("log₁₀(rank)  ·  bright = more likely", fontsize=8.5, color=MUTE)
    cb.ax.tick_params(colors=MUTE, labelsize=7.5)
    ax.set_title("Where the decision forms: lens rank of explore vs act tokens across layers "
                 "(Ornith-9B IQ2_M)", fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    p = OUT / "jlens_layers.png"
    fig.savefig(p, dpi=150, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    story()
    layers()

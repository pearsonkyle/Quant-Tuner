"""Release v3 comparison figure for the awq-cv-gate Gemma-4-31B-it upload.

All 11 shipped rows (vanilla×7 + QAT×4) plus the FP16 reference, on the
same external eval corpus (eaddario/imatrix-calibration code+math+tools,
~90k tokens, ctx=4096). Numbers reflect the proxy_tokens=1024 / q2k_super
/ q2k_b16+iq3_s-mix recipes from exp-031/034/035/036/038.

Four stacked subplots:
  1. Perplexity (log, ↓)
  2. Median KLD vs FP16 (↓)
  3. same_top_p % (↑)
  4. File size (GiB, log)

Source × method color scheme:
  vanilla plain    — gray
  vanilla imatrix  — light blue
  vanilla AWQ      — orange
  QAT     imatrix  — dark blue
  QAT     AWQ      — red
  FP16             — black
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RELEASE_DIR = REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
OUT_FIG = RELEASE_DIR / "awq_cv_gate_release.png"
OUT_FIG2 = REPO / "out" / "figures" / "awq_cv_gate_release.png"


# Hardcoded from the authoritative CSVs:
#   vanilla rows   : out/exp-034/results.csv   (+ exp-036 q2k_super, exp-038 iq3_s mix)
#   QAT rows       : out/exp-034/results.csv   (+ exp-035 q2k_super, exp-031 IQ2_XS AWQ)
# (label, group, size_gib, ppl, median_kld, top_p)
ROWS = [
    ("Q2_K plain",           "v_plain",   11.10, 3370.57, 5.147, 25.83),
    ("IQ2_XS imatrix",       "v_imatrix",  8.88, 12116.47, 3.327, 33.23),
    ("IQ2_XS AWQ",           "v_awq",      8.88,  327.28, 1.817, 46.29),
    ("IQ2_M imatrix",        "v_imatrix", 10.17, 2060.73, 1.496, 47.61),
    ("IQ2_M AWQ",            "v_awq",     10.17,  652.81, 1.548, 48.79),
    ("Q2_K_S imatrix",       "v_imatrix", 10.22, 1436.40, 2.138, 42.60),
    ("Q2_K_S AWQ",           "v_awq",     10.22,   73.09, 1.632, 51.14),
    ("qat-IQ2_XS imatrix",   "q_imatrix",  8.88,  209.00, 1.270, 47.09),
    ("qat-IQ2_XS AWQ",       "q_awq",      8.88,  108.65, 1.151, 48.94),
    ("qat-Q2_K_S imatrix",   "q_imatrix", 10.22,  110.71, 1.332, 47.65),
    ("qat-Q2_K_S AWQ",       "q_awq",     10.22,   88.18, 1.081, 49.40),
    ("FP16",                 "fp16",      57.20,  277.89, 0.000, 100.00),
]

COLORS = {
    "v_plain":   "#9ca3af",   # gray
    "v_imatrix": "#93c5fd",   # light blue
    "v_awq":     "#fb923c",   # orange
    "q_imatrix": "#1d4ed8",   # dark blue
    "q_awq":     "#dc2626",   # red
    "fp16":      "#1a1a1a",
}

LEGEND = [
    ("plain (no calibration)",        "v_plain"),
    ("vanilla imatrix-only",          "v_imatrix"),
    ("vanilla AWQ cv-gate",           "v_awq"),
    ("QAT imatrix-only",              "q_imatrix"),
    ("QAT AWQ cv-gate (headline)",    "q_awq"),
    ("FP16 reference",                "fp16"),
]


def main() -> int:
    labels = [r[0] for r in ROWS]
    groups = [r[1] for r in ROWS]
    sizes  = [r[2] for r in ROWS]
    ppls   = [r[3] for r in ROWS]
    klds   = [r[4] for r in ROWS]
    tops   = [r[5] for r in ROWS]
    colors = [COLORS[g] for g in groups]

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    ax_ppl, ax_kld, ax_top, ax_size = axes
    x = list(range(len(labels)))

    def _bar(ax, ys, ylabel, *, log=False, ymin=None, fmt=None):
        bars = ax.bar(x, ys, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, which="both" if log else "major")
        if log:
            ax.set_yscale("log")
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        for b, y in zip(bars, ys):
            label = fmt(y) if fmt else (f"{y:.1f}" if not log or y < 1000 else f"{y:.0f}")
            ax.annotate(
                label,
                (b.get_x() + b.get_width() / 2, y),
                ha="center", va="bottom", fontsize=8,
                xytext=(0, 2), textcoords="offset points",
            )

    _bar(ax_ppl, ppls, "Perplexity (log, ↓)", log=True)
    ax_ppl.set_title(
        "Gemma-4-31B-it · 2-bit AWQ cv-gate release\n"
        "(proxy_tokens=1024, ctx=4096, q2k_super / q2k_b16+iq3_s codebook proxies)"
    )

    _bar(ax_kld, klds, "Median KLD vs FP16 (↓)",
         fmt=lambda y: f"{y:.3f}" if y < 10 else f"{y:.1f}")
    ax_kld.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    _bar(ax_top, tops, "same_top_p % (↑)", ymin=0)
    ax_top.set_ylim(0, 108)
    ax_top.axhline(100.0, color="black", linestyle="--", linewidth=0.6, alpha=0.4)

    _bar(ax_size, sizes, "File size (GiB, log)", log=True)

    ax_size.set_xticks(x)
    ax_size.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[k], ec="black", lw=0.6, label=lbl)
        for lbl, k in LEGEND
    ]
    ax_ppl.legend(handles=handles, loc="upper right", fontsize=9, ncol=2)

    fig.text(
        0.5, -0.005,
        "All rows benched on the same external eval corpus "
        "(eaddario/imatrix-calibration: code + math + tools, ~90k tokens). "
        "ctx=4096 throughout. Median KLD chosen for robustness to per-token tails.",
        ha="center", fontsize=8, style="italic", color="#555",
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for out in (OUT_FIG, OUT_FIG2):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

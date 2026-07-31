"""Scatter alternative for the awq-cv-gate release figure.

Three vertically-stacked subplots vs file size (shared x-axis):
  1. Perplexity (log scale, ↓)
  2. Mean KLD vs FP16 (↓)
  3. same_top_p % (↑)

For each quant family (IQ2_XS / IQ2_M / Q2_K_S) the imatrix-only and
AWQ cv-gate points sit at the same x. A thin connecting line is drawn
between them so the "what does AWQ buy you over imatrix" delta is
immediately visible. Plain Q2_K and FP16 are standalone anchors.

Reuses ROWS and CSV-loader from ``plot_awq_cv_gate_release``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

# Re-use the same data loader so both views stay in sync.
from plot_awq_cv_gate_release import (  # noqa: E402
    COLORS,
    DEFAULT_RESULTS_ROOT,
    ROWS as DEFAULT_ROWS,
    _build_rows,
)

DEFAULT_OUT_FIG = _REPO / "out" / "figures" / "awq_cv_gate_release_scatter.png"
RELEASE_DIR = (
    _REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
)
RELEASE_FIG = RELEASE_DIR / "awq_cv_gate_release_scatter.png"

# Marker per technique group.
MARKERS = {
    "imatrix": "o",
    "cv-gate": "s",
    "plain":   "D",
    "fp16":    "*",
}
SIZES = {
    "imatrix": 90,
    "cv-gate": 90,
    "plain":   140,
    "fp16":    260,
}


def _by_group(ys: dict[str, float], rows) -> dict[str, list[tuple[float, float, str]]]:
    """Return {group: [(x, y, label), ...]}."""
    out: dict[str, list[tuple[float, float, str]]] = {}
    for label, group, size, ppl, kld, top in rows:
        out.setdefault(group, []).append((size, ys[label], label))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--imatrix-subdir", type=str, default="imatrix-only-rebench")
    ap.add_argument("--fp16-size", type=float, default=57.20)
    ap.add_argument("--fp16-ppl", type=float, default=277.8886)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_FIG)
    ap.add_argument("--release-out", type=Path, default=None)
    a = ap.parse_args()

    if (a.results_root == DEFAULT_RESULTS_ROOT
            and a.imatrix_subdir == "imatrix-only-rebench"):
        rows = DEFAULT_ROWS
    else:
        rows = _build_rows(
            a.results_root, a.imatrix_subdir, a.fp16_size, a.fp16_ppl,
        )

    # Index rows by label for the helper.
    ppl_by_label = {r[0]: r[3] for r in rows}
    kld_by_label = {r[0]: r[4] for r in rows}
    top_by_label = {r[0]: r[5] for r in rows}

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax_ppl, ax_kld, ax_top = axes

    metric_axes = [
        (ax_ppl, ppl_by_label, "Perplexity (log, ↓)", True,  "ppl"),
        (ax_kld, kld_by_label, "Mean KLD vs FP16 (↓)", False, "kld"),
        (ax_top, top_by_label, "same_top_p % (↑)",     False, "top"),
    ]
    metric_idx = {"ppl": 0, "kld": 1, "top": 2}

    # Pair imatrix → cv-gate by quant family for connecting lines.
    quant_of = {
        "IQ2_XS imatrix": "IQ2_XS",  "IQ2_XS awq-cv-gate": "IQ2_XS",
        "IQ2_M imatrix":  "IQ2_M",   "IQ2_M awq-cv-gate":  "IQ2_M",
        "Q2_K_S imatrix": "Q2_K_S",  "Q2_K_S awq-cv-gate": "Q2_K_S",
    }
    pairs: dict[str, dict[str, tuple]] = {}
    for label, group, size, ppl, kld, top in rows:
        q = quant_of.get(label)
        if q is None:
            continue
        pairs.setdefault(q, {})[group] = (size, ppl, kld, top)

    for ax, ys, ylabel, log, mkey in metric_axes:
        # Connecting lines first (so they sit behind markers).
        for q, by_g in pairs.items():
            if "imatrix" in by_g and "cv-gate" in by_g:
                x0, *vals0 = by_g["imatrix"]
                x1, *vals1 = by_g["cv-gate"]
                idx = metric_idx[mkey]
                ax.plot([x0, x1], [vals0[idx], vals1[idx]],
                        color="black", linewidth=0.8, alpha=0.35, zorder=1)

        for group, pts in _by_group(ys, rows).items():
            xs = [p[0] for p in pts]
            ys_ = [p[1] for p in pts]
            ax.scatter(
                xs, ys_,
                marker=MARKERS[group], color=COLORS[group],
                s=SIZES[group], edgecolor="black", linewidth=0.7,
                zorder=3,
                label={
                    "imatrix": "imatrix only",
                    "cv-gate": "AWQ cv-gate (this release)",
                    "plain":   "plain Q2_K (no imatrix, no AWQ)",
                    "fp16":    "FP16 reference",
                }[group],
            )

        ax.set_ylabel(ylabel)
        ax.grid(True, which="both" if log else "major", alpha=0.3)
        if log:
            ax.set_yscale("log")

    # Annotate quant labels on the PPL subplot. Offset placement avoids
    # overlap between same-size pairs and the legend.
    ANNOT_OFFSETS = {
        "IQ2_XS imatrix":     (8, 6),
        "IQ2_XS awq-cv-gate": (8, -14),
        "IQ2_M imatrix":      (-8, 10),     # left+up to dodge Q2_K_S
        "IQ2_M awq-cv-gate":  (8, 8),
        "Q2_K_S imatrix":     (8, -12),     # right+down
        "Q2_K_S awq-cv-gate": (-8, -14),
        "Q2_K plain":         (8, 6),
        "FP16":               (-8, 8),
    }
    ANNOT_ALIGN = {
        "IQ2_M imatrix":      "right",
        "Q2_K_S awq-cv-gate": "right",
        "FP16":               "right",
    }
    for label, group, size, ppl, kld, top in rows:
        short = (label.replace(" awq-cv-gate", "·gate")
                      .replace(" imatrix", "·imatrix")
                      .replace(" plain", "·plain"))
        ax_ppl.annotate(
            short, (size, ppl),
            textcoords="offset points",
            xytext=ANNOT_OFFSETS.get(label, (6, 6)),
            ha=ANNOT_ALIGN.get(label, "left"),
            fontsize=7.5, color=COLORS[group], alpha=0.95,
        )

    # FP16 reference lines on KLD & top_p.
    ax_kld.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_top.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_top.set_ylim(0, 110)
    # Give PPL annotations breathing room above the imatrix peaks.
    ymin, ymax = ax_ppl.get_ylim()
    ax_ppl.set_ylim(ymin, ymax * 3)

    ax_ppl.set_title(
        "Gemma-4-31B-it · AWQ cv-gate release · file size vs quality"
    )
    # Legend lives on the KLD subplot (empty mid-x region) so it doesn't
    # collide with PPL annotations.
    ax_kld.legend(loc="upper right", fontsize=9)

    ax_top.set_xlabel("File size (GiB, log scale)")
    ax_top.set_xscale("log")

    fig.text(
        0.5, -0.005,
        "All rows benched on the same external eval corpus "
        "(eaddario/imatrix-calibration: code + math + tools, ~90k tokens). "
        "Thin lines connect each quant family's imatrix-only point to its "
        "AWQ cv-gate point.",
        ha="center", fontsize=8, style="italic", color="#555",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {a.out}")
    if a.release_out is not None:
        a.release_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.release_out, dpi=140, bbox_inches="tight")
        print(f"saved -> {a.release_out}")
    elif a.out == DEFAULT_OUT_FIG and RELEASE_DIR.exists():
        fig.savefig(RELEASE_FIG, dpi=140, bbox_inches="tight")
        print(f"saved -> {RELEASE_FIG.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

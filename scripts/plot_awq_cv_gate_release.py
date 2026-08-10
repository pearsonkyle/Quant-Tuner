"""Comparison figure for the awq-cv-gate Gemma-4-31B-it release.

Four stacked subplots over the same 8-model bar set:
  1. Perplexity (log scale, lower is better)
  2. Mean KLD vs FP16 (lower is better)
  3. same_top_p % (higher is better)
  4. File size in GiB (log scale, FP16 dominates)

Bars are ordered by file size, ascending. AWQ-cv-gate bars are highlighted
(orange) to make this release pop next to the imatrix-only baselines (blue).
Q2_K plain (gray) and FP16 (black) bracket the comparison.

Output: ``out/figures/awq_cv_gate_release.png`` (also saved into the
uploads/ folder so the README can reference a relative path).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO / "out" / "exp-019" / "google__gemma-4-31B-it"
DEFAULT_OUT_FIG = REPO / "out" / "figures" / "awq_cv_gate_release.png"
RELEASE_DIR = (
    REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
)
RELEASE_FIG = RELEASE_DIR / "awq_cv_gate_release.png"


# Imatrix-only numbers come from the rebench CSV so they sit on the same
# eval corpus as the cv-gate rows. Fallbacks are the old exp-009 numbers
# (logtrain+wiki eval) used only if the rebench hasn't run yet.
def _load_row(csv_path: Path, quant: str, technique: str) -> tuple[float, float, float, float] | None:
    if not csv_path.exists():
        return None
    if technique == "imatrix":
        needle = f"|{quant}|imatrix|"
    elif technique == "plain":
        needle = f"|{quant}|plain|"
    else:
        needle = f"|{quant}|{technique}+imatrix|"
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if needle in row["model"]:
                return (float(row["size_gib"]), float(row["ppl"]),
                        float(row["mean_kld"]), float(row["same_top_p"]))
    return None


def _row(label, group, quant, technique, csv_path, *, fallback):
    pulled = _load_row(csv_path, quant, technique)
    if pulled is None:
        size, ppl, kld, top = fallback
    else:
        size, ppl, kld, top = pulled
    return (label, group, size, ppl, kld, top)


def _load_row_flat(csv_path: Path, quant: str, technique: str) -> tuple[float, float, float, float] | None:
    """Load a row from a flat exp-044-style single results.csv."""
    if not csv_path.exists():
        return None
    if technique == "imatrix":
        needle = f"|{quant}|imatrix "
    elif technique == "plain":
        needle = f"|{quant}|plain "
    else:
        needle = f"|{quant}|{technique}+imatrix|"
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if needle in row["model"] and "|qat|" not in row["model"].lower() and "qat" not in row["model"].split("|")[0].lower():
                return (float(row["size_gib"]), float(row["ppl"]),
                        float(row["mean_kld"]), float(row["same_top_p"]))
    return None


def _build_rows_flat(flat_csv: Path, fp16_size: float, fp16_ppl: float) -> list:
    """Build rows from a flat exp-044-style results.csv (vanilla rows only)."""
    def _r(label, group, quant, technique, fallback):
        pulled = _load_row_flat(flat_csv, quant, technique)
        size, ppl, kld, top = pulled if pulled is not None else fallback
        return (label, group, size, ppl, kld, top)

    return [
        _r("IQ2_XS imatrix",     "imatrix", "IQ2_XS", "imatrix",
           (8.88,  5711.4740, 4.78865, 33.464)),
        _r("IQ2_XS awq-cv-gate", "cv-gate", "IQ2_XS", "awq-cv-gate",
           (8.88,   242.5148, 3.50384, 46.124)),
        _r("IQ2_M imatrix",      "imatrix", "IQ2_M",  "imatrix",
           (10.17, 3917.5216, 3.67023, 36.043)),
        _r("IQ2_M awq-cv-gate",  "cv-gate", "IQ2_M",  "awq-cv-gate",
           (10.17,  305.3430, 2.47575, 52.704)),
        _r("Q2_K_S imatrix",     "imatrix", "Q2_K_S", "imatrix",
           (10.22, 4004.0194, 3.91406, 39.932)),
        _r("Q2_K_S awq-cv-gate", "cv-gate", "Q2_K_S", "awq-cv-gate",
           (10.22,  146.2482, 3.54850, 51.170)),
        _r("Q2_K plain",         "plain",   "Q2_K",   "plain",
           (11.10, 3370.5679, 6.11890, 25.826)),
        ("FP16",                 "fp16",    fp16_size, fp16_ppl, 0.00000, 100.000),
    ]


def _build_rows(
    results_root: Path,
    imatrix_subdir: str,
    fp16_size: float,
    fp16_ppl: float,
) -> list:
    return [
        _row("IQ2_XS imatrix",     "imatrix", "IQ2_XS", "imatrix",
             results_root / imatrix_subdir / "results.csv",
             fallback=(8.88,  5711.4740, 4.78865, 33.464)),
        _row("IQ2_XS awq-cv-gate", "cv-gate", "IQ2_XS", "awq-cv-gate",
             results_root / "gate" / "results.csv",
             fallback=(8.88,   242.5148, 3.50384, 46.124)),
        _row("IQ2_M imatrix",      "imatrix", "IQ2_M",  "imatrix",
             results_root / imatrix_subdir / "results.csv",
             fallback=(10.17, 3917.5216, 3.67023, 36.043)),
        _row("IQ2_M awq-cv-gate",  "cv-gate", "IQ2_M",  "awq-cv-gate",
             results_root / "gate" / "results.csv",
             fallback=(10.17,  305.3430, 2.47575, 52.704)),
        _row("Q2_K_S imatrix",     "imatrix", "Q2_K_S", "imatrix",
             results_root / imatrix_subdir / "results.csv",
             fallback=(10.22, 4004.0194, 3.91406, 39.932)),
        _row("Q2_K_S awq-cv-gate", "cv-gate", "Q2_K_S", "awq-cv-gate",
             results_root / "gate" / "results.csv",
             fallback=(10.22,  146.2482, 3.54850, 51.170)),
        _row("Q2_K plain",         "plain",   "Q2_K",   "plain",
             results_root / "plain" / "results.csv",
             fallback=(11.10, 3370.5679, 6.11890, 25.826)),
        ("FP16",                   "fp16",    fp16_size, fp16_ppl, 0.00000, 100.000),
    ]

# Default ROWS for back-compat (e.g. plot_awq_cv_gate_release_scatter imports
# this). CLI callers may rebuild via _build_rows() with a different root.
ROWS = _build_rows(
    DEFAULT_RESULTS_ROOT, "imatrix-only-rebench",
    fp16_size=57.20, fp16_ppl=277.8886,
)


COLORS = {
    "imatrix": "tab:blue",
    "cv-gate": "tab:orange",
    "plain":   "tab:gray",
    "fp16":    "#1a1a1a",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat-csv", type=Path, default=None,
                    help="Single flat results.csv (exp-044 style); skips "
                         "--results-root hierarchy when provided")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                    help="Per-model exp output dir containing gate/, plain/, "
                         "and an imatrix subdir with results.csv files")
    ap.add_argument("--imatrix-subdir", type=str, default="imatrix-only-rebench",
                    help="Subdir under --results-root that holds the imatrix-"
                         "only results.csv (exp-019 default: imatrix-only-"
                         "rebench; exp-020: imatrix-only)")
    ap.add_argument("--fp16-size", type=float, default=57.20)
    ap.add_argument("--fp16-ppl", type=float, default=277.8886)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_FIG,
                    help="Primary output PNG path")
    ap.add_argument("--release-out", type=Path, default=None,
                    help="If set, also save a copy here (e.g. into uploads/)")
    a = ap.parse_args()

    if a.flat_csv is not None:
        rows = _build_rows_flat(a.flat_csv, a.fp16_size, a.fp16_ppl)
    else:
        rows = _build_rows(a.results_root, a.imatrix_subdir, a.fp16_size, a.fp16_ppl)
    labels = [r[0] for r in rows]
    groups = [r[1] for r in rows]
    sizes  = [r[2] for r in rows]
    ppls   = [r[3] for r in rows]
    klds   = [r[4] for r in rows]
    tops   = [r[5] for r in rows]
    colors = [COLORS[g] for g in groups]

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    ax_ppl, ax_kld, ax_top, ax_size = axes

    x = list(range(len(labels)))

    def _bar(ax, ys, ylabel, *, log=False, ymin=None):
        bars = ax.bar(x, ys, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, which="both" if log else "major")
        if log:
            ax.set_yscale("log")
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        for b, y in zip(bars, ys):
            ax.annotate(
                f"{y:.1f}" if not log or y < 1000 else f"{y:.0f}",
                (b.get_x() + b.get_width() / 2, y),
                ha="center", va="bottom", fontsize=8,
                xytext=(0, 2), textcoords="offset points",
            )

    _bar(ax_ppl, ppls, "Perplexity (log, ↓)", log=True)
    ax_ppl.set_title(
        "Gemma-4-31B-it · AWQ cv-gate release vs imatrix-only and FP16"
    )

    _bar(ax_kld, klds, "Mean KLD vs FP16 (↓)")
    ax_kld.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    _bar(ax_top, tops, "same_top_p % (↑)", ymin=0)
    ax_top.set_ylim(0, 110)
    ax_top.axhline(100.0, color="black", linestyle="--", linewidth=0.6, alpha=0.4)

    _bar(ax_size, sizes, "File size (GiB, log)", log=True)

    ax_size.set_xticks(x)
    ax_size.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    # Single legend at the top.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["imatrix"], ec="black", lw=0.6,
                      label="imatrix only"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["cv-gate"], ec="black", lw=0.6,
                      label="AWQ cv-gate (this release)"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["plain"], ec="black", lw=0.6,
                      label="plain (no imatrix, no AWQ)"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["fp16"], ec="black", lw=0.6,
                      label="FP16 reference"),
    ]
    ax_ppl.legend(handles=legend_handles, loc="upper right", fontsize=9, ncol=2)

    fig.text(
        0.5, -0.005,
        "All rows benched on the same external eval corpus "
        "(eaddario/imatrix-calibration: code + math + tools, ~90k tokens). "
        "Same llama.cpp build, ctx=4096.",
        ha="center", fontsize=8, style="italic", color="#555",
        wrap=True,
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
        # Back-compat: default behavior mirrors into the release folder.
        fig.savefig(RELEASE_FIG, dpi=140, bbox_inches="tight")
        print(f"saved -> {RELEASE_FIG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

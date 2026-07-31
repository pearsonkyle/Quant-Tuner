"""Plot MTP acceptance rate vs draft depth WITH error bars.

Reads the exp-048 reps JSON (mean ± stdev across reps per quant/n) and renders
the acceptance-vs-n figure with per-point error bars.

Usage:
    .venv/bin/python scripts/plot_mtp_acceptance_errorbars.py \
        --in out/exp-048/acceptance_reps.json \
        --out out/exp-048/mtp_acceptance_rate_vs_draft_depth.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

N_VALUES = [1, 2, 3, 4]
# Stable plot order regardless of JSON key order.
ORDER = ["Q5_K_S", "IQ4_XS", "IQ3_M", "IQ2_M"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("out/exp-048/acceptance_reps.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("out/exp-048/mtp_acceptance_rate_vs_draft_depth.png"))
    ap.add_argument("--err", choices=["stdev", "sem"], default="stdev",
                    help="error bar source (default: stdev across reps)")
    ap.add_argument("--title",
                    default="MTP Acceptance Rate vs. Draft Depth (n) For Gemma4-31b Quants")
    ap.add_argument("--legend-title", default="Quantization Level")
    ap.add_argument("--order", default=None,
                    help="comma-separated series order (default: built-in trunk order)")
    ap.add_argument("--kind", choices=["line", "bar"], default="line",
                    help="line plot (default) or grouped bar chart")
    args = ap.parse_args()

    order = [s.strip() for s in args.order.split(",")] if args.order else ORDER

    data = json.loads(args.inp.read_text())
    nreps = max((len(r.get("reps", [])) for rows in data.values() for r in rows),
                default=0)

    try:
        plt.style.use("ggplot")
    except Exception:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 6))

    series = [q for q in order if q in data] + [q for q in data if q not in order]

    def yvals(quant):
        rows = {r["n_max"]: r for r in data[quant]}
        ys, yerr = [], []
        for n in N_VALUES:
            r = rows.get(n, {})
            m = r.get("mean_accept_rate")
            ys.append(m * 100 if m is not None else float("nan"))
            yerr.append((r.get(args.err) or 0.0) * 100)
        return ys, yerr

    if args.kind == "line":
        for quant in series:
            ys, yerr = yvals(quant)
            ax.errorbar(N_VALUES, ys, yerr=yerr, marker="o", linewidth=2.5,
                        capsize=4, capthick=1.5, elinewidth=1.5, label=quant)
            if ys[-1] == ys[-1]:  # annotate final (n=4) point, not NaN
                ax.text(4, ys[-1], f" {ys[-1]:.1f}%", ha="left", va="bottom",
                        fontsize=10, fontweight="bold")
    else:  # grouped bar chart: groups = draft depths, bars = series
        import numpy as np
        x = np.arange(len(N_VALUES))
        nser = len(series)
        width = 0.8 / max(nser, 1)
        for i, quant in enumerate(series):
            ys, yerr = yvals(quant)
            off = (i - (nser - 1) / 2) * width
            bars = ax.bar(x + off, ys, width, yerr=yerr, capsize=3,
                          label=quant, error_kw={"elinewidth": 1.2})
            ax.bar_label(bars, labels=[f"{v:.0f}" for v in ys], padding=2,
                         fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_VALUES])

    err_label = "±1σ across reps" if args.err == "stdev" else "±1 SEM"
    ax.set_title(args.title, fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Draft Depth (n tokens)", fontsize=12)
    ax.set_ylabel("Acceptance Rate (%)", fontsize=12)
    if args.kind == "line":
        ax.set_xticks(N_VALUES)
    ax.set_ylim(55, 95)
    ax.legend(title=args.legend_title, loc="upper right", frameon=True, fontsize=11)
    sub = f"error bars: {err_label}"
    if nreps:
        sub += f" (n_reps={nreps}, 5 prompts × 200 tok, T=0.3)"
    ax.text(0.01, 0.01, sub, transform=ax.transAxes, fontsize=9,
            color="#555555", ha="left", va="bottom")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

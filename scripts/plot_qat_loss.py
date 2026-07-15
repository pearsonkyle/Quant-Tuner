"""Plot QAT training loss vs step from a run log (steps-vs-loss + EMA smoothing).

The per-step masked loss is noisy (batch of a few heterogeneous windows), so the
raw trace is plotted faint with an EMA overlay that shows the real trend.

    PYTHONPATH=src .venv/bin/python scripts/plot_qat_loss.py \
        --log out/exp-058/iter3.log --out out/exp-058/iter3_loss.png --title "iter-3 (influential layers)"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_log(log: Path) -> tuple[list[int], list[float], list[float]]:
    steps, loss, lr = [], [], []
    pat = re.compile(r"step (\d+)/\d+ loss=([\d.]+) lr=([\d.eE+-]+)")
    pat2 = re.compile(r"step (\d+)/\d+ loss=([\d.]+)")  # exp-057 v1 (no lr)
    for ln in log.read_text().splitlines():
        m = pat.search(ln)
        if m:
            steps.append(int(m.group(1))); loss.append(float(m.group(2))); lr.append(float(m.group(3)))
            continue
        m = pat2.search(ln)
        if m:
            steps.append(int(m.group(1))); loss.append(float(m.group(2))); lr.append(float("nan"))
    return steps, loss, lr


def ema(xs: list[float], alpha: float = 0.3) -> list[float]:
    out, s = [], None
    for x in xs:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="ternary QAT")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, loss, lr = parse_log(args.log)
    if not steps:
        print(f"[plot] no step/loss lines in {args.log}"); return 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, loss, color="#08519c", lw=1.6)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("masked cross-entropy loss")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)

    # LR on a twin axis if present
    if any(x == x for x in lr):  # not all-nan
        ax2 = ax.twinx()
        ax2.plot(steps, lr, color="#cb181d", lw=1.2, ls="--", alpha=0.6, label="lr")
        ax2.set_ylabel("learning rate", color="#cb181d")
        ax2.tick_params(axis="y", labelcolor="#cb181d")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[plot] {len(steps)} points -> {args.out}  (loss {loss[0]:.3f} -> {sm[-1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

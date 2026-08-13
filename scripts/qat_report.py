#!/usr/bin/env python
"""The QAT run report: one page, seven figures, two tables.

A ternary model stores `w = s·c`, `c ∈ {−1,0,+1}`, so its loss can fall on scale drift
with zero codes changed. Every panel here is therefore anchored on **code flips** — the
only change that survives export to a 2-bit GGUF — and each answers one operational
question: is the schedule healthy, is it still learning, which mechanism, where in the
stack, is another GPU-hour worth it, and is the box about to swap.

Pure CSV in, one self-contained HTML out — no model load, safe to run beside training.

    # data (once per checkpoint you care about)
    python scripts/parse_qat_log.py train.log --out out/run/telemetry
    python scripts/ternary_distribution.py census --model MODEL \\
        --tensors out/run/telemetry/flips.csv --out out/run/telemetry/census.csv
    python scripts/ternary_distribution.py census --latents CKPT.pt \\
        --tensors out/run/telemetry/flips.csv --out out/run/telemetry/census_latest.csv

    # report
    python scripts/qat_report.py --telemetry out/run/telemetry \\
        --census out/run/telemetry/census.csv \\
        --latest out/run/telemetry/census_latest.csv --latest-step 325 \\
        --window 8064 --grad-accum 4 --out out/run/report.html

Categorical colors are Okabe-Ito (CVD-safe), assigned to tensor kind in fixed order so a
kind keeps its color across every figure.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_distribution import trajectory  # noqa: E402  (sibling script, path set above)

# Okabe-Ito: CVD-safe qualitative. Fixed order, never cycled — a 9th kind would need a
# different encoding, not a generated hue.
KINDS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
KIND_COLOR = dict(zip(KINDS, ["#0072b2", "#56b4e9", "#009e73",
                              "#e69f00", "#d55e00", "#cc79a7"], strict=True))
INK, MUTED, GRID = "#222", "#888", "#eee"
LOSS, VAL = "#0072b2", "#d55e00"


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    span = (hi - lo) or 1.0
    raw = span / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    out, v = [], math.floor(lo / step) * step
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            out.append(round(v, 10))
        v += step
    return out


class Panel:
    """Minimal linear-scale SVG panel with a recessive grid. Log scale via `logy`."""

    def __init__(self, w, h, pad, xlim, ylim, *, logy=False, xlabel="", ylabel="",
                 xticks=None, yfmt="{:.2f}", yticks_n=4):
        self.w, self.h = w, h
        # Horizontal padding is for tick labels; VERTICAL padding must scale with the
        # panel or a short panel has no plot area left. At h=130 a flat pad of 60 left
        # 10px of drawable height: the LR curve rendered as a straight line and every
        # y-label stacked on the same row.
        self.pad = pad
        self.pad_y = min(pad, max(18, int(h * 0.20)))
        self.yticks_n = yticks_n
        self.x0, self.x1 = xlim
        self.logy = logy
        self.y0, self.y1 = (math.log10(max(1e-9, ylim[0])), math.log10(max(1e-9, ylim[1]))) \
            if logy else ylim
        if self.y1 - self.y0 < 1e-12:
            self.y1 = self.y0 + 1.0
        self.parts: list[str] = []
        self._grid(xticks, yfmt, xlabel, ylabel)

    def px(self, x):
        return self.pad + (x - self.x0) / max(1e-9, self.x1 - self.x0) * (self.w - 2 * self.pad)

    def py(self, y):
        v = math.log10(max(1e-9, y)) if self.logy else y
        return self.h - self.pad_y - (v - self.y0) / max(1e-9, self.y1 - self.y0) \
            * (self.h - 2 * self.pad_y)

    def _grid(self, xticks, yfmt, xlabel, ylabel):
        for t in (xticks if xticks is not None else nice_ticks(self.x0, self.x1)):
            if not self.x0 - 1e-9 <= t <= self.x1 + 1e-9:
                continue
            self.parts.append(f'<line x1="{self.px(t):.1f}" y1="{self.pad_y}" '
                              f'x2="{self.px(t):.1f}" y2="{self.h - self.pad_y}" '
                              f'stroke="{GRID}" stroke-width="1"/>')
            lab = f"{t:g}"
            self.parts.append(f'<text x="{self.px(t):.1f}" y="{self.h - self.pad_y + 14}" '
                              f'font-size="10" text-anchor="middle" fill="{MUTED}">{lab}</text>')
        lo, hi = (10 ** self.y0, 10 ** self.y1) if self.logy else (self.y0, self.y1)
        if self.logy:
            ticks = [m * 10 ** e
                     for e in range(math.floor(self.y0), math.ceil(self.y1) + 1)
                     for m in (1, 2, 3, 5)]
            ticks = [t for t in sorted(ticks) if lo <= t <= hi]
        else:
            ticks = nice_ticks(lo, hi, self.yticks_n)
        drawn: list[float] = []
        for t in ticks:
            y = self.py(t)
            if not self.pad_y - 1 <= y <= self.h - self.pad_y + 1:
                continue
            if any(abs(y - d) < 11 for d in drawn):  # never stack two labels on one row
                continue
            drawn.append(y)
            self.parts.append(f'<line x1="{self.pad}" y1="{y:.1f}" x2="{self.w - self.pad}" '
                              f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            self.parts.append(f'<text x="{self.pad - 6}" y="{y + 3:.1f}" font-size="10" '
                              f'text-anchor="end" fill="{MUTED}">{yfmt.format(t)}</text>')
        if xlabel:
            self.parts.append(f'<text x="{self.w / 2:.0f}" y="{self.h - 4}" font-size="11" '
                              f'text-anchor="middle" fill="{MUTED}">{xlabel}</text>')
        if ylabel:
            # above the plot area, left-aligned to the axis — not inside it, where the
            # peak annotation used to land on top of it
            self.parts.append(f'<text x="{self.pad - 44}" y="{max(11, self.pad_y - 8):.0f}" '
                              f'font-size="11" fill="{MUTED}">{ylabel}</text>')

    def line(self, pts, color, width=2, opacity=1.0, title=""):
        if len(pts) < 2:
            return
        d = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in pts)
        t = f"<title>{title}</title>" if title else ""
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                          f'stroke-width="{width}" opacity="{opacity}">{t}</polyline>')

    def dots(self, pts, color, r=3.5, title_fn=None):
        for x, y in pts:
            t = f"<title>{title_fn(x, y)}</title>" if title_fn else ""
            # 2px surface ring so overlapping marks stay countable
            self.parts.append(f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="{r}" '
                              f'fill="{color}" stroke="#fff" stroke-width="1.5">{t}</circle>')

    def rule(self, y, color="#bbb", dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{self.pad}" y1="{self.py(y):.1f}" x2="{self.w - self.pad}" '
                          f'y2="{self.py(y):.1f}" stroke="{color}" stroke-width="1"{d}/>')

    def note(self, x, y, text, anchor="start", color=MUTED, size=10):
        self.parts.append(f'<text x="{self.px(x):.1f}" y="{self.py(y):.1f}" font-size="{size}" '
                          f'text-anchor="{anchor}" fill="{color}">{text}</text>')

    def svg(self):
        return (f'<svg width="{self.w}" height="{self.h}" role="img">'
                f'{"".join(self.parts)}</svg>')


def legend(items: list[tuple[str, str]]) -> str:
    """Identity is never color-alone; a legend is always present for >= 2 series."""
    sp = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in items)
    return f'<p class="legend">{sp}</p>'


# ---------------------------------------------------------------- figures


def fig_loss_lr(steps, vals, W, H, PAD):
    """Loss (train + val) and LR as STACKED panels sharing an x-axis.

    Deliberately not a dual-axis chart: two y-scales on one frame invite reading a
    crossing as meaningful when it is an artifact of the scaling.
    """
    xs = [r["step"] for r in steps]
    xlim = (min(xs), max(xs))
    xt = nice_ticks(*xlim)
    losses = [r["loss"] for r in steps] + [v["val_masked_ce"] for v in vals]
    p1 = Panel(W, H, PAD, xlim, (min(losses) * 0.9, max(losses) * 1.05), logy=True,
               ylabel="masked CE (log)", xticks=xt, yfmt="{:g}")
    p1.line([(r["step"], r["loss"]) for r in steps], LOSS, 2, 0.9, "train")
    p1.line([(v["step"], v["val_masked_ce"]) for v in vals], VAL, 2, 0.9, "val")
    p1.dots([(v["step"], v["val_masked_ce"]) for v in vals], VAL, 3,
            lambda x, y: f"step {x:g}: val {y:.4f}")
    peak = max(steps, key=lambda r: r["loss"])
    p1.note(peak["step"] + (xlim[1] - xlim[0]) * 0.04, peak["loss"] * 0.92,
            f'peak {peak["loss"]:.1f} @ step {peak["step"]}', "start", INK)
    p2 = Panel(W, 190, PAD, xlim, (0, max(r["lr"] for r in steps) * 1.15),
               xlabel="training step", ylabel="learning rate (×1e-4)", xticks=xt,
               yfmt="{:.1f}", yticks_n=3)
    # relabel in units of 1e-4: "5.0" reads instantly where "5.0e-04" did not
    p2.parts = [q for q in p2.parts if 'text-anchor="end"' not in q]
    for t in nice_ticks(0, max(r["lr"] for r in steps) * 1.15, 3):
        y = p2.py(t)
        if p2.pad_y - 1 <= y <= p2.h - p2.pad_y + 1:
            p2.parts.append(f'<text x="{PAD - 6}" y="{y + 3:.1f}" font-size="10" '
                            f'text-anchor="end" fill="{MUTED}">{t * 1e4:.1f}</text>')
    p2.line([(r["step"], r["lr"]) for r in steps], "#666", 2)
    peak_lr = max(steps, key=lambda r: r["lr"])
    p2.dots([(peak_lr["step"], peak_lr["lr"])], "#666", 3,
            lambda x, y: f"peak lr {y:.2e} @ step {x:g}")
    return p1.svg() + p2.svg() + legend([("train", LOSS), ("val", VAL)])


def fig_velocity(flips, W, H, PAD):
    """Per-checkpoint change in cumulative flip % — is the run still learning?"""
    rows = [r for r in flips if r.get("flip_pct_delta") not in (None, "")]
    if not rows:
        return "<p>no velocity data</p>"
    xs = [r["step"] for r in rows]
    ys = [r["flip_pct_delta"] for r in rows]
    p = Panel(W, H, PAD, (min(xs), max(xs)), (0, max(ys) * 1.1),
              xlabel="training step", ylabel="Δ flip % per checkpoint", yfmt="{:.2f}")
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["tensor"], []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["step"])
        p.line([(r["step"], r["flip_pct_delta"]) for r in rs],
               KIND_COLOR.get(rs[0]["kind"], MUTED), 2, 0.8,
               f'{rs[0]["layer"]}.{rs[0]["kind"]}')
    return p.svg() + legend([(k, KIND_COLOR[k]) for k in KINDS])


def fig_recruit_prune(flips, W, H, PAD):
    W = H = max(H, 420)  # square: see below
    """Recruited (0->±) vs pruned (±->0), log-log, against the balance diagonal.

    On the diagonal a tensor is substituting weights at constant density; below it the
    model is switching dead capacity on. This is the mechanism split in one view.
    """
    last = max(r["step"] for r in flips)
    rows = [r for r in flips if r["step"] == last and r["zero_to_nonzero"] > 0
            and r["nonzero_to_zero"] > 0]
    if not rows:
        return "<p>no recruit/prune data</p>"
    v = [r["zero_to_nonzero"] for r in rows] + [r["nonzero_to_zero"] for r in rows]
    lim = (min(v) * 0.6, max(v) * 1.6)
    p = Panel(W, H, PAD, (math.log10(lim[0]), math.log10(lim[1])), lim, logy=True,
              xlabel="weights recruited  0 → ±1  (log)",
              ylabel="weights pruned  ±1 → 0 (log)",
              xticks=[e for e in range(math.floor(math.log10(lim[0])),
                                       math.ceil(math.log10(lim[1])) + 1)],
              yfmt="{:g}")
    # relabel the log-x ticks as powers of ten
    p.parts = [q for q in p.parts if 'text-anchor="middle"' not in q or "fill=\"#888\"" not in q]
    for e in range(math.floor(math.log10(lim[0])), math.ceil(math.log10(lim[1])) + 1):
        p.parts.append(f'<text x="{p.px(e):.1f}" y="{H - PAD + 14}" font-size="10" '
                       f'text-anchor="middle" fill="{MUTED}">1e{e}</text>')
    p.parts.append(f'<line x1="{p.px(math.log10(lim[0])):.1f}" y1="{p.py(lim[0]):.1f}" '
                   f'x2="{p.px(math.log10(lim[1])):.1f}" y2="{p.py(lim[1]):.1f}" '
                   f'stroke="#bbb" stroke-width="1" stroke-dasharray="4 3"/>')
    for r in rows:
        x, y = math.log10(r["zero_to_nonzero"]), r["nonzero_to_zero"]
        c = KIND_COLOR.get(r["kind"], MUTED)
        p.parts.append(f'<circle cx="{p.px(x):.1f}" cy="{p.py(y):.1f}" r="5" fill="{c}" '
                       f'stroke="#fff" stroke-width="1.5"><title>{r["tensor"]}: '
                       f'recruited {r["zero_to_nonzero"]:,}, pruned {r["nonzero_to_zero"]:,}'
                       f'</title></circle>')
        p.parts.append(f'<text x="{p.px(x) + 8:.1f}" y="{p.py(y) + 3:.1f}" font-size="9" '
                       f'fill="{MUTED}">{r["layer"]}</text>')
    # annotations hug the axis corners, clear of the data cloud that sits on the diagonal
    p.parts.append(f'<text x="{PAD + 6}" y="{p.pad_y + 14}" font-size="10" '
                   f'fill="{MUTED}">above the line = net pruning</text>')
    p.parts.append(f'<text x="{W - PAD - 6}" y="{H - p.pad_y - 8}" font-size="10" '
                   f'text-anchor="end" fill="{MUTED}">below = net densifying</text>')
    return p.svg() + legend([(k, KIND_COLOR[k]) for k in KINDS])


def fig_depth(flips, W, H, PAD):
    """Cumulative flip % against layer index — where is the learning concentrated?"""
    last = max(r["step"] for r in flips)
    rows = sorted((r for r in flips if r["step"] == last), key=lambda r: r["layer"])
    if not rows:
        return "<p>no depth data</p>"
    p = Panel(W, H, PAD, (0, max(r["layer"] for r in rows)),
              (0, max(r["flip_pct"] for r in rows) * 1.15),
              xlabel="layer index", ylabel=f"cumulative flip % @ step {last}", yfmt="{:.1f}")
    placed: list[tuple[float, float]] = []
    for r in rows:
        c = KIND_COLOR.get(r["kind"], MUTED)
        # stem to the baseline: each layer contributes ONE sampled tensor, so these are
        # discrete categories, not a curve. Never connect them.
        p.parts.append(f'<line x1="{p.px(r["layer"]):.1f}" y1="{p.py(0):.1f}" '
                       f'x2="{p.px(r["layer"]):.1f}" y2="{p.py(r["flip_pct"]):.1f}" '
                       f'stroke="{c}" stroke-width="1.5" opacity="0.35"/>')
        p.dots([(r["layer"], r["flip_pct"])], c, 5,
               lambda x, y, r=r: f'{r["tensor"]}: {y:.3f}%')
        lx, ly = p.px(r["layer"]), p.py(r["flip_pct"]) - 9
        while any(abs(lx - a) < 26 and abs(ly - b) < 11 for a, b in placed):
            ly -= 11
        placed.append((lx, ly))
        p.parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" text-anchor="middle" '
                       f'fill="{MUTED}">{r["layer"]}.{r["kind"].replace("_proj", "")}</text>')
    return p.svg() + legend([(k, KIND_COLOR[k]) for k in KINDS])


def fig_efficiency(flips, steps, W, H, PAD, tokens_per_step):
    """Codes changed per GPU-hour — the diminishing-returns curve.

    Wall-clock comes from the trainer's running s/step. For each checkpoint interval we
    take the change in total code changes across tracked tensors and divide by the hours
    that interval took. A falling curve means each additional hour is buying fewer
    changes: the actual "when do I stop?" signal, which loss does not give you (loss
    keeps drifting down on scale alone).
    """
    spm = {r["step"]: r["s_per_step"] for r in steps}
    steps_sorted = sorted(spm)

    def elapsed_h(step):
        near = min(steps_sorted, key=lambda s: abs(s - step))
        return step * spm[near] / 3600.0

    per_step: dict[int, float] = {}
    for r in flips:  # total changed codes across the tracked sample
        per_step[r["step"]] = per_step.get(r["step"], 0.0) + r["flip_pct"] / 100.0 * r["numel"]
    ck = sorted(per_step)
    if len(ck) < 3:
        return "<p>not enough checkpoints</p>"
    pts, tok_pts = [], []
    for a, b in zip(ck, ck[1:], strict=False):
        dh = elapsed_h(b) - elapsed_h(a)
        if dh <= 0:
            continue
        pts.append((b, (per_step[b] - per_step[a]) / dh))
        dt = (b - a) * tokens_per_step
        tok_pts.append((b, (per_step[b] - per_step[a]) / max(1, dt) * 1e6))
    if not pts:
        return "<p>not enough timing data</p>"
    xlim = (min(p[0] for p in pts), max(p[0] for p in pts))
    xt = nice_ticks(*xlim)
    p1 = Panel(W, H, PAD, xlim, (0, max(p[1] for p in pts) * 1.1),
               ylabel="codes changed per GPU-hour (tracked sample)", xticks=xt, yfmt="{:,.0f}")
    p1.line(pts, "#0072b2", 2)
    p1.dots(pts, "#0072b2", 3, lambda x, y: f"step {x:g}: {y:,.0f}/h")
    peak = max(pts, key=lambda t: t[1])
    p1.note(peak[0], peak[1] * 1.04, f"peak {peak[1]:,.0f}/h @ {peak[0]:g}", "middle", INK)
    p2 = Panel(W, 200, PAD, xlim, (0, max(p[1] for p in tok_pts) * 1.1),
               xlabel="training step", ylabel="codes changed per 1M tokens",
               xticks=xt, yfmt="{:,.0f}")
    p2.line(tok_pts, "#009e73", 2)
    p2.dots(tok_pts, "#009e73", 3, lambda x, y: f"step {x:g}: {y:,.0f}/1M tok")
    return p1.svg() + p2.svg()


def fig_zero_fraction(traj, W, H, PAD):
    """Change in zero-fraction per tensor — the capacity view of the same flips."""
    if not traj:
        return "<p>no trajectory data</p>"
    xs = [r["step"] for r in traj]
    d = [(r["zero_frac"] - r["zero_frac_start"]) * 100 for r in traj]
    p = Panel(W, H, PAD, (min(xs), max(xs)), (min(d + [0.0]) * 1.15, max(d + [0.0]) * 1.15),
              xlabel="training step", ylabel="Δ zero-fraction (pp)", yfmt="{:+.2f}")
    p.rule(0.0, "#bbb")
    by: dict[str, list] = {}
    for r in traj:
        by.setdefault(r["tensor"], []).append(r)
    ends = []
    for name, rs in by.items():
        rs.sort(key=lambda r: r["step"])
        pts = [(r["step"], (r["zero_frac"] - r["zero_frac_start"]) * 100) for r in rs]
        c = KIND_COLOR.get(rs[0]["kind"], MUTED)
        p.line(pts, c, 2, 0.85, name)
        ends.append((p.py(pts[-1][1]),
                     f'{rs[0]["layer"]}.{rs[0]["kind"].replace("_proj", "")}', c))
    lx = p.px(max(xs)) + 6
    prev = -1e9
    for y, label, c in sorted(ends):
        ly = max(y + 3, prev + 10)
        prev = ly
        if abs(ly - (y + 3)) > 1:
            p.parts.append(f'<line x1="{lx - 3:.1f}" y1="{y:.1f}" x2="{lx:.1f}" '
                           f'y2="{ly - 3:.1f}" stroke="{c}" stroke-width="1" opacity="0.35"/>')
        p.parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" '
                       f'fill="{MUTED}">{label}</text>')
    return p.svg() + legend([(k, KIND_COLOR[k]) for k in KINDS])


def dist_table(rows, ref=None):
    """-1/0/+1 composition. With `ref`, adds the zero-fraction delta against it."""
    head = ("<tr><th>tensor</th><th>−1</th><th>0</th><th>+1</th>"
            + ("<th>Δ0 (pp)</th>" if ref else "") + "</tr>")
    body = []
    for r in rows:
        cells = (f"<td>{r['neg_frac'] * 100:.2f}%</td>"
                 f"<td><b>{r['zero_frac'] * 100:.2f}%</b></td>"
                 f"<td>{r['pos_frac'] * 100:.2f}%</td>")
        if ref:
            b = ref.get(r["tensor"])
            dv = (r["zero_frac"] - b["zero_frac"]) * 100 if b else 0.0
            cells += f'<td style="color:{"#0072b2" if dv < 0 else "#d55e00"}">{dv:+.3f}</td>'
        body.append(f"<tr><td>{r['tensor'].replace('model.layers.', '')}</td>{cells}</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def fig_throughput(steps, W, H, PAD):
    """s/step and MPS resident memory — the run's stability trace."""
    xs = [r["step"] for r in steps]
    xlim = (min(xs), max(xs))
    xt = nice_ticks(*xlim)
    sp = [r["s_per_step"] for r in steps]
    p1 = Panel(W, 200, PAD, xlim, (min(sp) * 0.98, max(sp) * 1.02),
               ylabel="s/step (running mean)", xticks=xt, yfmt="{:.0f}")
    p1.line([(r["step"], r["s_per_step"]) for r in steps], "#cc79a7", 2)
    mem = [r["mem_gib"] for r in steps]
    p2 = Panel(W, 190, PAD, xlim, (min(mem) * 0.9, max(mem) * 1.1),
               xlabel="training step", ylabel="MPS resident (GiB)", xticks=xt, yfmt="{:.1f}")
    p2.line([(r["step"], r["mem_gib"]) for r in steps], "#e69f00", 2)
    return p1.svg() + p2.svg()


# ---------------------------------------------------------------- io


def read(p: Path, casts: dict) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for row in csv.DictReader(fh):
            r = dict(row)
            for k, fn in casts.items():
                if k in r:
                    try:
                        r[k] = fn(r[k]) if r[k] not in ("", "None") else None
                    except ValueError:
                        r[k] = None
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", type=Path, required=True,
                    help="directory holding steps.csv / val.csv / flips.csv")
    ap.add_argument("--census", type=Path,
                    help="step-0 census CSV (ternary_distribution.py census --model ...)")
    ap.add_argument("--latest", type=Path, help="latest-step census CSV (census --latents ...)")
    ap.add_argument("--latest-step", type=int)
    ap.add_argument("--window", type=int, default=8064)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--title", default="QAT training dynamics")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t = args.telemetry
    steps = read(t / "steps.csv", {"step": int, "loss": float, "lr": float,
                                   "mem_gib": float, "s_per_step": float})
    vals = read(t / "val.csv", {"step": int, "val_masked_ce": float})
    flips = read(t / "flips.csv", {"step": int, "flip_pct": float, "zero_to_nonzero": int,
                                   "nonzero_to_zero": int, "scale_drift_pct": float,
                                   "flip_pct_delta": float, "densify_ratio": float})
    fl = {"numel": int, "layer": int, "neg_frac": float, "zero_frac": float, "pos_frac": float}
    cen_rows = read(args.census or (t / "census.csv"), fl)
    latest = read(args.latest, fl) if args.latest else []
    cen = {r["tensor"]: r for r in cen_rows}
    for r in flips:  # numel/layer/kind live in the census, not the log
        b = cen.get(r["tensor"])
        r["numel"] = b["numel"] if b else 0
        r["layer"] = b["layer"] if b else -1
        r["kind"] = b["kind"] if b else "?"
    flips = [r for r in flips if r["numel"]]
    traj = trajectory(flips, cen) if cen else []
    cen_rows = sorted(cen_rows, key=lambda r: (r["layer"], r["kind"]))
    latest = sorted(latest, key=lambda r: (r["layer"], r["kind"]))
    if not steps:
        raise SystemExit(f"[report] no steps.csv under {t} — run parse_qat_log.py first")

    W, H, PAD = 900, 300, 60
    tps = args.window * args.grad_accum
    last = steps[-1]
    hrs = last["step"] * last["s_per_step"] / 3600
    kpis = headline(steps, vals, flips, traj, hrs, tps)

    cen = cen_rows
    figs = [
        ("Loss &amp; LR",
         "Is the schedule healthy? Stacked panels, shared x — not a dual axis.",
         fig_loss_lr(steps, vals, W, H, PAD)),
        ("Flip velocity",
         "Still learning? Codes are the only thing that survives export, so this is the "
         "convergence signal — loss falls on scale drift alone. Past the peak = annealing.",
         fig_velocity(flips, W, H, PAD)),
        ("Capacity: Δ zero-fraction",
         "Below the line = dead weights switched on. This is the same event as a flip, "
         "counted as capacity rather than churn.",
         fig_zero_fraction(traj, W, H, PAD)),
        ("Mechanism: recruit vs prune",
         "On the dashed diagonal a tensor substitutes weights at constant density; below "
         "it, it recruits. Square axes — the 45° reading only holds at equal aspect.",
         fig_recruit_prune(flips, W, H, PAD)),
        ("Depth profile",
         "One sampled tensor per layer. A big spread means a cheaper partial-layer run may "
         "buy the same thing.",
         fig_depth(flips, W, H, PAD)),
        ("Efficiency",
         "Codes changed per GPU-hour and per 1M tokens. The stop signal: when this "
         "flattens, more hours stop changing the shipped model.",
         fig_efficiency(flips, steps, W, H, PAD, tps)),
        ("Throughput &amp; memory",
         "Rising s/step at flat resident memory = allocator fragmentation heading for swap.",
         fig_throughput(steps, W, H, PAD)),
    ]
    body = "".join(f"<section><h2>{i + 1}. {name}</h2><p>{desc}</p>{svg}</section>"
                   for i, (name, desc, svg) in enumerate(figs))

    tables = ""
    if cen:
        ref = {r["tensor"]: r for r in cen}
        later = (f"<div><h3>step {args.latest_step or 'latest'}</h3>"
                 f"{dist_table(latest, ref)}</div>") if latest else ""
        tables = (
            f"<section><h2>{len(figs) + 1}. Code distribution</h2>"
            f"<p>Per-tensor −1 / 0 / +1 split. The zero column is unused capacity; "
            f"Δ0 negative means training switched weights on.</p>"
            f'<div class="cols"><div><h3>step 0 (as shipped)</h3>{dist_table(cen)}</div>'
            f"{later}</div></section>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html><meta charset="utf-8">
<title>{args.title}</title>
<style>
 body{{font:14px/1.55 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:1000px;
       color:{INK};padding:0 1rem}}
 h1{{font-size:21px;margin:0 0 .2rem}} h2{{font-size:15px;margin:0 0 .2rem}}
 h3{{font-size:12px;color:{MUTED};font-weight:600;margin:0 0 .3rem;text-transform:uppercase;
     letter-spacing:.04em}}
 section{{margin:2rem 0}} p{{color:#555;margin:.1rem 0 .5rem;max-width:78ch}}
 svg{{display:block}}
 .legend{{margin:.3rem 0 0;font-size:11px;color:#666}}
 .legend span{{margin-right:14px;white-space:nowrap}}
 .legend i{{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:-1px}}
 .meta{{font-size:12px;color:{MUTED};margin-bottom:1.2rem}}
 /* fixed 3 columns: 6 metrics in a flex row wrap 5+1 and leave an orphan */
 .kpi{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid #eee;
       border-radius:6px;overflow:hidden;margin:0 0 .5rem}}
 .kpi div{{padding:.55rem .8rem;border-right:1px solid #eee;border-top:1px solid #eee}}
 .kpi div:nth-child(3n){{border-right:0}}
 .kpi div:nth-child(-n+3){{border-top:0}}
 .kpi b{{display:block;font-size:17px;font-weight:600;letter-spacing:-.01em}}
 .kpi span{{font-size:11px;color:{MUTED}}}
 .cols{{display:flex;gap:2rem;flex-wrap:wrap}}
 table{{border-collapse:collapse;font-size:12px;margin:0}}
 td,th{{border-bottom:1px solid #eee;padding:3px 10px;text-align:right}}
 th{{color:{MUTED};font-weight:500}}
 td:first-child,th:first-child{{text-align:left;font-family:ui-monospace,monospace}}
</style>
<h1>{args.title}</h1>
<p class="meta">step {last['step']}/{last.get('total_steps', '?')} ·
 {hrs:.1f} GPU-h · {last['s_per_step']:.0f} s/step · {tps:,} tok/step ·
 {last['step'] * tps / 1e6:.1f}M tokens seen</p>
{kpis}
<p class="meta">A natively-ternary model stores <code>w = s·c</code>, <code>c ∈ {{−1,0,+1}}</code>.
Loss can fall on scale drift alone, so every panel below is anchored on <b>code flips</b> —
the only change that survives export to a 2-bit GGUF.</p>
{body}
{tables}
""")
    print(f"[report] {len(figs)} figures{' + tables' if tables else ''} -> {args.out}")
    return 0


def headline(steps, vals, flips, traj, hrs, tps) -> str:
    """The six numbers to read before any chart."""
    cells: list[tuple[str, str]] = []
    last = steps[-1]
    cells.append((f"{last['loss']:.3f}", "train loss"))
    if vals:
        best = min(vals, key=lambda v: v["val_masked_ce"])
        cells.append((f"{vals[-1]['val_masked_ce']:.3f}",
                      f"val (best {best['val_masked_ce']:.3f} @ {best['step']})"))
    if flips:
        at = max(r["step"] for r in flips)
        tail = [r for r in flips if r["step"] == at]
        changed = sum(r["flip_pct"] / 100 * r["numel"] for r in tail)
        total = sum(r["numel"] for r in tail)
        cells.append((f"{changed / total * 100:.2f}%", "codes changed (tracked)"))
        cells.append((f"{max(r['flip_pct'] for r in tail):.2f}%", "most-changed tensor"))
        # efficiency now vs peak — the "should I keep going" number
        per = {}
        for r in flips:
            per[r["step"]] = per.get(r["step"], 0.0) + r["flip_pct"] / 100 * r["numel"]
        ck = sorted(per)
        rate = []
        for a, b in zip(ck, ck[1:], strict=False):
            dh = (b - a) * last["s_per_step"] / 3600
            if dh > 0:
                rate.append((b, (per[b] - per[a]) / dh))
        if rate:
            peak = max(rate, key=lambda t: t[1])
            cells.append((f"{rate[-1][1] / peak[1] * 100:.0f}%",
                          f"of peak efficiency (peak @ {peak[0]:g})"))
    if traj:
        at = max(r["step"] for r in traj)
        d = [(r["zero_frac"] - r["zero_frac_start"]) * 100 for r in traj if r["step"] == at]
        if d:
            cells.append((f"{sum(d) / len(d):+.3f}pp", "mean Δ zero-fraction"))
    return ('<div class="kpi">'
            + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for v, k in cells)
            + "</div>")


if __name__ == "__main__":
    raise SystemExit(main())

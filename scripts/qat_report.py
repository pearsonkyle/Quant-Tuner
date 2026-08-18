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
import json
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
#: assistant turns a run needs before its repetition fraction means anything
MIN_LOOP_TURNS = 8
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


def fig_kd_kl(steps, alpha, W, H, PAD):
    """KL(teacher‖student) over training — the teacher-tracking signal KD adds.

    Falling KL = the student's distribution is moving toward the teacher's. This is the
    quantity CE cannot express: CE sees one target per position, the KL the whole shape,
    and the shape is what pins the termination policy. With ``alpha`` given, the CE
    component is derived from the logged total — CE = (loss − α·KL)/(1 − α), exact at
    T=1 — so both terms read on one axis without a second log pass.
    """
    rows = [r for r in steps if r.get("kd_kl") is not None]
    xs = [r["step"] for r in rows]
    xlim = (min(xs), max(xs)) if len(xs) > 1 else (min(xs), min(xs) + 1)
    kl = [(r["step"], r["kd_kl"]) for r in rows]
    ce = []
    if alpha is not None and 0.0 < alpha < 1.0:
        ce = [(r["step"], (r["loss"] - alpha * r["kd_kl"]) / (1.0 - alpha))
              for r in rows]
    ymax = max([v for _, v in kl] + [v for _, v in ce] + [0.1]) * 1.15
    p = Panel(W, H, PAD, xlim, (0, ymax), xlabel="training step", ylabel="nats",
              xticks=nice_ticks(*xlim), yfmt="{:.2f}")
    p.line(kl, "#0072b2", 2.2, 1.0, "KL(teacher‖student)")
    p.dots([kl[-1]], "#0072b2", 3,
           lambda x, y: f"step {x:g}: KL {y:.4f}")
    if ce:
        p.line(ce, "#e69f00", 1.6, 0.85, "CE (derived)")
    p.note(kl[-1][0], min(kl[-1][1] * 0.85, ymax * 0.9),
           f"KL {kl[0][1]:.3f} → {kl[-1][1]:.3f}", "end", INK)
    items = [("KL(teacher‖student)", "#0072b2")]
    if ce:
        items.append(("CE (derived)", "#e69f00"))
    return p.svg() + legend(items)


# Okabe-Ito, distinguishable in both common colour-vision deficiencies. The diagnostic and
# the control get the two strongest hues because they are the two lines a reader must
# separate at a glance; the other three are context.
PROBE_COLORS = {
    "sentence_period": "#d55e00",   # THE diagnostic — high is broken
    "after_tool_call": "#009e73",   # the control — high is CORRECT
    "sentence_newline": "#cc79a7",
    "start": "#0072b2",
    "mid_sentence": "#56b4e9",
}
# Measured on the shipped weights through this same torch path.
VANILLA_REF = {"sentence_period": 0.0017, "after_tool_call": 0.99996}


def fig_stop_probe(probes, W, H, PAD, teacher=None):
    """P(<|im_end|>) at fixed positions, over training.

    The termination collapse this pipeline keeps hitting is invisible to masked-CE — a
    model scoring well on the corpus's own continuations can still put 0.95 on the stop
    token after a single sentence, and sft32k's validation went FLAT for 225 steps while
    exactly that happened. So this is plotted on its own axis, log-scaled because the
    healthy and broken regimes are three orders of magnitude apart and a linear axis would
    render every healthy value as the same flat line on zero.

    Read `sentence_period` (should stay LOW, ~0.002) against `after_tool_call` (should stay
    HIGH, ~1.0). Losing either is a failure; losing both is the loss of position-dependence.
    """
    keys = [k for k in PROBE_COLORS if any(r.get(k) is not None for r in probes)]
    xs = [r["step"] for r in probes]
    xlim = (min(xs), max(xs)) if len(xs) > 1 else (min(xs), min(xs) + 1)
    lo = min([v for r in probes for k in keys
              if (v := r.get(k)) not in (None, "")] + [1e-4])
    p1 = Panel(W, H, PAD, xlim, (max(lo * 0.5, 1e-6), 1.6), logy=True,
               xlabel="training step", ylabel="P(<|im_end|>)  (log)",
               xticks=nice_ticks(*xlim), yfmt="{:g}")
    # Reference bands first, so the data draws over them. For a KD run the teacher's
    # own probe values (dotted) are the asymptote the KL pulls toward — the vanilla
    # lines (dashed) are where the student STARTED, not where it should end.
    refsets = [("vanilla", VANILLA_REF, "4 4", "start")]
    if teacher:
        refsets.append(("teacher", teacher, "1 3", "end"))
    for label, refs, dash, anchor in refsets:
        for name, ref in refs.items():
            if name not in keys:
                continue
            y = p1.py(max(ref, 1e-6))
            xa = PAD + 4 if anchor == "start" else W - PAD - 4
            p1.parts.append(
                f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" '
                f'y2="{y:.1f}" stroke="{PROBE_COLORS[name]}" stroke-width="1" '
                f'stroke-dasharray="{dash}" opacity="0.45"/>')
            p1.parts.append(
                f'<text x="{xa}" y="{y - 4:.1f}" font-size="9" '
                f'text-anchor="{anchor}" fill="{PROBE_COLORS[name]}" opacity="0.8">'
                f'{label} {name} {"<1e-6" if ref < 1e-6 else f"{ref:g}"}</text>')
    for k in keys:
        pts = [(r["step"], max(r[k], 1e-6)) for r in probes
               if r.get(k) not in (None, "")]
        if not pts:
            continue
        wide = k in VANILLA_REF
        p1.line(pts, PROBE_COLORS[k], 2.2 if wide else 1.4, 0.95 if wide else 0.75, k)
        p1.dots(pts, PROBE_COLORS[k], 3 if wide else 2,
                lambda x, y, _k=k: f"step {x:g}: P({_k}) = {y:.5f}")
    return p1.svg() + legend([(k, PROBE_COLORS[k]) for k in keys])


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
    ap.add_argument("--arch-repo", default="prism-ml/Ternary-Bonsai-8B-unpacked",
                    help="HF repo for the hfviewer architecture graph "
                         "(default: the unpacked view of the model this pipeline trains)")
    ap.add_argument("--model-config", type=Path,
                    help="config.json to build the shape table from")
    ap.add_argument("--no-arch", action="store_true", help="skip the architecture section")
    ap.add_argument("--swe-workspace", type=Path,
                    help="run_swebench_eval workspace — adds the agentic outcome table")
    ap.add_argument("--stop-prob-csv", type=Path,
                    help="probe_stop_prob.py CSV — adds the termination-policy table")
    ap.add_argument("--notes", type=Path,
                    help="a text file of findings/next-steps; '## ' starts a heading, "
                         "'- ' a bullet, blank lines separate paragraphs")
    ap.add_argument("--window", type=int, default=8064)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--kd-alpha", type=float, default=None,
                    help="the run's KD mixing weight; lets the KD panel derive the CE "
                         "component from the logged total (exact at T=1)")
    ap.add_argument("--teacher-probe", action="append", default=[],
                    metavar="NAME=PROB",
                    help="teacher's own stop-probe reading (repeatable, e.g. "
                         "after_tool_call=0.99999) — drawn as dotted asymptote lines "
                         "on the termination panel")
    ap.add_argument("--title", default="QAT training dynamics")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t = args.telemetry
    steps = read(t / "steps.csv", {"step": int, "loss": float, "kd_kl": float,
                                   "lr": float, "mem_gib": float, "s_per_step": float})
    vals = read(t / "val.csv", {"step": int, "val_masked_ce": float})
    probes = read(t / "stopprobe.csv",
                  {"step": int, **{k: float for k in PROBE_COLORS}})
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
    # Flip telemetry is only printed at checkpoints, so a report generated in the first
    # interval of a run has none. That is the normal state of a live run, not an error —
    # emit the curves that do exist and say why the rest are missing, rather than dying on
    # max() of an empty sequence.
    figs = [
        ("Loss &amp; LR",
         "Is the schedule healthy? Stacked panels, shared x — not a dual axis.",
         fig_loss_lr(steps, vals, W, H, PAD)),
    ]
    if any(r.get("kd_kl") is not None for r in steps):
        figs += [
            ("Distillation: KL to the teacher",
             "The teacher-tracking signal offline KD adds. Falling KL = the student's "
             "distribution moving toward the teacher's — the SHAPE constraint that pins "
             "the termination policy, which one-target-per-position CE cannot express.",
             fig_kd_kl(steps, args.kd_alpha, W, H, PAD)),
        ]
    if probes:
        figs += [
            ("Termination policy over training",
             "P(&lt;|im_end|&gt;) at five fixed positions, measured on the live model. "
             "<b>sentence_period</b> is the diagnostic (must stay LOW) and "
             "<b>after_tool_call</b> the control (must stay HIGH). Masked-CE cannot see "
             "this: sft32k's validation was flat for 225 steps while its "
             "sentence_period went to 0.97.",
             fig_stop_probe(probes, W, H, PAD,
                            teacher={k: float(v) for k, v in
                                     (s.split("=", 1) for s in args.teacher_probe)})),
        ]
    if flips:
        figs += [
            ("Flip velocity",
             "Still learning? Codes are the only thing that survives export, so this is the "
             "convergence signal — loss falls on scale drift alone. Past the peak = annealing.",
             fig_velocity(flips, W, H, PAD)),
        ]
        # Δ zero-fraction is the one panel that needs per-tensor density, which lives in
        # the census rather than the log.
        if traj:
            figs += [
                ("Capacity: Δ zero-fraction",
                 "Below the line = dead weights switched on. This is the same event as a "
                 "flip, counted as capacity rather than churn.",
                 fig_zero_fraction(traj, W, H, PAD)),
            ]
        figs += [
            ("Mechanism: recruit vs prune",
             "On the dashed diagonal a tensor substitutes weights at constant density; below "
             "it, it recruits. Square axes — the 45° reading only holds at equal aspect.",
             fig_recruit_prune(flips, W, H, PAD)),
            ("Depth profile",
             "One sampled tensor per layer. A big spread means a cheaper partial-layer run "
             "may buy the same thing.",
             fig_depth(flips, W, H, PAD)),
            ("Efficiency",
             "Codes changed per GPU-hour and per 1M tokens. The stop signal: when this "
             "flattens, more hours stop changing the shipped model.",
             fig_efficiency(flips, steps, W, H, PAD, tps)),
        ]
    figs += [
        ("Throughput &amp; memory",
         "Rising s/step at flat resident memory = allocator fragmentation heading for swap.",
         fig_throughput(steps, W, H, PAD)),
    ]
    pending = "" if flips else (
        '<p class="meta">Code-flip figures appear from the first checkpoint onward — this '
        "run has not reached one yet, so only the schedule and throughput curves are "
        "shown. Their absence says nothing about whether codes are moving.</p>")
    body = "".join(f"<section><h2>{i + 1}. {name}</h2><p>{desc}</p>{svg}</section>"
                   for i, (name, desc, svg) in enumerate(figs))

    arch = ""
    if not args.no_arch:
        spec = ""
        if args.model_config and args.model_config.exists():
            spec = arch_spec(json.loads(args.model_config.read_text()))
        arch = (f'<section><h2>Architecture</h2>'
                f'<p>What the flips below are distributed over — the ternarization is a weight '
                f'format, not an architecture change. Shapes are read from the model\'s '
                f'own <code>config.json</code>.</p>'
                f'<div class="cols"><div>{spec}</div>'
                f'<div class="archwrap">{arch_card(args.arch_repo)}</div></div></section>')

    stopsec = ""
    if args.stop_prob_csv:
        t_ = stop_prob_section(args.stop_prob_csv)
        if t_:
            stopsec = (
                f"<section><h2>Termination policy — P(&lt;|im_end|&gt;)</h2>"
                f"<p>The endpoint the loss curve cannot show. Probability of the stop token "
                f"at fixed points in one agentic turn, greedy, read straight off "
                f"<code>/completion</code> logprobs. Rank is in parentheses.</p>"
                f"<p>During an agentic turn the correct continuation is a tool call, so a "
                f"healthy model keeps P(stop) low everywhere except after a complete "
                f"<code>&lt;/tool_call&gt;</code> block — the one point where stopping is "
                f"right, and the only column where a high value is good. "
                f"<b>sentence_period</b> is the diagnostic: a model that has learned to stop "
                f"too early turns every sentence boundary into an absorbing state.</p>"
                f"{t_}</section>")

    swe = ""
    if args.swe_workspace:
        t_ = swe_section(args.swe_workspace)
        if t_:
            swe = (f"<section><h2>Agentic outcome (SWE-rebench)</h2>"
                   f"<p><b>resolved</b> is the only column that says the model got better "
                   f"at the task; the rest say how it behaved getting there. <b>loop</b> = "
                   f"largest share of a run's assistant turns that are one repeated "
                   f"message (runs of &lt;{MIN_LOOP_TURNS} turns are 'n/a' — too short "
                   f"for the fraction to mean anything).</p>{t_}</section>")

    notes = ""
    if args.notes and args.notes.exists():
        buf = []
        for line in args.notes.read_text().splitlines():
            ln = line.rstrip()
            if ln.startswith("## "):
                buf.append(f"<h3>{ln[3:]}</h3>")
            elif ln.startswith("- "):
                buf.append(f"<li>{ln[2:]}</li>")
            elif ln:
                buf.append(f"<p>{ln}</p>")
        html = "".join(buf).replace("<li>", "<ul><li>", 1)
        notes = f"<section><h2>Findings &amp; next steps</h2>{html}</ul></section>"

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
 /* This page commits to a light ground on purpose: the Okabe-Ito series colors are chosen
    for contrast against white, and re-deriving them per theme would break comparability
    with every published figure. So paint it explicitly rather than inheriting — an
    unpainted body renders #222 text on the host's ground, which is unreadable anywhere
    the page is embedded in a dark context. */
 :root{{color-scheme:light}}
 body{{font:14px/1.55 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:1000px;
       color:{INK};background:#fff;padding:0 1rem}}
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
 .cols{{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start}}
 /* the global 78ch on p would force this column past the wrap point */
 .cols p{{max-width:44ch}} .cols>div{{max-width:540px}}
 li{{color:#444;margin:.15rem 0;max-width:78ch}} ul{{padding-left:1.1rem}}
 h3{{margin-top:1rem}}
 .archwrap{{flex:0 1 380px;min-width:280px}}
 .archwrap svg,.archwrap img{{max-height:460px;max-width:100%;width:auto;height:auto;
   display:block;border-radius:8px}}
 a.arch{{display:block}}
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
{arch}
<p class="meta">A natively-ternary model stores <code>w = s·c</code>, <code>c ∈ {{−1,0,+1}}</code>.
Loss can fall on scale drift alone, so every panel below is anchored on <b>code flips</b> —
the only change that survives export to a 2-bit GGUF.</p>
{pending}
{body}
{tables}
{stopsec}
{swe}
{notes}
""")
    print(f"[report] {len(figs)} figures{' + tables' if tables else ''} -> {args.out}")
    return 0


HFVIEWER = ("https://hfviewer.com/api/card.svg?source={src}&granularity=auto"
            "&v=20260516-title-pills-card")
HFVIEWER_PAGE = ("https://hfviewer.com/{repo}?utm_source=huggingface"
                 "&utm_medium=embedded_model_card&utm_campaign={camp}_card"
                 "&utm_content=embedded_card_open_viewer&from=embedded-model-card")


def arch_card(repo: str, timeout: float = 20.0) -> str:
    """hfviewer's architecture graph, inlined when it is a real graph.

    Inlining keeps the report self-contained and offline-readable, and drops the
    tracking query string from the rendered page. But hfviewer answers 200 with a
    ~950-byte placeholder SVG ("Graph temporarily unavailable" / "not available yet")
    for repos it has not indexed — baking that in would freeze a broken graph into the
    document. So: inline a real graph, otherwise fall back to the live <img> so the
    report starts working by itself once the graph exists.
    """
    import urllib.parse
    import urllib.request

    src = urllib.parse.quote(repo, safe="")
    page = HFVIEWER_PAGE.format(repo=repo, camp=repo.replace("/", "__"))
    svg = ""
    try:
        with urllib.request.urlopen(HFVIEWER.format(src=src), timeout=timeout) as r:
            svg = r.read().decode("utf-8", "replace")
    except (OSError, ValueError) as e:
        print(f"[report] architecture card fetch failed ({e}); using the live embed",
              flush=True)
    placeholder = ("aria-label" in svg and "vailable" in svg) or len(svg) < 2000
    if svg and not placeholder:
        # strip any XML prolog so it can sit inline in HTML
        svg = svg[svg.index("<svg"):]
        return (f'<a href="{page}" target="_blank" rel="noopener" class="arch">{svg}</a>'
                f'<p class="meta">Graph: hfviewer, inlined. '
                f'<code>{repo}</code></p>')
    if svg and placeholder:
        print(f"[report] hfviewer has no graph for {repo} yet — embedding the live card",
              flush=True)
    return (f'<a href="{page}" target="_blank" rel="noopener">'
            f'<img src="{HFVIEWER.format(src=src)}" width="100%" '
            f'alt="Architecture graph for {repo}. Open in hfviewer"/></a>'
            f'<p class="meta">Graph: hfviewer (loaded live — not yet available for '
            f'inlining). <code>{repo}</code></p>')


def arch_spec(cfg: dict) -> str:
    """Shape table straight from the model's own config.json.

    Verified locally rather than taken from the card: this is the config the run is
    actually training against.
    """
    h, ff = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", h // nh)
    nl = cfg["num_hidden_layers"]
    per_layer = 2 * h * nh * hd + 2 * h * nkv * hd + 3 * h * ff   # q,o + k,v + gate,up,down
    embed = cfg["vocab_size"] * h * (1 if cfg.get("tie_word_embeddings") else 2)
    # Verified against Qwen/Qwen3-8B's config.json: every shape below is identical
    # except vocab_size (151,936 -> 151,669) and max_position_embeddings
    # (40,960 -> 65,536). Nothing the flip analysis depends on differs.
    stock = {"vocab_size": 151936, "max_position_embeddings": 40960}
    delta = [k for k, v in stock.items() if cfg.get(k) != v]
    rows = [
        ("layers", f"{nl}"),
        ("hidden / FFN", f"{h} / {ff}"),
        ("attention", f"{nh} heads × {hd} (GQA {nh // nkv}:1, {nkv} KV)"),
        ("vocab / ctx", f"{cfg['vocab_size']:,} / {cfg['max_position_embeddings']:,}"
                        + (" ✻" if delta else "")),
        ("act / norm", f"{cfg.get('hidden_act', '?')} · RMSNorm {cfg.get('rms_norm_eps')}"),
        ("rope θ", f"{cfg.get('rope_theta', 0):,.0f}"),
        ("ternary linears", f"{nl * 7} ({per_layer * nl / 1e9:.2f}B weights)"),
        ("non-ternary", f"{embed / 1e9:.2f}B embed/head + norms (frozen)"),
    ]
    note = ('<p class="meta">Stock <code>Qwen3ForCausalLM</code>. ✻ = the only shapes that '
            'differ from <code>Qwen/Qwen3-8B</code> (vocab 151,936; ctx 40,960); every '
            'other dimension is identical.</p>') if delta else ""
    return ("<table>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
            + "</table>" + note)


def stop_prob_section(path: Path) -> str:
    """P(<|im_end|>) at fixed probe points, one column per model.

    The endpoint no loss curve can report. A ternary QAT run can leave grammar intact and
    still move the *stopping policy*: the sft32k run (--stop-weight 6.0) writes one correct
    sentence and then emits the stop token instead of the tool call, which shows up here as
    P(stop) going to ~1.0 at `sentence_period` while `mid_sentence` stays near zero.

    Written by scripts/probe_stop_prob.py. Rows are probe points, columns are models in the
    order they were measured, so the reference sits beside the run under test.
    """
    if not path.exists():
        return ""
    rows = [r for r in csv.DictReader(path.open()) if r.get("probe")]
    if not rows:
        return ""

    labels = list(dict.fromkeys(r["label"] for r in rows))
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    by = {(r["label"], r["probe"]): r for r in rows}

    out = ["<tr><th>probe point</th>" + "".join(f"<th>{lab}</th>" for lab in labels) + "</tr>"]
    for p in probes:
        cells = []
        for lab in labels:
            r = by.get((lab, p))
            if r is None:
                cells.append(f'<td style="color:{MUTED}">—</td>')
                continue
            raw = (r.get("stop_prob") or "").strip()
            if not raw:
                # Outside the top-N window: the tail bound is the honest report.
                try:
                    bound = f"&lt;{float(r['tail_bound']):.0e}"
                except (KeyError, ValueError):
                    bound = "&lt;tail"
                cells.append(f'<td style="color:{MUTED}">{bound} '
                             f'<span>(r&gt;{r.get("n_returned", "?")})</span></td>')
                continue
            v = float(raw)
            # after_tool_call is the one point where stopping is CORRECT, so a high value
            # there is the healthy reading and must not be flagged as a regression.
            broken = v > 0.5 and p != "after_tool_call"
            col = "#d55e00" if broken else INK
            # The healthy values run to 1e-7; fixed decimals would print them all as
            # 0.00000 and lose the three orders that separate "never stops here" from
            # "occasionally stops here".
            shown = f"{v:.5f}" if v >= 1e-4 else f"{v:.1e}"
            cells.append(f'<td style="color:{col}">{shown}'
                         f'<span style="color:{MUTED}"> (r{r.get("stop_rank", "?")})</span></td>')
        out.append(f"<tr><td>{p}</td>{''.join(cells)}</tr>")
    return "<table>" + "".join(out) + "</table>"


def swe_section(ws: Path) -> str:
    """Agentic SWE-rebench outcome table, including the loop metric.

    `resolved` is the only number that says the model got better at the task; the rest
    say how it behaved getting there. `loop` is the fraction of a run's assistant turns
    that are the single most-repeated message — a model that cannot terminate scores
    near 1.0 and burns its whole token budget saying the same thing.
    """
    import collections

    res = ws / "results.csv"
    if not res.exists():
        return ""
    T = lambda v: str(v).strip().lower() in ("true", "1", "yes")  # noqa: E731
    rows = list(csv.DictReader(res.open()))
    out = ["<tr><th>model</th><th>resolved</th><th>patch</th><th>tool&nbsp;err</th>"
           "<th>steps</th><th>tokens</th><th>max_turns</th><th>loop</th></tr>"]
    for m in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == m]
        n = len(rs)
        def f(k, _rs=rs, _n=n):
            return sum(float(r.get(k) or 0) for r in _rs) / _n
        mt = sum(1 for r in rs if r.get("exit_status") == "max_turns")
        loops = []
        for r in rs:
            t = ws / "trajectories" / m.replace(".gguf", "") / f"{r['instance_id']}.traj.json"
            if not t.exists():
                continue
            blob = json.loads(t.read_text())
            # backends differ: mini-swe writes "trajectory", openai-agents "messages"
            msgs = (blob if isinstance(blob, list)
                    else (blob.get("trajectory") or blob.get("messages") or []))
            c = [str(x.get("content"))[:150] for x in msgs
                 if isinstance(x, dict) and x.get("role") == "assistant" and x.get("content")]
            # A run with 1-3 turns is trivially "100% repeated"; requiring a floor is
            # what keeps a model that quits immediately from outscoring one that loops.
            if len(c) >= MIN_LOOP_TURNS:
                loops.append(collections.Counter(c).most_common(1)[0][1] / len(c))
        loop = f"{max(loops):.2f}" if loops else "n/a"
        tool = f("tool_errors") / max(1e-9, f("tools_used"))
        out.append(
            f"<tr><td>{m.replace('.gguf', '')}</td>"
            f"<td><b>{sum(T(r['resolved']) for r in rs)}/{n}</b></td>"
            f"<td>{sum(T(r['patch_produced']) for r in rs)}/{n}</td>"
            f"<td>{tool:.2f}</td><td>{f('tools_used'):.0f}</td>"
            f"<td>{f('total_tokens'):,.0f}</td><td>{mt}/{n}</td><td>{loop}</td></tr>")
    return f"<table>{''.join(out)}</table>"


def headline(steps, vals, flips, traj, hrs, tps) -> str:
    """The six numbers to read before any chart."""
    cells: list[tuple[str, str]] = []
    last = steps[-1]
    cells.append((f"{last['loss']:.3f}", "train loss"))
    klrows = [r for r in steps if r.get("kd_kl") is not None]
    if klrows:
        cells.append((f"{klrows[-1]['kd_kl']:.3f}",
                      f"KL to teacher (start {klrows[0]['kd_kl']:.3f})"))
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

#!/usr/bin/env python
"""Training-dynamics report for a ternary QAT run.

Five figures a loss curve cannot give you, each answering a question that comes up when
deciding whether to keep spending GPU-hours on a run:

  1. **Loss & LR**       — did the schedule cause the excursion? (stacked panels, never a
                           second y-axis: two scales on one frame is the classic lie)
  2. **Flip velocity**   — is the model still changing, or has it converged?
  3. **Recruit vs prune**— WHICH mechanism: switching dead weights on, or substituting?
  4. **Depth profile**   — where in the stack is the learning happening?
  5. **Efficiency**      — codes changed per GPU-hour. The diminishing-returns curve, and
                           the number that says when to stop.

Inputs are the CSVs from `parse_qat_log.py` plus the censuses from
`ternary_distribution.py`. Pure CSV in, one self-contained HTML out — no model load.

    python scripts/qat_report.py --telemetry out/exp-058/telemetry \\
        --window 8064 --grad-accum 4 --out out/exp-058/telemetry/report.html

Categorical colors are Okabe-Ito, a published colorblind-safe qualitative palette;
assigned to tensor kind in fixed order so a kind keeps its color across every figure.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

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
                 xticks=None, yfmt="{:.2f}"):
        self.w, self.h, self.pad = w, h, pad
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
        return self.h - self.pad - (v - self.y0) / max(1e-9, self.y1 - self.y0) \
            * (self.h - 2 * self.pad)

    def _grid(self, xticks, yfmt, xlabel, ylabel):
        for t in (xticks if xticks is not None else nice_ticks(self.x0, self.x1)):
            if not self.x0 - 1e-9 <= t <= self.x1 + 1e-9:
                continue
            self.parts.append(f'<line x1="{self.px(t):.1f}" y1="{self.pad}" '
                              f'x2="{self.px(t):.1f}" y2="{self.h - self.pad}" '
                              f'stroke="{GRID}" stroke-width="1"/>')
            lab = f"{t:g}"
            self.parts.append(f'<text x="{self.px(t):.1f}" y="{self.h - self.pad + 14}" '
                              f'font-size="10" text-anchor="middle" fill="{MUTED}">{lab}</text>')
        lo, hi = (10 ** self.y0, 10 ** self.y1) if self.logy else (self.y0, self.y1)
        ticks = ([10 ** e for e in range(math.floor(self.y0), math.ceil(self.y1) + 1)]
                 if self.logy else nice_ticks(lo, hi))
        for t in ticks:
            y = self.py(t)
            if not self.pad - 1 <= y <= self.h - self.pad + 1:
                continue
            self.parts.append(f'<line x1="{self.pad}" y1="{y:.1f}" x2="{self.w - self.pad}" '
                              f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            self.parts.append(f'<text x="{self.pad - 6}" y="{y + 3:.1f}" font-size="10" '
                              f'text-anchor="end" fill="{MUTED}">{yfmt.format(t)}</text>')
        if xlabel:
            self.parts.append(f'<text x="{self.w / 2:.0f}" y="{self.h - 4}" font-size="11" '
                              f'text-anchor="middle" fill="{MUTED}">{xlabel}</text>')
        if ylabel:
            self.parts.append(f'<text x="{self.pad - 6}" y="{self.pad - 10}" font-size="11" '
                              f'fill="{MUTED}">{ylabel}</text>')

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
    p1.note(peak["step"], peak["loss"] * 1.35, f'peak {peak["loss"]:.1f} @ {peak["step"]}',
            "middle", INK)
    p2 = Panel(W, 130, PAD, xlim, (0, max(r["lr"] for r in steps) * 1.1),
               xlabel="training step", ylabel="learning rate", xticks=xt, yfmt="{:.1e}")
    p2.line([(r["step"], r["lr"]) for r in steps], "#666", 2)
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
    p.parts.append(f'<text x="{W - PAD - 4}" y="{PAD + 14}" font-size="10" '
                   f'text-anchor="end" fill="{MUTED}">above = net pruning</text>')
    p.parts.append(f'<text x="{W - PAD - 4}" y="{H - PAD - 8}" font-size="10" '
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
    p.line([(r["layer"], r["flip_pct"]) for r in rows], "#ccc", 1.5, 1.0)
    for r in rows:
        c = KIND_COLOR.get(r["kind"], MUTED)
        p.dots([(r["layer"], r["flip_pct"])], c, 5,
               lambda x, y, r=r: f'{r["tensor"]}: {y:.3f}%')
        p.note(r["layer"], r["flip_pct"] * 1.06, r["kind"].replace("_proj", ""), "middle")
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
    p2 = Panel(W, 150, PAD, xlim, (0, max(p[1] for p in tok_pts) * 1.1),
               xlabel="training step", ylabel="codes changed per 1M tokens",
               xticks=xt, yfmt="{:,.0f}")
    p2.line(tok_pts, "#009e73", 2)
    p2.dots(tok_pts, "#009e73", 3, lambda x, y: f"step {x:g}: {y:,.0f}/1M tok")
    return p1.svg() + p2.svg()


def fig_throughput(steps, W, H, PAD):
    """s/step and MPS resident memory — the run's stability trace."""
    xs = [r["step"] for r in steps]
    xlim = (min(xs), max(xs))
    xt = nice_ticks(*xlim)
    sp = [r["s_per_step"] for r in steps]
    p1 = Panel(W, 150, PAD, xlim, (min(sp) * 0.98, max(sp) * 1.02),
               ylabel="s/step (running mean)", xticks=xt, yfmt="{:.0f}")
    p1.line([(r["step"], r["s_per_step"]) for r in steps], "#cc79a7", 2)
    mem = [r["mem_gib"] for r in steps]
    p2 = Panel(W, 130, PAD, xlim, (min(mem) * 0.9, max(mem) * 1.1),
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
    cen = {r["tensor"]: r for r in read(t / "census.csv", {"numel": int, "layer": int})}
    for r in flips:  # numel/layer/kind live in the census, not the log
        b = cen.get(r["tensor"])
        r["numel"] = b["numel"] if b else 0
        r["layer"] = b["layer"] if b else -1
        r["kind"] = b["kind"] if b else "?"
    flips = [r for r in flips if r["numel"]]
    if not steps:
        raise SystemExit(f"[report] no steps.csv under {t} — run parse_qat_log.py first")

    W, H, PAD = 900, 300, 60
    tps = args.window * args.grad_accum
    figs = [
        ("Loss and learning rate",
         "Train loss (log scale) with validation overlaid, and the LR schedule beneath it "
         "on a shared x-axis. Separate panels on purpose — a second y-axis would let the "
         "two curves cross wherever the scaling put them.",
         fig_loss_lr(steps, vals, W, H, PAD)),
        ("Flip velocity — is it still learning?",
         "Change in cumulative flip percentage per checkpoint. A ternary model only learns "
         "by flipping codes, so this, not loss, is the signal that a run still has "
         "something left to give. Every line past its peak = converging.",
         fig_velocity(flips, W, H, PAD)),
        ("Mechanism: recruit vs prune",
         "Weights switched on (0 → ±1) against weights switched off (±1 → 0) at the latest "
         "checkpoint, log-log. The dashed diagonal is balanced churn — substitution at "
         "constant density. Distance below it is net recruitment of dead capacity.",
         fig_recruit_prune(flips, W, H, PAD)),
        ("Depth profile",
         "Cumulative flip percentage by layer index. Tells you whether continued training "
         "is reaching the whole stack or concentrating in a few tensors — which is what "
         "decides whether a partial-layer run would have been just as good and far cheaper.",
         fig_depth(flips, W, H, PAD)),
        ("Training efficiency — diminishing returns",
         "Codes changed per GPU-hour, and per million tokens. This is the stop signal: "
         "loss keeps drifting down on scale drift alone, but once this curve flattens, "
         "further hours are not changing the model that ships.",
         fig_efficiency(flips, steps, W, H, PAD, tps)),
        ("Throughput and memory",
         "s/step and MPS resident memory. A monotonically rising s/step at flat resident "
         "memory means allocator fragmentation pushing the working set into swap — the "
         "precursor to an OOM kill on this box.",
         fig_throughput(steps, W, H, PAD)),
    ]
    body = "".join(f"<section><h2>{i + 1}. {name}</h2><p>{desc}</p>{svg}</section>"
                   for i, (name, desc, svg) in enumerate(figs))
    last = steps[-1]
    hrs = last["step"] * last["s_per_step"] / 3600
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html><meta charset="utf-8">
<title>{args.title}</title>
<style>
 body{{font:14px/1.6 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:1000px;
       color:{INK};padding:0 1rem}}
 h1{{font-size:22px;margin-bottom:.2rem}} h2{{font-size:15px;margin:0 0 .3rem}}
 section{{margin:2.5rem 0}} p{{color:#555;margin:.2rem 0 .6rem;max-width:70ch}}
 .legend{{margin:.4rem 0 0;font-size:11px;color:#666}}
 .legend span{{margin-right:14px;white-space:nowrap}}
 .legend i{{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:-1px}}
 .meta{{font-size:12px;color:{MUTED}}}
</style>
<h1>{args.title}</h1>
<p class="meta">step {last['step']}/{last.get('total_steps', '?')} ·
 {hrs:.1f} GPU-hours · {last['s_per_step']:.0f} s/step ·
 {tps:,} tokens/step · {last['step'] * tps / 1e6:.1f}M tokens seen</p>
{body}
""")
    print(f"[report] {len(figs)} figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

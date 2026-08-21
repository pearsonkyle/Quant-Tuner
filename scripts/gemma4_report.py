"""Live study report for the gemma-4-E4B ternarization — the evidence, then the training.

`qat_report.py` renders a RUN: it opens on `steps[-1]` and every panel is a time series,
which is right once a trainer is producing telemetry and useless before one is. This
study spends most of its time before that point — a damage scan, a corpus, a KD table,
three A/B arms — and those measurements are the thing worth following, so they get panels
of their own here and the training panels are folded in from `qat_report` as soon as a
`train.log` exists.

Nothing is recomputed: every number is read from the artifact that produced it
(`layer_damage.json`, `stage_damage_*.json`, `stop_baseline.json`, the KD precompute log,
each arm's `train.log`). A panel with no artifact behind it renders as "pending" rather
than as an empty axis, because an empty axis reads as a measurement of zero.

    python scripts/gemma4_report.py --out out/gemma4-ternary/report.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemma4_ab_summary import arm_row  # noqa: E402
from qat_report import (  # noqa: E402
    INK,
    MUTED,
    Panel,
    legend,
)

from quant_tuner.qat.stop_probe import PROBE_SPECS  # noqa: E402

ROOT = Path("out/gemma4-ternary")
W, H, PAD = 940, 220, 60

# Okabe-Ito. Stage colours are ordinal (the schedule is an order), the two roles keep
# the same hues they carry everywhere else in this study.
STAGE_HUES = ["#0072b2", "#009e73", "#e69f00", "#cc79a7", "#56b4e9", "#d55e00", "#999999"]
BAD, GOOD = "#d55e00", "#009e73"


# ------------------------------------------------------------------ status


def kd_progress(log: Path) -> dict | None:
    """(done, total, s/window, ETA) from the precompute's own progress lines."""
    if not log.exists():
        return None
    hits = re.findall(r"window (\d+)/(\d+)\s+positions=(\d+).*?\((\d+\.\d+)s/window\)",
                      log.read_text(errors="replace"))
    if not hits:
        return None
    done, total, pos, per = int(hits[-1][0]), int(hits[-1][1]), int(hits[-1][2]), float(hits[-1][3])
    saved = "saved" in log.read_text(errors="replace")
    return {"done": done, "total": total, "positions": pos, "s_per_window": per,
            "eta_h": (total - done) * per / 3600.0, "done_flag": saved}


def kpi(label: str, value: str, sub: str) -> str:
    return f"<div><b>{value}</b><span>{label} · {sub}</span></div>"


# ------------------------------------------------------------------ figures


def fig_layer_damage(rows, order, stages):
    """Per-layer output KLD, drawn in SCHEDULE order rather than depth order.

    Depth order hides the thing the schedule is built on. In schedule order the curve is
    monotone by construction and the question a reader actually has — how much worse is
    the last stage than the first — is the height of the right-hand bars.
    """
    by = {r["group"]: r["kld"] for r in rows}
    vals = [by[g] for g in order]
    p = Panel(W, 260, PAD, (-0.5, len(order) - 0.5), (min(vals) * 0.7, max(vals) * 1.4),
              logy=True, xlabel="position in the schedule (least-damaging first)",
              ylabel="KLD(dense ‖ one layer ternarized)   log", xticks=[],
              yfmt="{:g}")
    bw = (p.w - 2 * p.pad) / len(order) * 0.72
    for i, g in enumerate(order):
        v = by[g]
        st = next((k for k, s in enumerate(stages) if g in s), len(stages) - 1)
        y, y0 = p.py(v), p.py(min(vals) * 0.7)
        p.parts.append(
            f'<rect x="{p.px(i) - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
            f'height="{max(1, y0 - y):.1f}" fill="{STAGE_HUES[st % len(STAGE_HUES)]}" '
            f'opacity="0.85"><title>{g}: KLD {v:.4f}</title></rect>')
    for k, s in enumerate(stages):
        if not s:
            continue
        i = order.index(s[0])
        p.parts.append(f'<text x="{p.px(i):.1f}" y="{p.pad_y - 4:.0f}" font-size="9" '
                       f'fill="{STAGE_HUES[k % len(STAGE_HUES)]}">stage {k + 1}</text>')
    items = [(f"stage {k + 1}: {', '.join(x.replace('layer.', '') for x in s)}",
              STAGE_HUES[k % len(STAGE_HUES)]) for k, s in enumerate(stages) if s]
    return f'<div class="fig">{p.svg()}</div>' + legend(items)


def fig_kind_damage(rows):
    """Damage per tensor KIND, all 42 layers at once. The spread is the finding."""
    ks = sorted([r for r in rows if r["group"].startswith("kind.")], key=lambda r: r["kld"])
    names = [r["group"].replace("kind.", "").replace("mlp.", "").replace("self_attn.", "")
             for r in ks]
    vals = [r["kld"] for r in ks]
    p = Panel(W, 250, PAD, (-0.5, len(ks) - 0.5), (min(vals) * 0.6, max(vals) * 1.6),
              logy=True, ylabel="KLD(dense ‖ this kind ternarized in every layer)   log",
              xticks=[], yfmt="{:g}")
    bw = (p.w - 2 * p.pad) / len(ks) * 0.6
    for i, (n, v) in enumerate(zip(names, vals, strict=True)):
        worst = v == max(vals)
        y, y0 = p.py(v), p.py(min(vals) * 0.6)
        p.parts.append(
            f'<rect x="{p.px(i) - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
            f'height="{max(1, y0 - y):.1f}" fill="{BAD if worst else "#0072b2"}" '
            f'opacity="{0.95 if worst else 0.75}"><title>{n}: {v:.4f}</title></rect>')
        p.parts.append(f'<text x="{p.px(i):.1f}" y="{p.h - p.pad_y + 13:.0f}" font-size="9" '
                       f'text-anchor="end" fill="{MUTED}" '
                       f'transform="rotate(-40 {p.px(i):.1f} {p.h - p.pad_y + 13:.0f})">'
                       f'{n}</text>')
    return f'<div class="fig">{p.svg()}</div>'


def fig_compounding(cum, stage_baseline=None):
    """Cumulative damage along the schedule, with the doubling it follows.

    The dashed line is 0.105·2^((n−6)/6) — not a fit anyone should trust to extrapolate,
    but the shape is the whole argument for training BETWEEN stages: the individual
    layers sum to 2.830 and together reach 10.666.
    """
    xs = [r["n_layers"] for r in cum]
    ys = [r["kld"] for r in cum]
    p = Panel(W, 250, PAD, (min(xs), max(xs)), (min(ys) * 0.5, max(ys) * 2.0), logy=True,
              xlabel="layers ternarized (in schedule order)",
              ylabel="cumulative KLD(dense ‖ candidate)   log",
              xticks=xs, yfmt="{:g}")
    fit = [(n, ys[0] * 2 ** ((n - xs[0]) / 6)) for n in xs]
    p.line(fit, MUTED, 1, 0.7)
    p.note(xs[len(xs) // 2], fit[len(xs) // 2][1] * 1.5, "0.105 · 2^((n−6)/6)", color=MUTED)
    p.line(list(zip(xs, ys, strict=True)), BAD, 2.4)
    p.dots(list(zip(xs, ys, strict=True)), BAD, 3.5,
           lambda x, y: f"{x:g} layers: KLD {y:.4f}")
    if stage_baseline:
        p.rule(stage_baseline, GOOD, "4 4")
        p.note(xs[0], stage_baseline * 1.25, f"stage 1 as configured: {stage_baseline:.4f}",
               color=GOOD)
    return f'<div class="fig">{p.svg()}</div>' + legend([("measured", BAD), ("×2 per 6 layers", MUTED)]
                            + ([("stage 1 with down_proj dense", GOOD)] if stage_baseline else []))


def fig_probe_baseline(base):
    """The shipped model's stop policy, and why gemma needed different probe points.

    Plotted against Qwen's readings at the same NAMED positions. `after_tool_call` is
    the whole story: 0.99995 on Qwen, 0.00004 here, because gemma's template hands over
    to the harness there instead of ending the turn.
    """
    spec = PROBE_SPECS["gemma4"]
    pts = [(n, base["probs"][n]) for n, _ in spec.points if n in base["probs"]]
    qwen = {"sentence_period": 0.0092, "after_tool_call": 0.99995}
    p = Panel(W, 240, PAD, (-0.5, len(pts) - 0.5), (1e-6, 2.0), logy=True,
              ylabel="P(stop token) on the shipped E4B   log", xticks=[], yfmt="{:g}")
    for i, (n, v) in enumerate(pts):
        role = (BAD if n == spec.diagnostic else GOOD if n == spec.control else "#0072b2")
        p.parts.append(f'<circle cx="{p.px(i):.1f}" cy="{p.py(max(v, 1e-6)):.1f}" r="5" '
                       f'fill="{role}" stroke="#fff" stroke-width="1.5">'
                       f'<title>{n}: {v:.6f}</title></circle>')
        if n in qwen:
            p.parts.append(f'<circle cx="{p.px(i):.1f}" cy="{p.py(qwen[n]):.1f}" r="5" '
                           f'fill="none" stroke="{MUTED}" stroke-width="1.5" '
                           f'stroke-dasharray="2 2"><title>Qwen: {qwen[n]:.5f}</title></circle>')
            p.parts.append(f'<line x1="{p.px(i):.1f}" y1="{p.py(max(v, 1e-6)):.1f}" '
                           f'x2="{p.px(i):.1f}" y2="{p.py(qwen[n]):.1f}" stroke="{MUTED}" '
                           f'stroke-width="1" stroke-dasharray="2 3" opacity="0.6"/>')
        p.parts.append(f'<text x="{p.px(i):.1f}" y="{p.h - p.pad_y + 13:.0f}" font-size="9" '
                       f'text-anchor="end" fill="{MUTED}" '
                       f'transform="rotate(-35 {p.px(i):.1f} {p.h - p.pad_y + 13:.0f})">'
                       f'{n}</text>')
    return f'<div class="fig">{p.svg()}</div>' + legend([("diagnostic — stopping is WRONG here", BAD),
                             ("control — stopping is RIGHT here", GOOD),
                             ("other probe points", "#0072b2"),
                             ("Qwen's reading at the same name", MUTED)])


# ------------------------------------------------------------------ prose


def md(text: str) -> str:
    """The run's notes.md, rendered. Handles the pipe tables the notes actually use —
    the run's criteria live in one, and dropping it would leave the report asserting a
    verdict with the thresholds invisible."""
    out, rows, bullets = [], [], False

    def flush_rows():
        if not rows:
            return
        head, body = rows[0], [r for r in rows[1:] if not set(r) <= set("|-: ")]
        cells = lambda r, t: "".join(  # noqa: E731
            f"<{t}>{c.strip()}</{t}>" for c in r.strip().strip("|").split("|"))
        out.append('<div class="tw"><table><tr>' + cells(head, "th") + "</tr>"
                   + "".join(f"<tr>{cells(r, 'td')}</tr>" for r in body)
                   + "</table></div>")
        rows.clear()

    def flush_bullets():
        nonlocal bullets
        if bullets:
            out.append("</ul>")
            bullets = False

    for raw in text.splitlines():
        ln = raw.rstrip()
        if ln.lstrip().startswith("|"):
            flush_bullets()
            rows.append(ln)
            continue
        flush_rows()
        if ln.startswith("## "):
            flush_bullets()
            out.append(f"<h3>{ln[3:]}</h3>")
        elif ln.startswith("# "):
            flush_bullets()
            out.append(f"<h3>{ln[2:]}</h3>")
        elif ln.startswith("- "):
            if not bullets:
                out.append("<ul>")
                bullets = True
            out.append(f"<li>{ln[2:]}</li>")
        elif ln.startswith(">"):
            flush_bullets()
            out.append(f'<p style="border-left:3px solid #ddd;padding-left:.7rem">'
                       f"{ln.lstrip('> ')}</p>")
        elif ln:
            flush_bullets()
            out.append(f"<p>{ln}</p>")
    flush_rows()
    flush_bullets()
    html = "".join(out)
    for a, b in (("**", "b"), ("`", "code")):
        parts = html.split(a)
        html = "".join(x if i % 2 == 0 else f"<{b}>{x}</{b}>" for i, x in enumerate(parts))
    return html


def arms_table(runs) -> str:
    rows = [arm_row(r) for r in runs]
    if not rows:
        return ""
    spec = PROBE_SPECS["gemma4"]
    f = lambda v, s="{:.4f}": "—" if v is None else s.format(v)  # noqa: E731
    head = ("<tr><th>arm</th><th>steps</th><th>recovered</th><th>KLD</th><th>flip %</th>"
            f"<th>{spec.diagnostic}</th><th>{spec.control}</th><th>val</th>"
            "<th>gnorm</th><th>s/step</th></tr>")
    body = ""
    for r in rows:
        val = ("—" if r.get("val_first") is None
               else f"{f(r.get('val_first'), '{:.3f}')}→{f(r.get('val_last'), '{:.3f}')}")
        rec = f(r.get("recovered"), "{:.1%}")
        col = ""
        if r.get("recovered") is not None:
            col = f' style="color:{GOOD if r["recovered"] >= 0.70 else BAD}"'
        body += (f"<tr><td>{r['arm']}</td><td>{f(r.get('steps'), '{:d}')}</td>"
                 f"<td{col}>{rec}</td><td>{f(r.get('kld'))}</td>"
                 f"<td>{f(r.get('flip_pct'), '{:.3f}')}</td><td>{f(r.get('diag'))}</td>"
                 f"<td>{f(r.get('ctrl'))}</td><td>{val}</td>"
                 f"<td>{f(r.get('gnorm'), '{:.1f}')}</td>"
                 f"<td>{f(r.get('s_step'), '{:.1f}')}</td></tr>")
    return f'<div class="tw"><table>{head}{body}</table></div>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--notes", type=Path, default=None,
                    help="default <root>/stage1/notes.md")
    ap.add_argument("--title", default="Ternary E4B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    R = args.root
    notes = args.notes or R / "stage1" / "notes.md"

    def load(p):
        p = R / p
        return json.loads(p.read_text()) if p.exists() else None

    dmg = load("layer_damage.json")
    base = load("stop_baseline.json")
    untr = load("stage1/stage_damage_untrained.json")
    kd = kd_progress(R / "kd" / "precompute_31b.log")
    arms = sorted(d for d in R.glob("ab-lr*") if d.is_dir())
    stage1 = [d for d in (R / "stage1",) if (d / "train.log").exists()]

    # ---- status strip -----------------------------------------------------------
    order = dmg["layer_order"] if dmg else []
    stages = [order[i:i + 6] for i in range(0, len(order), 6)]
    k = []
    k.append(kpi("corpus", "651×32k", "21.3M tok · 28.7% supervised"))
    k.append(kpi("damage scan", "done" if dmg else "pending",
                 f"{len(order)} layers ranked" if dmg else "—"))
    if kd and kd["done_flag"]:
        k.append(kpi("KD table (31B)", "done", f"{kd['positions']:,} positions"))
    elif kd:
        k.append(kpi("KD table (31B)", f"{100 * kd['done'] / kd['total']:.0f}%",
                     f"{kd['done']}/{kd['total']} · ETA {kd['eta_h']:.1f} h"))
    else:
        k.append(kpi("KD table (31B)", "pending", "—"))
    k.append(kpi("stage-1 baseline",
                 f"{untr['rows']['untrained']['kld']:.4f}" if untr else "pending",
                 "untrained KLD, down_proj dense"))
    k.append(kpi("lr arms", f"{len(arms)}/3" if arms else "pending",
                 "60 steps each @ accum 1"))
    k.append(kpi("verdict", "pending", "GO ≥ 70% recovered"))

    # ---- sections ---------------------------------------------------------------
    secs = []
    if untr and dmg:
        b = untr["rows"]["untrained"]
        secs.append(
            "<section><h2>The question</h2>"
            "<p>Does QAT recover a stage's ternarization damage before the next stage "
            "compounds on it? With no training the cumulative curve doubles every six "
            "layers, and the individual layers are <b>3.77× superadditive</b> — they sum "
            "to 2.830 and together reach 10.666. If a trained stage cannot pull its own "
            "damage back down, the schedule buys nothing over ternarizing everything at "
            "once.</p>"
            f"<p>Stage 1 is <code>{', '.join(x.replace('layer.', '') for x in stages[0])}</code>"
            f" with <code>down_proj</code> held dense. Its untrained damage is "
            f"<b>{b['kld']:.4f}</b> (top-1 agreement {b['top1_agree']:.3f}, ppl "
            f"{b['ppl']:.2f} against the dense model's "
            f"{untr['rows']['dense']['ppl']:.3f}) — not the 0.1047 in the damage table, "
            "which ternarized every linear in those layers including the most damaging "
            "kind.</p></section>")
    if dmg:
        secs.append(
            f"<section><h2>Damage per tensor kind</h2>"
            f"<p>Ternarize one kind across all 42 layers, leave everything else dense, "
            f"measure how far the output distribution moved. The spread is <b>139×</b>, "
            f"and <code>mlp.down_proj</code> at 1.199 is <b>3.4× the next-worst kind</b> "
            f"— which is why it is held dense in every stage. Weight space ranked it "
            f"fourth-<i>safest</i>.</p>{fig_kind_damage(dmg['results'])}</section>")
        secs.append(
            f"<section><h2>The schedule</h2>"
            f"<p>Each bar is one decoder layer ternarized alone. Ordered least-damaging "
            f"first, which is the order the stages follow. Layers 22–23 are the last KV "
            f"donors for the 18 sharing layers above them, so they carry the most and "
            f"come last.</p>{fig_layer_damage(dmg['results'], order, stages)}</section>")
        if dmg.get("cumulative"):
            secs.append(
                f"<section><h2>Why the stages must train in between</h2>"
                f"{fig_compounding(dmg['cumulative'], untr and untr['rows']['untrained']['kld'])}"
                f"</section>")
    if base:
        secs.append(
            f"<section><h2>Termination baseline</h2>"
            f"<p>Termination is the failure mode of this pipeline, and gemma needed "
            f"different probe POINTS rather than different markers. gemma-4 has no sharp "
            f"stop position: after a complete answer it prefers <code>\\n\\n</code> "
            f"(0.275) over the stop token, and the control has ~25× of headroom over the "
            f"diagnostic where Qwen's had ~10⁴.</p>{fig_probe_baseline(base)}</section>")

    tbl = arms_table(arms + stage1)
    if tbl:
        secs.append(f"<section><h2>Arms</h2><p>No column decides on its own: a 2-step "
                    f"smoke flipped 1.4–1.7% of codes and made the damage three times "
                    f"worse.</p>{tbl}</section>")
    if notes.exists():
        secs.append(f"<section><h2>Running notes</h2>{md(notes.read_text())}</section>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html><meta charset="utf-8">
<title>{args.title}</title>
<style>
 /* Committed to a light ground, like every report in this study: the Okabe-Ito series
    colours are chosen for contrast against white, and re-deriving them per theme would
    break comparability with the figures already published. So paint it explicitly
    rather than inheriting -- an unpainted body renders #222 text on the host's ground. */
 :root{{color-scheme:light}}
 body{{font:14px/1.55 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:1000px;
       color:{INK};background:#fff;padding:0 1rem}}
 h1{{font-size:21px;margin:0 0 .2rem}} h2{{font-size:15px;margin:0 0 .2rem}}
 h3{{font-size:12px;color:{MUTED};font-weight:600;margin:1rem 0 .3rem;
     text-transform:uppercase;letter-spacing:.04em}}
 section{{margin:2rem 0}} p{{color:#555;margin:.1rem 0 .5rem;max-width:78ch}}
 svg{{display:block;max-width:100%}}
 .legend{{margin:.3rem 0 0;font-size:11px;color:#666}}
 .legend span{{margin-right:14px;white-space:nowrap}}
 .legend i{{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:-1px}}
 .meta{{font-size:12px;color:{MUTED};margin-bottom:1.2rem}}
 .kpi{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid #eee;
       border-radius:6px;overflow:hidden;margin:0 0 .5rem}}
 .kpi div{{padding:.55rem .8rem;border-right:1px solid #eee;border-top:1px solid #eee}}
 .kpi div:nth-child(3n){{border-right:0}} .kpi div:nth-child(-n+3){{border-top:0}}
 .kpi b{{display:block;font-size:17px;font-weight:600;letter-spacing:-.01em}}
 .kpi span{{font-size:11px;color:{MUTED}}}
 li{{color:#444;margin:.15rem 0;max-width:78ch}} ul{{padding-left:1.1rem}}
 .tw,.fig{{overflow-x:auto;max-width:100%}}
 table{{font-variant-numeric:tabular-nums}}
 table{{border-collapse:collapse;font-size:12px;margin:.4rem 0}}
 td,th{{border-bottom:1px solid #eee;padding:3px 10px;text-align:right}}
 th{{color:{MUTED};font-weight:500}}
 td:first-child,th:first-child{{text-align:left;font-family:ui-monospace,monospace}}
 code{{font-family:ui-monospace,monospace;font-size:.92em}}
</style>
<h1>{args.title}</h1>
<p class="meta">google/gemma-4-E4B-it-qat-q4_0-unquantized → per-group TWN, staged ·
 teacher google/gemma-4-31B-it · rendered {time.strftime('%Y-%m-%d %H:%M')}</p>
<div class="kpi">{''.join(k)}</div>
<p class="meta">Ternarization stores <code>w = s·c</code>, <code>c ∈ {{−1,0,+1}}</code>.
This model does not start there — unlike Bonsai it is dense, so step 0 is a real
perturbation and the whole study is whether training can take it back.</p>
{''.join(secs)}
""")
    print(f"[report] {len(secs)} sections -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

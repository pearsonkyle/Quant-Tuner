#!/usr/bin/env python
"""Ternary code distribution (-1 / 0 / +1) per tensor, and how it moves during QAT.

Two things a ternary QAT run needs to show and a loss curve cannot:

  * **census** — the actual proportion of -1, 0 and +1 codes in each tensor. This is the
    model's capacity budget: a weight sitting at 0 contributes nothing, so the zero
    fraction IS the sparsity, and training moves it.
  * **trajectory** — how that zero fraction moves as steps accumulate. Reconstructed from
    the flip telemetry, which counts recruitment (`0->±`) and pruning (`±->0`) separately,
    so the net density at any checkpoint is `density_0 + (z2nz - nz2z) / numel`.

Usage:

    # baseline census of the shipped weights (add --all for every trainable linear;
    # that reads the whole model, so don't do it while a run is training)
    python scripts/ternary_distribution.py census --model out/exp-057/model \\
        --tensors out/exp-058/telemetry/flips.csv --out out/exp-058/telemetry/census.csv

    # figure: per-layer composition + zero-fraction trajectory
    python scripts/ternary_distribution.py plot --census out/exp-058/telemetry/census.csv \\
        --flips out/exp-058/telemetry/flips.csv --out out/exp-058/telemetry/ternary.html

Colors are ColorBrewer RdBu — a diverging scheme, because -1 <- 0 -> +1 is polarity with a
neutral midpoint, and RdBu is CVD-safe by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NEG, ZERO, POS = "#2166ac", "#cccccc", "#b2182b"  # RdBu poles + neutral midpoint


def census(model_dir: Path, names: list[str] | None, want_all: bool) -> list[dict]:
    """Count -1/0/+1 per tensor by re-deriving codes from the fp weights.

    Reads ONE tensor at a time (safetensors is lazy), so peak memory is a single tensor
    rather than the model. Still touches every shard when `want_all` is set.
    """
    import torch
    from safetensors import safe_open

    from quant_tuner.qat.ternary import ternarize_group

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    if want_all:
        # the trainable linears: attention + MLP projections inside a decoder layer
        wanted = [k for k in weight_map
                  if re.search(r"layers\.\d+\.(self_attn|mlp)\.\w+_proj\.weight$", k)]
    else:
        wanted = [f"{n}.weight" for n in (names or [])]
        missing = [w for w in wanted if w not in weight_map]
        if missing:
            raise SystemExit(f"[census] not in the checkpoint: {missing[:3]}")

    by_shard: dict[str, list[str]] = {}
    for key in wanted:
        by_shard.setdefault(weight_map[key], []).append(key)

    rows: list[dict] = []
    for shard, keys in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt") as f:
            for key in sorted(keys):
                w = f.get_tensor(key).float()
                codes, _, _ = ternarize_group(w)
                n = codes.numel()
                neg = int((codes < 0).sum())
                pos = int((codes > 0).sum())
                name = key[: -len(".weight")]
                m = re.search(r"layers\.(\d+)\.", name)
                rows.append({
                    "tensor": name,
                    "layer": int(m.group(1)) if m else -1,
                    "kind": name.rsplit(".", 1)[-1],
                    "numel": n,
                    "neg": neg, "zero": n - neg - pos, "pos": pos,
                    "neg_frac": neg / n, "zero_frac": (n - neg - pos) / n, "pos_frac": pos / n,
                })
                del w, codes
                print(f"[census] {name}: -1 {neg / n:.1%}  0 {(n - neg - pos) / n:.1%}  "
                      f"+1 {pos / n:.1%}", flush=True)
    del torch
    return sorted(rows, key=lambda r: (r["layer"], r["kind"]))


def trajectory(flips: list[dict], base: dict[str, dict]) -> list[dict]:
    """Zero-fraction per tracked tensor per checkpoint.

    The flip telemetry gives cumulative `0->±` and `±->0` counts against the start-of-run
    snapshot, so density at step t is exactly `density_0 + (z2nz - nz2z)/numel`. `numel`
    comes from the baseline census.
    """
    out: list[dict] = []
    for r in flips:
        b = base.get(r["tensor"])
        if not b:
            continue
        n = b["numel"]
        z2nz, nz2z = int(r["zero_to_nonzero"]), int(r["nonzero_to_zero"])
        zero_frac = b["zero_frac"] - (z2nz - nz2z) / n
        # total code changes = recruit + prune + sign; sign is the remainder
        total_flips = float(r["flip_pct"]) / 100.0 * n
        out.append({
            "step": int(r["step"]), "tensor": r["tensor"], "layer": b["layer"],
            "kind": b["kind"], "numel": n,
            "zero_frac": zero_frac,
            "zero_frac_start": b["zero_frac"],
            "recruited": z2nz, "pruned": nz2z,
            "sign_flipped": max(0, round(total_flips - z2nz - nz2z)),
        })
    return sorted(out, key=lambda r: (r["tensor"], r["step"]))


def _bars(rows: list[dict], w: int, h: int, pad: int) -> str:
    """Per-tensor stacked composition bars: -1 | 0 | +1, summing to 1."""
    if not rows:
        return ""
    n = len(rows)
    bw = (w - 2 * pad) / n
    out = []
    for i, r in enumerate(rows):
        x = pad + i * bw
        y = pad
        for frac, color in ((r["neg_frac"], NEG), (r["zero_frac"], ZERO), (r["pos_frac"], POS)):
            seg = frac * (h - 2 * pad)
            # 2px surface gap between fills keeps adjacent segments readable
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" '
                       f'height="{max(0, seg - 2):.1f}" fill="{color}"/>')
            y += seg
        if i % max(1, n // 12) == 0:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{h - pad + 14}" font-size="10" '
                       f'text-anchor="middle" fill="#666">{r["layer"]}</text>')
    return "".join(out)


def _lines(traj: list[dict], w: int, h: int, pad: int) -> str:
    """Zero-fraction over steps, one line per tracked tensor."""
    if not traj:
        return ""
    by: dict[str, list[dict]] = {}
    for r in traj:
        by.setdefault(r["tensor"], []).append(r)
    steps = [r["step"] for r in traj]
    x0, x1 = min(steps), max(steps)
    # scale each line to its OWN start so the shared axis is "change in zero fraction";
    # absolute zero fractions differ per tensor and would compress every line to a stripe
    deltas = [(r["zero_frac"] - r["zero_frac_start"]) * 100 for r in traj]
    y0, y1 = min(deltas + [0.0]), max(deltas + [0.0])
    span = (y1 - y0) or 1.0

    def px(s):
        return pad + (s - x0) / max(1, x1 - x0) * (w - 2 * pad)

    def py(d):
        return h - pad - (d - y0) / span * (h - 2 * pad)

    out = [f'<line x1="{pad}" y1="{py(0):.1f}" x2="{w - pad}" y2="{py(0):.1f}" '
           f'stroke="#ddd" stroke-width="1"/>']
    for name, rs in by.items():
        rs = sorted(rs, key=lambda r: r["step"])
        pts = " ".join(f"{px(r['step']):.1f},{py((r['zero_frac'] - r['zero_frac_start']) * 100):.1f}"
                       for r in rs)
        # blue where the tensor is densifying (zeros going away), red where it is pruning
        last = rs[-1]["zero_frac"] - rs[-1]["zero_frac_start"]
        color = NEG if last < 0 else POS
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="2" opacity="0.75"><title>{name}</title></polyline>')
        lr = rs[-1]
        out.append(f'<text x="{px(lr["step"]) + 6:.1f}" '
                   f'y="{py((lr["zero_frac"] - lr["zero_frac_start"]) * 100) + 3:.1f}" '
                   f'font-size="9" fill="#666">{lr["layer"]}.{lr["kind"]}</text>')
    out.append(f'<text x="{pad}" y="{h - 6}" font-size="10" fill="#666">step {x0} → {x1}</text>')
    out.append(f'<text x="{pad}" y="{pad - 8}" font-size="10" fill="#666">'
               f'Δ zero-fraction (pp), {y0:+.2f} … {y1:+.2f}</text>')
    return "".join(out)


def render_html(cen: list[dict], traj: list[dict]) -> str:
    W, H, PAD = 900, 320, 40
    kinds = sorted({r["kind"] for r in cen})
    panels = []
    for kind in kinds:
        rows = [r for r in cen if r["kind"] == kind]
        panels.append(
            f'<h3>{kind} — composition by layer</h3>'
            f'<svg width="{W}" height="{H}" role="img">{_bars(rows, W, H, PAD)}</svg>'
        )
    legend = (f'<p class="legend">'
              f'<span style="background:{NEG}"></span> −1 '
              f'<span style="background:{ZERO}"></span> 0 '
              f'<span style="background:{POS}"></span> +1</p>')
    rows_html = "".join(
        f"<tr><td>{r['tensor']}</td><td>{r['neg_frac']:.3f}</td>"
        f"<td>{r['zero_frac']:.3f}</td><td>{r['pos_frac']:.3f}</td></tr>" for r in cen)
    return f"""<!doctype html><meta charset="utf-8">
<title>Ternary code distribution</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:2rem;max-width:960px;color:#222}}
 h1{{font-size:20px}} h3{{font-size:14px;margin:1.5rem 0 .3rem;color:#444}}
 .legend span{{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin:0 2px 0 10px}}
 table{{border-collapse:collapse;font-size:12px;margin-top:1rem}}
 td,th{{border-bottom:1px solid #eee;padding:2px 10px;text-align:right}}
 td:first-child{{text-align:left;font-family:ui-monospace,monospace}}
</style>
<h1>Ternary code distribution</h1>
<p>Each bar is one tensor's weights split into −1 / 0 / +1. The zero band is the model's
unused capacity — training moves it, and that movement is what a loss curve hides.</p>
{legend}
{''.join(panels)}
<h3>Change in zero-fraction over training</h3>
<p>Per tracked tensor, relative to its own start. Below the line = zeros being recruited
into live weights (densifying); above = weights being pruned to zero.</p>
<svg width="{W}" height="{H}" role="img">{_lines(traj, W, H, PAD)}</svg>
<h3>Table</h3>
<table><tr><th>tensor</th><th>−1</th><th>0</th><th>+1</th></tr>{rows_html}</table>
"""


def _read_csv(p: Path) -> list[dict]:
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _write_csv(p: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="count -1/0/+1 per tensor from a model's weights")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--tensors", type=Path,
                   help="flips.csv — restricts the census to the tracked tensors (cheap)")
    c.add_argument("--all", action="store_true",
                   help="every trainable linear; reads the WHOLE model — not while training")
    c.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("plot", help="render the composition + trajectory figure")
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--flips", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--traj-out", type=Path)

    args = ap.parse_args()
    if args.cmd == "census":
        names = None
        if args.tensors:
            names = sorted({r["tensor"] for r in _read_csv(args.tensors)})
        rows = census(args.model, names, args.all)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.out, rows)
        print(f"[census] {len(rows)} tensors -> {args.out}")
        return 0

    cen = [{**r, **{k: float(r[k]) for k in ("neg_frac", "zero_frac", "pos_frac")},
            "layer": int(r["layer"]), "numel": int(r["numel"])} for r in _read_csv(args.census)]
    traj: list[dict] = []
    if args.flips:
        base = {r["tensor"]: r for r in cen}
        traj = trajectory(_read_csv(args.flips), base)
        if args.traj_out:
            _write_csv(args.traj_out, traj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(cen, traj))
    print(f"[plot] {len(cen)} tensors, {len(traj)} trajectory points -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

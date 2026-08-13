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
import math
import re
from pathlib import Path

NEG, ZERO, POS = "#2166ac", "#cccccc", "#b2182b"  # RdBu poles + neutral midpoint


def _row_for(name: str, codes) -> dict:
    n = codes.numel()
    neg = int((codes < 0).sum())
    pos = int((codes > 0).sum())
    m = re.search(r"layers\.(\d+)\.", name)
    return {"tensor": name, "layer": int(m.group(1)) if m else -1,
            "kind": name.rsplit(".", 1)[-1], "numel": n,
            "neg": neg, "zero": n - neg - pos, "pos": pos,
            "neg_frac": neg / n, "zero_frac": (n - neg - pos) / n, "pos_frac": pos / n}


def census_latents(latents_path: Path, names: list[str]) -> tuple[list[dict], int]:
    """Exact -1/0/+1 for the CURRENT codes, read from a training checkpoint.

    The checkpoint holds only the trainable latents, and ``mmap=True`` keeps the read
    lazy — pulling 12 named tensors costs ~1 GB, not the file's 28. That is what makes
    this safe to run beside a live training job, unlike the full-model census.

    A concurrent checkpoint save is harmless: the trainer writes to a temp file and
    ``os.replace``s it, so an already-open mapping keeps referring to the old inode.
    """
    import torch

    from quant_tuner.qat.ternary import ternarize_group

    ck = torch.load(latents_path, map_location="cpu", weights_only=False, mmap=True)
    latents = ck["latents"]
    step = int(ck.get("step", -1))
    rows = []
    for name in names:
        # the trainer wraps each linear as TernaryLinear, so latents are keyed
        # "<module>.linear.weight" while the flip telemetry names the module
        cand = [name, f"{name}.weight", f"{name}.linear.weight"]
        key = next((k for k in cand if k in latents), None)
        if key is None:
            raise SystemExit(f"[census] {name} not in {latents_path} (tried {cand})")
        codes, _, _ = ternarize_group(latents[key].float())
        rows.append(_row_for(key.removesuffix(".weight").removesuffix(".linear"), codes))
        del codes
    ck.clear()
    del latents, ck
    return sorted(rows, key=lambda r: (r["layer"], r["kind"])), step


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
                row = _row_for(key.removesuffix(".weight"), codes)
                rows.append(row)
                del w, codes
                print(f"[census] {row['tensor']}: -1 {row['neg_frac']:.1%}  "
                      f"0 {row['zero_frac']:.1%}  +1 {row['pos_frac']:.1%}", flush=True)
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


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick values covering [lo, hi] at a 1/2/5 x 10^n step."""
    span = (hi - lo) or 1.0
    raw = span / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _lines(traj: list[dict], w: int, h: int, pad: int, step_grid: int = 100) -> str:
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

    out = []
    # Recessive grid: vertical every `step_grid` steps, horizontal on round pp values.
    # The zero line is the reference the whole chart is read against, so it is darker.
    for s in range(0, x1 + 1, step_grid):  # +1, not +step_grid: never draw past the last step
        if s < x0:
            continue
        out.append(f'<line x1="{px(s):.1f}" y1="{pad}" x2="{px(s):.1f}" y2="{h - pad}" '
                   f'stroke="#eee" stroke-width="1"/>')
        out.append(f'<text x="{px(s):.1f}" y="{h - pad + 15}" font-size="10" '
                   f'text-anchor="middle" fill="#888">{s}</text>')
    for t in _nice_ticks(y0, y1):
        if not (y0 - 1e-9 <= t <= y1 + 1e-9):
            continue
        out.append(f'<line x1="{pad}" y1="{py(t):.1f}" x2="{w - pad}" y2="{py(t):.1f}" '
                   f'stroke="{"#bbb" if t == 0 else "#eee"}" stroke-width="1"/>')
        out.append(f'<text x="{pad - 6}" y="{py(t) + 3:.1f}" font-size="10" '
                   f'text-anchor="end" fill="#888">{t:+.2f}</text>')
    ends: list[tuple[float, str, str]] = []
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
        ends.append((py((lr["zero_frac"] - lr["zero_frac_start"]) * 100),
                     f'{lr["layer"]}.{lr["kind"]}', color))

    # Direct labels beat a legend here (12 series), but several lines converge near zero
    # and their labels would overlap. Nudge each down to clear the previous one; a 1px
    # leader keeps the label tied to its line once nudged.
    lx = px(x1) + 6
    prev_y = -1e9
    for y, label, color in sorted(ends):
        ly = max(y + 3, prev_y + 10)
        prev_y = ly
        if abs(ly - (y + 3)) > 1:
            out.append(f'<line x1="{lx - 3:.1f}" y1="{y:.1f}" x2="{lx:.1f}" y2="{ly - 3:.1f}" '
                       f'stroke="{color}" stroke-width="1" opacity="0.35"/>')
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" fill="#666">{label}</text>')
    out.append(f'<text x="{w / 2:.0f}" y="{h - 4}" font-size="11" text-anchor="middle" '
               f'fill="#666">training step</text>')
    out.append(f'<text x="{pad - 6}" y="{pad - 12}" font-size="11" fill="#666">'
               f'Δ zero-fraction (pp)</text>')
    return "".join(out)


def _dist_table(rows: list[dict], ref: dict[str, dict] | None = None) -> str:
    """Composition table. With `ref`, adds the zero-fraction delta against it."""
    head = ("<tr><th>tensor</th><th>−1</th><th>0</th><th>+1</th>"
            + ("<th>Δ0 (pp)</th>" if ref else "") + "</tr>")
    body = []
    for r in rows:
        cells = (f"<td>{r['neg_frac'] * 100:.2f}%</td><td><b>{r['zero_frac'] * 100:.2f}%</b></td>"
                 f"<td>{r['pos_frac'] * 100:.2f}%</td>")
        if ref:
            b = ref.get(r["tensor"])
            d = (r["zero_frac"] - b["zero_frac"]) * 100 if b else 0.0
            # blue = zeros recruited into live weights, red = weights pruned to zero
            color = NEG if d < 0 else POS
            cells += f'<td style="color:{color}">{d:+.3f}</td>'
        body.append(f"<tr><td>{r['tensor'].replace('model.layers.', '')}</td>{cells}</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def render_html(cen: list[dict], traj: list[dict], latest: list[dict] | None = None,
                latest_step: int | None = None) -> str:
    W, H, PAD = 900, 380, 52
    steps = [r["step"] for r in traj]
    span = f"steps {min(steps)}–{max(steps)}" if steps else "no trajectory"
    later = ""
    if latest:
        ref = {r["tensor"]: r for r in cen}
        later = (f"<h2>Distribution at step {latest_step}</h2>"
                 f"<p>Same tensors after training. Δ0 is the change in zero-fraction: "
                 f"negative = the model switched dead weights on.</p>"
                 f"{_dist_table(latest, ref)}")
    return f"""<!doctype html><meta charset="utf-8">
<title>Ternary code distribution</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:2rem;max-width:960px;color:#222}}
 h1{{font-size:20px}} h2{{font-size:15px;margin:2rem 0 .3rem}}
 p{{color:#555;margin:.3rem 0 .8rem}}
 table{{border-collapse:collapse;font-size:12px;margin:.5rem 0 1rem}}
 td,th{{border-bottom:1px solid #eee;padding:3px 12px;text-align:right}}
 th{{color:#888;font-weight:500}}
 td:first-child,th:first-child{{text-align:left;font-family:ui-monospace,monospace}}
</style>
<h1>Ternary code distribution</h1>
<p>A natively-ternary weight is −1, 0 or +1. The zero fraction is the model's unused
capacity, and continued QAT moves it — which a loss curve cannot show.</p>

<h2>Change in zero-fraction over training ({span})</h2>
<p>Per tracked tensor, relative to its own starting value. <b>Below the zero line</b> =
zeros recruited into live weights (densifying); <b>above</b> = live weights pruned to zero.
Hover a line for its tensor name.</p>
<svg width="{W}" height="{H}" role="img">{_lines(traj, W, H, PAD)}</svg>

<h2>Distribution at step 0 (as shipped)</h2>
{_dist_table(cen)}
{later}
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
    c.add_argument("--model", type=Path, help="HF model dir (the step-0 / shipped codes)")
    c.add_argument("--latents", type=Path,
                   help="a training checkpoint (trained_latents.pt) — the CURRENT codes. "
                        "Reads lazily, so restricting with --tensors is safe beside a "
                        "live run; --model --all is not.")
    c.add_argument("--tensors", type=Path,
                   help="flips.csv — restricts the census to the tracked tensors (cheap)")
    c.add_argument("--all", action="store_true",
                   help="every trainable linear; reads the WHOLE model — not while training")
    c.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("plot", help="render the composition + trajectory figure")
    p.add_argument("--census", type=Path, required=True, help="step-0 census CSV")
    p.add_argument("--flips", type=Path)
    p.add_argument("--latest", type=Path,
                   help="second census CSV (e.g. from `census --latents`) for the "
                        "after-training table")
    p.add_argument("--latest-step", type=int)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--traj-out", type=Path)

    args = ap.parse_args()
    if args.cmd == "census":
        names = None
        if args.tensors:
            names = sorted({r["tensor"] for r in _read_csv(args.tensors)})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.latents:
            if not names:
                raise SystemExit("[census] --latents needs --tensors (reading every latent "
                                 "defeats the point of the lazy read)")
            rows, step = census_latents(args.latents, names)
            for r in rows:
                print(f"[census] {r['tensor']}: -1 {r['neg_frac']:.1%}  "
                      f"0 {r['zero_frac']:.1%}  +1 {r['pos_frac']:.1%}", flush=True)
            print(f"[census] checkpoint step {step}")
        elif args.model:
            rows = census(args.model, names, args.all)
        else:
            raise SystemExit("[census] pass --model or --latents")
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
    latest = None
    if args.latest:
        latest = [{**r, **{k: float(r[k]) for k in ("neg_frac", "zero_frac", "pos_frac")},
                   "layer": int(r["layer"]), "numel": int(r["numel"])}
                  for r in _read_csv(args.latest)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(cen, traj, latest, args.latest_step))
    print(f"[plot] {len(cen)} tensors, {len(traj)} trajectory points -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

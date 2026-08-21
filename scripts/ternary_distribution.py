#!/usr/bin/env python
"""Ternary code distribution (-1 / 0 / +1) per tensor, and how it moves during QAT.

Two things a ternary QAT run needs to show and a loss curve cannot:

  * **census** — the actual proportion of -1, 0 and +1 codes in each tensor. This is the
    model's capacity budget: a weight sitting at 0 contributes nothing, so the zero
    fraction IS the sparsity, and training moves it.
  * **trajectory** — `trajectory()` turns the flip telemetry (which counts recruitment
    `0->±` and pruning `±->0` separately) into net density per checkpoint:
    `density_0 + (z2nz - nz2z) / numel`. Consumed by `qat_report.py`.

This module is DATA ONLY; the figures live in `scripts/qat_report.py`.

    # step-0 census from the shipped weights (--all reads the whole model: not while training)
    python scripts/ternary_distribution.py census --model out/exp-057/model \\
        --tensors out/exp-058/telemetry/flips.csv --out out/exp-058/telemetry/census.csv

    # current codes from a live checkpoint (lazy: ~1 GB of a 28 GB file)
    python scripts/ternary_distribution.py census --latents .../trained_latents.pt \\
        --tensors out/exp-058/telemetry/flips.csv --out .../census_latest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


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

    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        weight_map: dict[str, str] = json.loads(idx.read_text())["weight_map"]
    else:
        # A checkpoint small enough to ship as one file has no index -- e.g.
        # gemma-4-E4B-it-qat-q4_0-unquantized, 15.9 GB in a single model.safetensors.
        # Reading its key list gives the same map with one shard in it.
        single = model_dir / "model.safetensors"
        if not single.exists():
            raise SystemExit(f"[census] neither {idx.name} nor {single.name} in {model_dir}")
        with safe_open(single, framework="pt") as f:
            weight_map = dict.fromkeys(f.keys(), single.name)  # noqa: SIM118
    if want_all:
        # The trainable linears: attention + MLP projections inside a DECODER layer.
        # The tower exclusion is not belt-and-braces. gemma-4's audio tower ships
        # `model.audio_tower.layers.0.self_attn.relative_k_proj.weight`, which matches the
        # projection pattern exactly and would enter the census as a decoder tensor -- a
        # tower the training never touches, reported beside the ones it did.
        wanted = [k for k in weight_map
                  if re.search(r"layers\.\d+\.(self_attn|mlp)\.\w+_proj\.weight$", k)
                  and "_tower" not in k]
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

    raise SystemExit("[plot] moved to scripts/qat_report.py")


if __name__ == "__main__":
    raise SystemExit(main())

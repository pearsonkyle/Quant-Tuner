#!/usr/bin/env python
"""Turn a QAT training stdout log into machine-readable time series.

The trainer prints its telemetry (loss, lr, code flips, validation) as text. For a run
already in flight that text is the ONLY record, so this parser is how a finished — or
still-running — log becomes a table you can plot and write up.

New runs also write `metrics.jsonl` directly (see `qat.train`); this parser stays the
path for logs predating that, and for cross-checking the two agree.

    python scripts/parse_qat_log.py /tmp/sft8k_full5.log --out out/exp-058/telemetry

writes `steps.csv` (per logged step), `flips.csv` (one row per tensor per checkpoint),
`val.csv` and `summary.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# "[qat] step 155/522 loss=1.2212 lr=4.30e-04 mem=30.8GiB 372.0s/step"
STEP_RE = re.compile(
    r"^\[qat\] step (\d+)/(\d+) loss=([\d.]+) lr=([\d.e+-]+) mem=([\d.]+)GiB ([\d.]+)s/step"
)
# "[qat] step 120 VAL masked-CE 1.1017"
VAL_RE = re.compile(r"^\[qat\] step (\d+) VAL masked-CE ([\d.]+)")
# "  model.layers.0.self_attn.q_proj: flips 1.8593% (0->±:131663 ±->0:179612) scale-drift 2.29%"
FLIP_RE = re.compile(
    r"^\s+(\S+): flips ([\d.]+)% \(0->\S+:(\d+) \S+->0:(\d+)\) scale-drift ([\d.]+)%"
)
CKPT_RE = re.compile(r"^\[qat\] checkpoint @ step (\d+)")


def parse(text: str) -> dict[str, list[dict]]:
    steps: list[dict] = []
    vals: list[dict] = []
    flips: list[dict] = []
    pending: list[dict] = []  # flip rows seen since the last checkpoint line
    last_step = 0

    for line in text.splitlines():
        if m := STEP_RE.match(line):
            last_step = int(m.group(1))
            steps.append({
                "step": last_step,
                "total_steps": int(m.group(2)),
                "loss": float(m.group(3)),
                "lr": float(m.group(4)),
                "mem_gib": float(m.group(5)),
                "s_per_step": float(m.group(6)),
            })
        elif m := VAL_RE.match(line):
            vals.append({"step": int(m.group(1)), "val_masked_ce": float(m.group(2))})
        elif m := FLIP_RE.match(line):
            z2nz, nz2z = int(m.group(3)), int(m.group(4))
            pending.append({
                "tensor": m.group(1),
                "flip_pct": float(m.group(2)),
                "zero_to_nonzero": z2nz,
                "nonzero_to_zero": nz2z,
                # >1 = recruiting dead weights, <1 = pruning, ~1 = sign reorganization.
                # The split by tensor type is the whole point of tracking both counts.
                "densify_ratio": (z2nz / nz2z) if nz2z else None,
                "net_density_delta": z2nz - nz2z,
                "scale_drift_pct": float(m.group(5)),
            })
        elif m := CKPT_RE.match(line):
            # the flip block is printed immediately BEFORE its checkpoint line, so the
            # rows queued since the last checkpoint belong to this step
            at = int(m.group(1))
            for row in pending:
                flips.append({"step": at, **row})
            pending = []

    # a run still in flight may have printed a flip block whose checkpoint line hasn't
    # landed yet; attribute it to the last step seen rather than dropping it
    for row in pending:
        flips.append({"step": last_step, **row})
    return {"steps": steps, "val": vals, "flips": flips}


def add_flip_velocity(flips: list[dict]) -> None:
    """Per-tensor change since the previous checkpoint.

    `flip_pct` is cumulative vs the start-of-run snapshot, which cannot distinguish a
    code that settled early from one still oscillating. The delta can: a tensor whose
    cumulative count rises while its per-interval count falls is converging.
    """
    prev: dict[str, dict] = {}
    for row in sorted(flips, key=lambda r: (r["step"], r["tensor"])):
        p = prev.get(row["tensor"])
        row["flip_pct_delta"] = None if p is None else round(row["flip_pct"] - p["flip_pct"], 6)
        row["z2nz_delta"] = None if p is None else row["zero_to_nonzero"] - p["zero_to_nonzero"]
        prev[row["tensor"]] = row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:  # union of keys, first-seen order
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summarize(data: dict[str, list[dict]]) -> dict:
    steps, vals, flips = data["steps"], data["val"], data["flips"]
    out: dict = {"n_steps_logged": len(steps), "n_val": len(vals), "n_flip_rows": len(flips)}
    if steps:
        peak = max(steps, key=lambda r: r["loss"])
        out |= {
            "step_last": steps[-1]["step"],
            "total_steps": steps[-1]["total_steps"],
            "loss_first": steps[0]["loss"],
            "loss_last": steps[-1]["loss"],
            "loss_peak": peak["loss"],
            "loss_peak_step": peak["step"],
            "s_per_step_first": steps[0]["s_per_step"],
            "s_per_step_last": steps[-1]["s_per_step"],
            "mem_gib_max": max(r["mem_gib"] for r in steps),
        }
    if vals:
        best = min(vals, key=lambda r: r["val_masked_ce"])
        out |= {"val_first": vals[0]["val_masked_ce"], "val_last": vals[-1]["val_masked_ce"],
                "val_best": best["val_masked_ce"], "val_best_step": best["step"]}
    if flips:
        last = max(r["step"] for r in flips)
        tail = [r for r in flips if r["step"] == last]
        out |= {
            "flip_report_step": last,
            "flip_pct_max": max(r["flip_pct"] for r in tail),
            "flip_pct_mean": round(sum(r["flip_pct"] for r in tail) / len(tail), 6),
            "flip_pct_min": min(r["flip_pct"] for r in tail),
            "scale_drift_pct_mean": round(
                sum(r["scale_drift_pct"] for r in tail) / len(tail), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="directory for the CSVs")
    args = ap.parse_args()

    data = parse(args.log.read_text(errors="replace"))
    add_flip_velocity(data["flips"])
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "steps.csv", data["steps"])
    write_csv(args.out / "val.csv", data["val"])
    write_csv(args.out / "flips.csv", data["flips"])
    summary = summarize(data)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

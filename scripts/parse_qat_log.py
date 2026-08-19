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

# Two generations of the same line, both accepted — a report must not silently lose a run
# because the trainer gained a field:
#   MPS:  "[qat] step 155/522 loss=1.2212 lr=4.30e-04 mem=30.8GiB 372.0s/step"
#   CUDA: "[qat] step 5/613 loss=1.0837 lr=6.67e-05 gnorm=1.51 mem=31.6/70.6GiB 62.7s/step"
#   KD:   "[qat] step 10/613 loss=0.8744 kl=0.5535 lr=1.67e-04 gnorm=1.42 ..."
#   +anchor: "... loss=0.8744 kl=0.5535 an=0.0121 lr=1.67e-04 ..."
# kl, gnorm and the /peak half of mem are optional; without them the columns come out
# empty rather than the row failing to parse at all.
STEP_RE = re.compile(
    r"^\[qat\] step (?P<step>\d+)/(?P<total>\d+) loss=(?P<loss>[\d.]+) "
    r"(?:kl=(?P<kl>[\d.eE+-]+) )?(?:an=(?P<an>[\d.eE+-]+) )?"
    r"(?:st=(?P<st>[\d.eE+-]+) )?"
    r"lr=(?P<lr>[\d.eE+-]+) (?:gnorm=(?P<gnorm>[\d.eE+-]+) )?"
    r"mem=(?P<mem>[\d.]+)(?:/(?P<peak>[\d.]+))?GiB (?P<sps>[\d.]+)s/step"
)
# "[qat] step 120 VAL masked-CE 1.1017" (newer runs append "(4 windows in 59s)")
VAL_RE = re.compile(r"^\[qat\] step (\d+) VAL masked-CE ([\d.]+)")
# The in-training termination probe. Masked-CE cannot see a collapsed stop decision
# (sft32k's val went flat for 225 steps while P(stop|sentence end) went to 0.97), so
# this series is parsed separately and plotted on its own axis.
STOPPROBE_RE = re.compile(r"^\[qat\] step (?P<step>\d+) STOPPROBE (?P<body>.*)$")
KV_RE = re.compile(r"(?P<k>[a-z_]+)=(?P<v>[\d.eE+-]+)")
# Also two generations. The newer one adds a third counter inside the parens and a density
# segment before scale-drift, so the tail is matched loosely on purpose:
#   old: "  ...q_proj: flips 1.8593% (0->±:131663 ±->0:179612) scale-drift 2.29%"
#   new: "  ...q_proj: flips 1.2445% (0->±:102280 ±->0:106511 ±->∓:0) density 64.9->64.9%
#         scale-drift 0.68% (+0.08%)"
FLIP_RE = re.compile(
    r"^\s+(?P<tensor>\S+): flips (?P<pct>[\d.]+)% "
    r"\(0->\S+:(?P<z2nz>\d+) \S+->0:(?P<nz2z>\d+)(?: \S+->\S+:(?P<sign>\d+))?\)"
    r".*?scale-drift (?P<drift>[\d.]+)%"
)
CKPT_RE = re.compile(r"^\[qat\] checkpoint @ step (\d+)")


def parse(text: str) -> dict[str, list[dict]]:
    steps: list[dict] = []
    vals: list[dict] = []
    probes: list[dict] = []
    flips: list[dict] = []
    pending: list[dict] = []  # flip rows seen since the last checkpoint line
    last_step = 0

    for line in text.splitlines():
        if m := STEP_RE.match(line):
            last_step = int(m.group("step"))
            steps.append({
                "step": last_step,
                "total_steps": int(m.group("total")),
                "loss": float(m.group("loss")),
                "kd_kl": float(m.group("kl")) if m.group("kl") else None,
                "stop_anchor": float(m.group("an")) if m.group("an") else None,
                "steer": float(m.group("st")) if m.group("st") else None,
                "lr": float(m.group("lr")),
                "grad_norm": float(m.group("gnorm")) if m.group("gnorm") else None,
                "mem_gib": float(m.group("mem")),
                "mem_peak_gib": float(m.group("peak")) if m.group("peak") else None,
                "s_per_step": float(m.group("sps")),
            })
        elif m := VAL_RE.match(line):
            vals.append({"step": int(m.group(1)), "val_masked_ce": float(m.group(2))})
        elif m := STOPPROBE_RE.match(line):
            # Only the "k=v" pairs before the bracketed gloss; the gloss repeats two of
            # them with reference values and would overwrite the real ones.
            body = m.group("body").split("[")[0]
            row = {"step": int(m.group("step"))}
            for kv in KV_RE.finditer(body):
                row[kv.group("k")] = float(kv.group("v"))
            probes.append(row)
        elif m := FLIP_RE.match(line):
            z2nz, nz2z = int(m.group("z2nz")), int(m.group("nz2z"))
            pending.append({
                "tensor": m.group("tensor"),
                "flip_pct": float(m.group("pct")),
                "zero_to_nonzero": z2nz,
                "nonzero_to_zero": nz2z,
                # ±->∓ is a straight sign reversal, a different event from recruit/prune:
                # it changes what a weight does without changing how many are live.
                "sign_flips": int(m.group("sign")) if m.group("sign") else None,
                # >1 = recruiting dead weights, <1 = pruning, ~1 = sign reorganization.
                # The split by tensor type is the whole point of tracking both counts.
                "densify_ratio": (z2nz / nz2z) if nz2z else None,
                "net_density_delta": z2nz - nz2z,
                "scale_drift_pct": float(m.group("drift")),
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
    return {"steps": steps, "val": vals, "flips": flips, "probes": probes}


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
    write_csv(args.out / "stopprobe.csv", data["probes"])
    write_csv(args.out / "flips.csv", data["flips"])
    summary = summarize(data)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

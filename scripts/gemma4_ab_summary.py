"""Read a set of gemma-4 stage arms side by side — the four numbers, on one screen.

No single column decides an arm. A ternary run can lower its loss purely by drifting
scales with **zero** code flips; it can hold a beautiful validation curve while its
termination policy collapses (sft32k went flat for 225 steps while `sentence_period`
climbed to 0.97); and it can flip plenty of codes in the wrong direction. So the arms
are compared on all of:

``recovered``  the go/no-go metric — how much of the stage's OWN untrained damage the
               training took back, from ``gemma4_stage_damage.py``
``flip%``      codes actually moved. Near zero means the arm learned nothing.
``diag/ctrl``  the stop probe against gemma's measured 0.00274 / 0.0703
``val``        masked CE, first -> last
``gnorm``      read next to ``--clip-norm``: a gnorm far above the clip means every step
               is a fixed-size step in the gradient direction, and the lr is not the
               lever it looks like

Usage::

    python scripts/gemma4_ab_summary.py out/gemma4-ternary/ab-lr*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_qat_log import parse  # noqa: E402

from quant_tuner.qat.stop_probe import PROBE_SPECS  # noqa: E402

SPEC = PROBE_SPECS["gemma4"]


def arm_row(run: Path) -> dict:
    row: dict = {"arm": run.name}
    log = run / "train.log"
    if log.exists():
        d = parse(log.read_text(errors="replace"))
        steps, vals, probes, flips = d["steps"], d["val"], d["probes"], d["flips"]
        if steps:
            row |= {"steps": steps[-1]["step"], "loss": steps[-1]["loss"],
                    "gnorm": steps[-1].get("gnorm"), "lr": steps[-1].get("lr"),
                    "s_step": steps[-1].get("s_per_step")}
        if vals:
            row |= {"val_first": vals[0]["val_masked_ce"], "val_last": vals[-1]["val_masked_ce"]}
        if probes:
            row |= {"diag": probes[-1].get(SPEC.diagnostic),
                    "ctrl": probes[-1].get(SPEC.control)}
        if flips:
            last = max(r["step"] for r in flips)
            fl = [r["flip_pct"] for r in flips if r["step"] == last and r.get("flip_pct")]
            row["flip_pct"] = sum(fl) / len(fl) if fl else None
        row["aborted"] = "PROBE-ABORT" in log.read_text(errors="replace")
    dmg = run / "stage_damage_trained.json"
    if dmg.exists():
        b = json.loads(dmg.read_text())["rows"]
        row["kld"] = b["trained"]["kld"]
        row["kld_untrained"] = b["untrained"]["kld"]
        row["recovered"] = b.get("recovered_frac", {}).get("kld")
    return row


def fmt(v, spec="{:.4f}", dash="—"):
    return dash if v is None else spec.format(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = [arm_row(r) for r in args.runs if r.is_dir()]
    hdr = (f"{'arm':22s} {'steps':>5s} {'recovered':>9s} {'kld':>8s} {'flip%':>7s} "
           f"{'diag':>8s} {'ctrl':>8s} {'val':>15s} {'gnorm':>7s} {'s/step':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        val = (f"{fmt(r.get('val_first'), '{:.3f}')}->{fmt(r.get('val_last'), '{:.3f}')}"
               if r.get("val_first") is not None else "—")
        rec = fmt(r.get("recovered"), "{:.1%}")
        print(f"{r['arm'][:22]:22s} {fmt(r.get('steps'), '{:d}', '—'):>5s} {rec:>9s} "
              f"{fmt(r.get('kld')):>8s} {fmt(r.get('flip_pct'), '{:.3f}'):>7s} "
              f"{fmt(r.get('diag')):>8s} {fmt(r.get('ctrl')):>8s} {val:>15s} "
              f"{fmt(r.get('gnorm'), '{:.1f}'):>7s} {fmt(r.get('s_step'), '{:.1f}'):>7s}"
              + ("  ABORTED" if r.get("aborted") else ""))
    print(f"\nreference: diagnostic {SPEC.vanilla[0]:.5f}  control {SPEC.vanilla[1]:.5f} "
          f"(shipped E4B); GO is recovered >= 70%")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the reasoning-ladder results as the markdown table the release card wants.

Reads the per-level CSVs written by reasoning_sweep.sh / the `high` run and emits one
table, ordered by reasoning depth rather than by whatever order the runs finished in.

Deliberately reports the raw scored-turn COUNT alongside each rate: at n=174 a single
turn is 0.6pp, so two levels differing by "0.006" differ by exactly one turn and should
not be read as a ranking. Printing the count makes that visible instead of implied.

    PYTHONPATH=src .venv/bin/python scripts/summarize_reasoning_ladder.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Ordered from most reasoning to least. `high` sits between medium and xhigh by design
# (its instruction is a strict clause-subset of xhigh's).
LEVELS = [
    ("xhigh",  "`xhigh` *(default)*", "think carefully + validate assumptions + consider alternatives"),
    ("high",   "`high`",              "think carefully + validate assumptions"),
    ("medium", "`medium`",            "no instruction — native reasoning"),
    ("low",    "`low`",               "keep thinking brief and focused"),
    ("off",    "`enable_thinking:false`", "pre-closed `<think></think>`"),
]


def read_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    return rows[-1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("out/exp-060-w4a16-32k/results"))
    ap.add_argument("--label", default="w4a16")
    ap.add_argument("--n-turns", type=int, default=174)
    a = ap.parse_args()

    print(f"| Reasoning level | What it injects | Tool-sel acc | Turns | Param acc | Schema |")
    print(f"|:---|:---|---:|---:|---:|---:|")
    best = (None, -1.0)
    seen = []
    for key, label, blurb in LEVELS:
        row = read_row(a.results / f"toolcall_{a.label}_{key}.csv")
        if row is None:
            print(f"| {label} | {blurb} | *pending* | — | *pending* | *pending* |")
            continue
        sel = float(row["tool_selection_acc"])
        par = float(row["param_acc_mean"])
        sch = float(row["schema_valid_rate"])
        seen.append((key, sel))
        if sel > best[1]:
            best = (label, sel)
        print(f"| {label} | {blurb} | **{sel:.3f}** | {round(sel * a.n_turns)}/{a.n_turns} "
              f"| {par:.3f} | {sch:.3f} |")

    if len(seen) >= 2:
        spread = max(s for _, s in seen) - min(s for _, s in seen)
        print(f"\nspread across measured levels: {spread:.3f} "
              f"({round(spread * a.n_turns)} turns of {a.n_turns})")
        print(f"best: {best[0]} at {best[1]:.3f}")
        print(f"one turn = {1 / a.n_turns:.4f} — treat gaps under ~0.02 as ties")


if __name__ == "__main__":
    main()

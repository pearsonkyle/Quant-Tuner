#!/usr/bin/env python
"""One table over every real-corpus stop read-out in ``out/gemma4-ternary/stopcorpus``.

The two halves of the termination failure are reported separately on purpose. They
moved independently under stop-weight (5.5 fixed discrimination and left commitment
flat), so a single blended number hides which one a run actually repaired:

* **commitment** — ``frac_top1``: at a real stop target, is stopping the model's top
  choice? This is what the shipped model does 35% of the time and every trained arm
  does ~3-5% of the time.
* **discrimination** — ``ratio_mean``: P(stop) at real stop targets over P(stop) at
  ordinary supervised positions. Low ratio = the model volunteers stops mid-turn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_DIR = Path("out/gemma4-ternary/stopcorpus")
ORDER = ["shipped", "untrained-ternary", "dense-ft", "ce-only", "selfkd"]


def rows(d: Path) -> list[dict]:
    got = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    rank = {name: i for i, name in enumerate(ORDER)}
    return sorted(got, key=lambda r: (rank.get(r["label"], len(ORDER)), r["label"]))


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    print(f"{'arm':26s} {'commit':>7s} {'p@stop':>8s} {'elsewhere':>11s} {'ratio':>9s}")
    for r in rows(d):
        at, el = r["at_stop_target"], r["elsewhere"]
        print(
            f"{r['label']:26s} {at['frac_top1']:6.1%} {at['mean']:8.4f} "
            f"{el['mean']:11.2e} {r['ratio_mean']:9.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

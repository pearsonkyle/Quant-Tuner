"""Refuse a KD table that would train on the wrong thing, before 3 GPU-hours ride on it.

Four failures this catches, none of which announce themselves during training:

* **wrong corpus** — window/position indices from another corpus resolve fine and distil
  against the wrong distribution at every position.
* **partial coverage** — uncovered windows silently train on plain CE while the rest
  train on CE+KL, so the run is two experiments averaged together.
* **no forced stop id** — `--stop-anchor` needs the stop token in EVERY support row.
  On a plain top-K table ~98% of rows lack it (measured on our corpus) and there is no
  per-position teacher P(stop) to anchor to.
* **low support coverage** — `coverage` is the teacher mass the stored top-K captured.
  Below ~0.8 the KL constrains far less of the distribution than it appears to, and
  nothing downstream can detect it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from quant_tuner.qat.kd_table import KDTable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--stop-id", type=int, default=106)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    args = ap.parse_args()

    blob = torch.load(args.corpus, weights_only=False, mmap=True)
    fp, n_win = blob.get("fingerprint"), blob["ids"].shape[0]
    # One mmap read: KDTable does not retain the payload, and loading a 2.4 GB table
    # twice to read one list from it is a silly way to spend the check.
    payload = torch.load(args.table, weights_only=False, mmap=True)
    t = KDTable(payload, corpus_fingerprint=fp)           # raises on a fingerprint clash

    fails = []
    missing = [w for w in range(n_win) if not t.has_window(w)]
    if missing:
        fails.append(f"{len(missing)}/{n_win} windows absent from the table "
                     f"(first: {missing[:5]}) — those would train on plain CE")
    inc = list(payload.get("include_ids") or [])
    if args.stop_id not in inc:
        fails.append(f"stop id {args.stop_id} not in include_ids={inc} — --stop-anchor "
                     f"cannot read a per-position teacher P(stop)")
    cov = t.coverage()
    if cov < args.min_coverage:
        fails.append(f"support coverage {cov:.4f} < {args.min_coverage} — the KL "
                     f"constrains much less of the teacher than it appears to")

    print(f"[kd-verify] {args.table}")
    print(f"  corpus fingerprint  {fp}  ({n_win} windows, all present: {not missing})")
    print(f"  forced ids          {inc}")
    print(f"  support coverage    {cov:.4f}")
    print(f"  {t!r}")
    for f in fails:
        print(f"  FAIL: {f}")
    print("[kd-verify] " + ("OK" if not fails else f"{len(fails)} FAILURE(S)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Splice the measured reasoning ladder into the release card.

The card carries a `<!-- LADDER_TABLE -->` marker; this replaces it (or a previously
generated table under it) with the current numbers. Idempotent — safe to re-run each time
another level finishes, which is the point: the card should never be hand-typed from a
CSV, because that is how a stale number survives into a release.

    PYTHONPATH=src .venv/bin/python scripts/fill_ladder_table.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MARKER = "<!-- LADDER_TABLE -->"
# Must match the card verbatim. A stale sentinel silently turns "replace" into "append",
# duplicating the table on every run — so this is asserted, not assumed.
END = "**Read the endpoints, not the ordering.**"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", type=Path,
                    default=Path("out/exp-060-w4a16-32k/release/README.md"))
    ap.add_argument("--results", type=Path,
                    default=Path("out/exp-060-w4a16-32k/results"))
    a = ap.parse_args()

    out = subprocess.run(
        [sys.executable, "scripts/summarize_reasoning_ladder.py",
         "--results", str(a.results)],
        capture_output=True, text=True, check=True).stdout
    table = "\n".join(l for l in out.splitlines() if l.startswith("|")).strip()
    if not table:
        print("no table produced — nothing written")
        return

    text = a.card.read_text()
    if MARKER not in text:
        print(f"marker {MARKER!r} not found in {a.card} — nothing written")
        return

    head, rest = text.split(MARKER, 1)
    if END not in rest:
        raise SystemExit(
            f"end sentinel {END!r} not found after the marker — refusing to splice, "
            "because appending instead of replacing would duplicate the table. "
            "Update END to match the card."
        )
    # Drop any previously spliced table, keeping everything from the note onward.
    tail = rest[rest.index(END):]
    a.card.write_text(f"{head}{MARKER}\n\n{table}\n\n{tail}")

    written = a.card.read_text()
    n = written.count("| Reasoning level | What it injects |")
    if n != 1:
        raise SystemExit(f"post-write check failed: {n} ladder tables in the card, expected 1")

    pending = table.count("*pending*")
    print(f"spliced ladder into {a.card}")
    print(f"  rows: {table.count(chr(10)) - 1}, still pending: {pending}")


if __name__ == "__main__":
    main()

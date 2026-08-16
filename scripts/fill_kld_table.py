#!/usr/bin/env python3
"""Splice the six-distribution KLD results into the release card.

Same contract as fill_ladder_table.py: replaces a `<!-- KLD_TABLE -->` marker, is
idempotent, and never requires a number to be retyped by hand from a CSV. Because
run_hf_kld.py now rewrites its CSV after every corpus, this can be run mid-sweep and will
show the completed distributions with the rest marked pending.

    PYTHONPATH=src .venv/bin/python scripts/fill_kld_table.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MARKER = "<!-- KLD_TABLE -->"
# Must match the card verbatim. If this sentinel ever stops matching, the splice silently
# degrades from "replace" to "append" and duplicates the table on every run — so the
# caller asserts a single occurrence rather than trusting it.
END = "**KLD is 3–10× lower"

# label -> (what it is, whether it is a holdout or a fit probe)
BLURB = {
    "external": ("code + math + tools, disjoint from calibration — **the headline**", True),
    "general":  ("broad English (`combined_en_tiny`)", True),
    "tools":    ("held-out CLI + agent log sessions", True),
    "agentic":  ("held-out SWE trajectories", True),
    "broad":    ("held-out broad-instruct", True),
    "cal8k":    ("slice of the *previous* 8192-packed corpus — a **fit** probe, not a holdout", False),
}
ORDER = ["external", "general", "tools", "agentic", "broad", "cal8k"]


def build_table(csv_path: Path) -> tuple[str, int]:
    rows = {}
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open()):
            rows[r["corpus"]] = r

    out = ["| eval | what it is | median KLD | top-1 agree | top-5 agree | ppl (bf16 → W4A16) |",
           "|:---|:---|---:|---:|---:|---:|"]
    pending = 0
    for label in ORDER:
        what, _ = BLURB[label]
        r = rows.get(label)
        if r is None:
            pending += 1
            out.append(f"| `{label}` | {what} | *pending* | — | — | — |")
            continue
        out.append(
            f"| `{label}` | {what} | **{float(r['median_kld']):.4f}** "
            f"| {float(r['top1_agree']):.1f}% | {float(r['top5_agree']):.1f}% "
            f"| {float(r['ref_ppl']):.3f} → {float(r['quant_ppl']):.3f} |"
        )
    return "\n".join(out), pending


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", type=Path,
                    default=Path("out/exp-060-w4a16-32k/release/README.md"))
    ap.add_argument("--csv", type=Path,
                    default=Path("out/exp-060-w4a16-32k/results/kld_results.csv"))
    a = ap.parse_args()

    table, pending = build_table(a.csv)
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
    tail = rest[rest.index(END):]
    a.card.write_text(f"{head}{MARKER}\n\n{table}\n\n{tail}")

    written = a.card.read_text()
    n = written.count("| eval | what it is |")
    if n != 1:
        raise SystemExit(f"post-write check failed: {n} KLD tables in the card, expected 1")
    print(f"spliced KLD table into {a.card} ({6 - pending}/6 distributions, {pending} pending)")


if __name__ == "__main__":
    main()

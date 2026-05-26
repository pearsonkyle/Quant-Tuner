"""Render exp-003 results.csv as the two markdown tables in README.md.

Reads out/exp-003/results.csv (one row per cell, labels of the form
`{model}|{sub/name}|{dataset}|tok=...|ctx=...K`) and writes out/exp-003/TABLES.md
with both sub-experiment A and sub-experiment B tables. B31 is duplicated
from A1 since they're the same row.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "out" / "exp-003" / "results.csv"
OUT = REPO / "out" / "exp-003" / "TABLES.md"


def _fmt(v: str, digits: int) -> str:
    if v in ("", None):
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except ValueError:
        return v


def _parse_label(label: str) -> dict[str, str]:
    """Pull `name`, `dataset`, `tok`, `ctx` out of the bench row label."""
    parts = label.split("|")
    if len(parts) < 5:
        return {}
    _model, sub_name, dataset, tok_part, ctx_part = parts[0], parts[1], parts[2], parts[3], parts[4]
    name = sub_name.split("/", 1)[1] if "/" in sub_name else sub_name
    return {
        "name": name,
        "dataset": dataset,
        "tok": tok_part.split("=", 1)[1] if "=" in tok_part else tok_part,
        "ctx": ctx_part.split("=", 1)[1] if "=" in ctx_part else ctx_part,
    }


def main() -> int:
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS} — run scripts/run_exp003.py first")

    by_name: dict[str, dict] = {}
    for r in csv.DictReader(RESULTS.open()):
        meta = _parse_label(r.get("model", ""))
        if not meta:
            continue
        by_name[meta["name"]] = {
            **meta,
            "ppl":         _fmt(r.get("ppl"), 4),
            "kld":         _fmt(r.get("mean_kld"), 5),
            "same_top_p":  _fmt(r.get("same_top_p"), 4),
        }

    # B31 is an alias of A1
    if "A1" in by_name and "B31" not in by_name:
        b31 = dict(by_name["A1"])
        b31["name"] = "B31"
        by_name["B31"] = b31

    lines = []

    lines.append("## A. Dataset mixing\n")
    lines.append("| cell | dataset | tokens | ctx | PPL | KLD (mean) | same_top_p |")
    lines.append("|------|---------|--------|-----|-----|------------|------------|")
    for name in ("A1", "A2", "A3"):
        r = by_name.get(name)
        if not r:
            lines.append(f"| {name} | _missing_ | | | | | |")
            continue
        lines.append(f"| {r['name']} | {r['dataset']} | {r['tok']} | {r['ctx']} | "
                     f"{r['ppl']} | {r['kld']} | {r['same_top_p']} |")

    lines.append("")
    lines.append("## B. Custom seq × tokens grid\n")
    lines.append("| cell | tokens | ctx | PPL | KLD (mean) | same_top_p |")
    lines.append("|------|--------|-----|-----|------------|------------|")
    for name in ("B11", "B12", "B13", "B21", "B22", "B23", "B31", "B32", "B33"):
        r = by_name.get(name)
        if not r:
            lines.append(f"| {name} | | | | | |")
            continue
        lines.append(f"| {r['name']} | {r['tok']} | {r['ctx']} | "
                     f"{r['ppl']} | {r['kld']} | {r['same_top_p']} |")

    md = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md)
    print(md)
    print(f"# wrote {OUT.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

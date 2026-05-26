"""Render out/exp-001/results.csv as the 8-column markdown table for README.md.

Columns: model | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p

The driver writes labels of the form "{repo_id}|{technique}|{dataset}" — this
script splits them back out so the table is sortable and readable.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "out" / "exp-001" / "results.csv"
TABLE = REPO / "out" / "exp-001" / "TABLE.md"

COLS = ["model", "technique", "dataset", "size (GiB)", "BPW", "PPL", "KLD (mean)", "same_top_p"]


def _fmt(v: str, digits: int) -> str:
    if v in ("", None):
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except ValueError:
        return v


def render(csv_path: Path) -> str:
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path} — run scripts/run_exp001.py first")
    rows = list(csv.DictReader(csv_path.open()))
    parsed = []
    for r in rows:
        label = r.get("model", "")
        parts = label.split("|", 2)
        if len(parts) != 3:
            # skip rows that don't look like exp-001 labels (e.g. stray rows)
            continue
        model, technique, dataset = parts
        parsed.append({
            "model": model,
            "technique": technique,
            "dataset": dataset,
            "size (GiB)": _fmt(r.get("size_gib", ""), 2),
            "BPW": _fmt(r.get("bpw", ""), 3),
            "PPL": _fmt(r.get("ppl", ""), 4),
            "KLD (mean)": _fmt(r.get("mean_kld", ""), 5),
            "same_top_p": _fmt(r.get("same_top_p", ""), 4),
        })

    technique_order = {"imatrix": 0, "none": 1}
    dataset_order = {"custom": 0, "wiki.test.raw": 1, "—": 2}
    parsed.sort(key=lambda r: (
        r["model"],
        technique_order.get(r["technique"], 99),
        dataset_order.get(r["dataset"], 99),
    ))

    out = ["| " + " | ".join(COLS) + " |",
           "|" + "|".join("---" for _ in COLS) + "|"]
    for r in parsed:
        out.append("| " + " | ".join(r[c] for c in COLS) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    md = render(RESULTS)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(md)
    print(md)
    print(f"# wrote {TABLE.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

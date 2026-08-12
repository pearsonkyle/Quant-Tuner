"""Build a single combined markdown table across all Gemma-31B AWQ experiments.

Ingests results.csv from exp-009 (imatrix only), exp-010 (naive AWQ),
exp-014 (AWQ + o_proj/down_proj), exp-015 (AWQ q2k_b16 proxy), and exp-016
(AWQ per-tensor α). Writes one row per (quant, technique) sorted first by
quant then by technique. Sources whose results.csv does not yet exist are
silently skipped, so this can be run incrementally as experiments finish.

Output: out/figures/awq_experiments_table.md
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "google__gemma-4-31B-it"

# (label, csv path, technique substring inside model column, sort-key)
SOURCES: list[tuple[str, Path, str, int]] = [
    ("imatrix (exp-009)",            REPO / "out" / "exp-009" / SLUG / "results.csv", "|imatrix|",                  0),
    ("awq+imatrix (exp-010)",        REPO / "out" / "exp-010" / SLUG / "results.csv", "|awq+imatrix|",              1),
    ("awq+outproj (exp-014)",        REPO / "out" / "exp-014" / SLUG / "results.csv", "|awq+outproj+imatrix|",      2),
    ("awq-q2kproxy (exp-015)",       REPO / "out" / "exp-015" / SLUG / "results.csv", "|awq-q2kproxy+imatrix|",     3),
    ("awq-pertensor (exp-016)",      REPO / "out" / "exp-016" / SLUG / "results.csv", "|awq-pertensor+imatrix|",    4),
    ("awq-cv-gate (exp-017)",        REPO / "out" / "exp-017" / SLUG / "results.csv", "|awq-cv-gate+imatrix|",      5),
    ("awq-cv-mixed (exp-018)",       REPO / "out" / "exp-018" / SLUG / "results.csv", "|awq-cv-mixed+imatrix|",     6),
]

# Quant display order; anything not listed falls to the end alphabetically.
QUANT_ORDER = ["FP16", "IQ2_XXS", "IQ2_XS", "IQ2_M", "Q2_K_S", "Q2_K",
               "IQ3_M", "IQ4_XS", "IQ4_NL", "Q4_K_M"]

OUT = REPO / "out" / "figures" / "awq_experiments_table.md"


def _fmt(v, places: int) -> str:
    try:
        return f"{float(v):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _quant_sort_key(q: str) -> tuple[int, str]:
    try:
        return (QUANT_ORDER.index(q), q)
    except ValueError:
        return (len(QUANT_ORDER), q)


def _fp16_from_exp09_log() -> tuple[float, float, float] | None:
    """Return (size_gib, bpw, ppl) for the FP16 reference if available."""
    f16 = REPO / "out" / "exp-009" / SLUG / "model-f16.gguf"
    log = REPO / "out" / "exp-009" / SLUG / "logs" / "baseline.log"
    if not f16.exists():
        return None
    size_gib = f16.stat().st_size / (1024 ** 3)
    # bpw is unavailable without n_params; compute from imatrix row if present.
    ppl = None
    if log.exists():
        m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log.read_text())
        if m:
            ppl = float(m.group(1))
    return (size_gib, float("nan"), ppl if ppl is not None else float("nan"))


def main() -> int:
    rows_by_quant: dict[str, list[tuple[int, str, dict]]] = {}
    sources_seen = 0
    for label, csv_path, needle, sort_key in SOURCES:
        if not csv_path.exists():
            print(f"skip (no csv): {csv_path.relative_to(REPO)}", file=sys.stderr)
            continue
        sources_seen += 1
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                model = row.get("model", "")
                if needle not in model:
                    continue
                parts = model.split("|")
                if len(parts) < 2:
                    continue
                quant = parts[1]
                rows_by_quant.setdefault(quant, []).append((sort_key, label, row))

    if sources_seen == 0:
        print("no source CSVs found; nothing to write", file=sys.stderr)
        return 1

    fp16 = _fp16_from_exp09_log()

    lines = [
        "# AWQ experiments — combined comparison (Gemma-4-31B-it)",
        "",
        "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|",
    ]
    if fp16 is not None:
        size_gib, bpw, ppl = fp16
        bpw_str = "—" if bpw != bpw else _fmt(bpw, 3)  # NaN check
        lines.append(
            f"| FP16 | none | {_fmt(size_gib, 2)} | {bpw_str} | "
            f"{_fmt(ppl, 4)} | 0.00000 | 100.0000 |"
        )

    for quant in sorted(rows_by_quant, key=_quant_sort_key):
        entries = sorted(rows_by_quant[quant], key=lambda t: t[0])
        for _, label, row in entries:
            lines.append(
                f"| {quant} | {label} | "
                f"{_fmt(row.get('size_gib'), 2)} | {_fmt(row.get('bpw'), 3)} | "
                f"{_fmt(row.get('ppl'), 4)} | {_fmt(row.get('mean_kld'), 5)} | "
                f"{_fmt(row.get('same_top_p'), 4)} |"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}  ({sources_seen} source CSVs, "
          f"{sum(len(v) for v in rows_by_quant.values())} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Refresh EXPERIMENT.md tables from mtp_acceptance.json and mtp_eval_summary.json.

Reads the two result files produced by run_mtp_pipeline.py and run_mtp_eval_suite.py,
then fills any empty cells in the four EXPERIMENT.md tables. Already-filled cells are
left untouched so hand-entered notes survive repeated runs.

Usage::

    uv run python scripts/update_experiment_tables.py
    uv run python scripts/update_experiment_tables.py --dry-run   # print diff only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EXPERIMENT_MD   = REPO / "EXPERIMENT.md"
ACCEPTANCE_JSON = REPO / "out/benchmark_9b_iq3s/mtp_acceptance.json"
EVAL_SUMMARY    = REPO / "out/benchmark_9b_iq3s/eval/mtp_suite/mtp_eval_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None, pct: bool = False, decimals: int = 1,
         stdev: float | None = None) -> str:
    if v is None or v != v:  # None or NaN
        return ""
    scale = 100 if pct else 1
    s = f"{v * scale:.{decimals}f}"
    if stdev is not None and stdev == stdev and stdev > 0:
        s += f" ± {stdev * scale:.{decimals}f}"
    return s


def _parse_table_row(line: str) -> list[str] | None:
    """Return list of cell strings if line looks like a markdown table row."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [c.strip() for c in stripped[1:-1].split("|")]
    return cells


def _rebuild_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _is_empty(cell: str, force: bool = False) -> bool:
    return force or cell.strip() == ""


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(re.fullmatch(r"\|[-| :]+\|", stripped))


# ---------------------------------------------------------------------------
# Load JSON data
# ---------------------------------------------------------------------------

def _load_acceptance() -> dict[tuple[str, str], dict]:
    """Returns {(slug, variant): row} from mtp_acceptance.json."""
    if not ACCEPTANCE_JSON.exists():
        return {}
    data = json.loads(ACCEPTANCE_JSON.read_text())
    out: dict[tuple[str, str], dict] = {}
    for row in data:
        key = (row["slug"], row["variant"])
        out[key] = row
    return out


def _load_eval_summary() -> dict[tuple[str, str], dict]:
    """Returns {(slug, variant): row} from mtp_eval_summary.json."""
    if not EVAL_SUMMARY.exists():
        return {}
    data = json.loads(EVAL_SUMMARY.read_text())
    out: dict[tuple[str, str], dict] = {}
    for row in data:
        key = (row["slug"], row["variant"])
        out[key] = row
    return out


# ---------------------------------------------------------------------------
# Row updaters
# ---------------------------------------------------------------------------

def _update_acceptance_row(cells: list[str], acc_data: dict[tuple[str, str], dict],
                           force: bool = False) -> list[str]:
    """MTP Performance Acceptance: | Model | Quant | MTP Variant | n=1 acc% | n=2 acc% | n=3 acc% |"""
    if len(cells) < 6:
        return cells
    slug, variant = cells[0], cells[2]
    key = (slug, variant)
    if key not in acc_data:
        return cells
    row = acc_data[key]
    sweep = {r["n_max"]: r for r in row.get("sweep", [])}
    new = cells[:]
    for idx, n in enumerate([1, 2, 3], start=3):
        if _is_empty(new[idx], force) and n in sweep:
            ar  = sweep[n].get("accept_rate")
            sd  = sweep[n].get("accept_rate_stdev")
            new[idx] = _fmt(ar, pct=True, stdev=sd) if ar is not None else ""
    return new


def _update_throughput_row(cells: list[str], acc_data: dict[tuple[str, str], dict],
                           force: bool = False) -> list[str]:
    """MTP Throughput: | Model | Quant | MTP Variant | baseline tok/s | n=1 tok/s | n=2 tok/s | n=3 tok/s |"""
    if len(cells) < 7:
        return cells
    slug, variant = cells[0], cells[2]
    key = (slug, variant)
    if key not in acc_data:
        return cells
    row = acc_data[key]
    sweep = {r["n_max"]: r for r in row.get("sweep", [])}
    new = cells[:]
    if _is_empty(new[3], force):
        bl   = row.get("baseline_tps")
        bl_s = row.get("baseline_tps_stdev")
        new[3] = _fmt(bl, stdev=bl_s) if bl is not None else ""
    for idx, n in enumerate([1, 2, 3], start=4):
        if _is_empty(new[idx], force) and n in sweep:
            tps  = sweep[n].get("mean_tps")
            tps_s = sweep[n].get("mean_tps_stdev")
            new[idx] = _fmt(tps, stdev=tps_s) if tps is not None else ""
    return new


def _update_general_row(cells: list[str], eval_data: dict[tuple[str, str], dict],
                        force: bool = False) -> list[str]:
    """General Performance: | Model | Quant | Technique | MTP Variant | Tool Sel% | Param Acc% | Schema% | Rollout% | MMLU-Pro% |"""
    if len(cells) < 9:
        return cells
    slug, variant = cells[0], cells[3]
    key = (slug, variant)
    if key not in eval_data:
        return cells
    row = eval_data[key]
    new = cells[:]
    mapping = [
        (4, "tool_selection_acc_mean",   True),
        (5, "param_acc_mean_mean",        True),
        (6, "schema_valid_rate_mean",     True),
        (7, "rollout_complete_rate_mean", True),
        (8, "accuracy_mean",              True),
    ]
    for idx, key_name, is_pct in mapping:
        if _is_empty(new[idx], force):
            val = row.get(key_name)
            new[idx] = _fmt(val, pct=is_pct) if val is not None else ""
    return new


# ---------------------------------------------------------------------------
# Main pass over EXPERIMENT.md
# ---------------------------------------------------------------------------

_TABLE_ACCEPTANCE = "MTP Performance Acceptance"
_TABLE_THROUGHPUT = "MTP Throughput"
_TABLE_GENERAL    = "General Performance"
_TABLE_REDTEAM    = "Red Team Performance"

_SKIP_TABLES = {_TABLE_REDTEAM}  # no automated source yet


def update(content: str, acc_data: dict, eval_data: dict, force: bool = False) -> str:
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    current_table: str | None = None
    in_header = False  # True while processing the header row
    past_separator = False  # True once we've seen the |---|...| row

    for line in lines:
        stripped = line.rstrip("\n")

        # Detect table section headers (lines like "General Performance:")
        for label in [_TABLE_GENERAL, _TABLE_REDTEAM, _TABLE_ACCEPTANCE, _TABLE_THROUGHPUT]:
            if stripped.strip().startswith(label):
                current_table = label
                in_header = True
                past_separator = False
                break

        cells = _parse_table_row(stripped)

        if cells is None or current_table in _SKIP_TABLES:
            result.append(line)
            continue

        if _is_separator(stripped):
            past_separator = True
            in_header = False
            result.append(line)
            continue

        if in_header or not past_separator:
            # header row — leave as-is
            result.append(line)
            continue

        # Data row — attempt to fill empty cells
        updated = cells
        if current_table == _TABLE_ACCEPTANCE:
            updated = _update_acceptance_row(cells, acc_data, force)
        elif current_table == _TABLE_THROUGHPUT:
            updated = _update_throughput_row(cells, acc_data, force)
        elif current_table == _TABLE_GENERAL:
            updated = _update_general_row(cells, eval_data, force)

        if updated != cells:
            eol = "\n" if line.endswith("\n") else ""
            result.append(_rebuild_row(updated) + eol)
        else:
            result.append(line)

    return "".join(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changed lines without writing")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite already-filled cells (use after a fresh sweep)")
    parser.add_argument("--experiment-md", type=Path, default=EXPERIMENT_MD)
    parser.add_argument("--acceptance-json", type=Path, default=ACCEPTANCE_JSON)
    parser.add_argument("--eval-summary", type=Path, default=EVAL_SUMMARY)
    args = parser.parse_args()

    if not args.experiment_md.exists():
        print(f"ERROR: {args.experiment_md} not found", file=sys.stderr)
        return 1

    acc_data  = _load_acceptance()
    eval_data = _load_eval_summary()

    if not acc_data and not eval_data:
        # Nothing to do yet — pipeline hasn't produced results
        return 0

    original = args.experiment_md.read_text()
    updated  = update(original, acc_data, eval_data, force=args.force)

    if updated == original:
        return 0

    if args.dry_run:
        orig_lines = original.splitlines()
        upd_lines  = updated.splitlines()
        for i, (o, u) in enumerate(zip(orig_lines, upd_lines), 1):
            if o != u:
                print(f"line {i}:")
                print(f"  - {o}")
                print(f"  + {u}")
        return 0

    args.experiment_md.write_text(updated)
    changed = sum(1 for o, u in zip(original.splitlines(), updated.splitlines()) if o != u)
    print(f"updated {args.experiment_md.name}: {changed} line(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

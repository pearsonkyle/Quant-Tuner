"""Compute per-turn pass@5 from existing per-rep JSONL logs.

Maps JSONL files (one per rep) to (model, rep) pairs via chronological
ordering — `run_toolcall_reps.py` executes models in CLI order, reps 0..4
sequentially, and writes `toolcall_reps_<unix_ts>_rep<NN>.jsonl` for each.
For each run-spec (per-rep CSV → list of (model, rep) rows in order), take
the next N matching JSONL files in timestamp order.

For each model, group per-turn records by (session, turn) and compute
`pass@5` = fraction of turns where ≥1 rep got it right (for boolean
metrics like `selection`, `schema_valid`) or max-over-reps (for
continuous metrics like `param_acc`).

Writes out/exp-002/toolcall_passat5.csv and prints a markdown table.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "out" / "exp-002" / "logs"
OUT_CSV = REPO / "out" / "exp-002" / "toolcall_passat5.csv"

# Per-rep CSV → label for output. Order matters: chronological execution.
RUN_SPECS: list[tuple[Path, str]] = [
    (REPO / "out/exp-002/toolcall_reps_results.csv",               "initial-4-models"),
    (REPO / "out/exp-002/toolcall_reps_outliers_partial_results.csv", "outliers-partial-52pct"),
    (REPO / "out/exp-002/toolcall_reps_outliers_full_results.csv", "outliers-full-100pct"),
    (REPO / "out/exp-002/toolcall_reps_fp16_results.csv",          "fp16-baseline"),
    (REPO / "out/exp-002/toolcall_reps_mixed8k_results.csv",       "mixed8k-from-exp001"),
    (REPO / "out/exp-002/toolcall_reps_mixed8k_outlier_l4_results.csv", "mixed8k+outlier_l4"),
]


def _load_rep_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _map_files_to_rows() -> dict[tuple[str, int], Path]:
    """(model_label, rep_idx) → JSONL file path."""
    files = sorted(LOGS.glob("toolcall_reps_*.jsonl"))  # chronological by filename ts
    out: dict[tuple[str, int], Path] = {}
    cursor = 0
    for csv_path, run_label in RUN_SPECS:
        rows = _load_rep_rows(csv_path)
        n = len(rows)
        if n == 0:
            continue
        chunk = files[cursor:cursor + n]
        if len(chunk) < n:
            print(f"WARN: {run_label} expects {n} files but only {len(chunk)} available "
                  f"(cursor={cursor}, total files={len(files)})", file=sys.stderr)
        for row, f in zip(rows, chunk):
            # Sanity: filename's rep suffix must match row's rep
            fname_rep = int(f.stem.rsplit("_rep", 1)[-1])
            row_rep = int(row["rep"])
            assert fname_rep == row_rep, f"rep mismatch: {f.name} vs row {row_rep}"
            # Sanity: seed must be 1000 + rep
            assert int(row["seed"]) == 1000 + row_rep, f"seed mismatch on {row['model']}"
            out[(row["model"], row_rep)] = f
        cursor += n
    if cursor < len(files):
        print(f"WARN: {len(files) - cursor} unaccounted JSONL files past cursor={cursor}",
              file=sys.stderr)
    return out


def _load_turns(f: Path) -> list[dict]:
    return [json.loads(line) for line in f.read_text().splitlines()
            if line.strip()]


def _pass_at_5(file_by_rep: dict[int, Path]) -> dict[str, float | int]:
    """For one model, compute per-turn pass@5 across the 5 reps.

    A 'turn' is keyed by (session, turn_idx, kind). For each key seen in
    ≥1 rep, contribute to the metric:
      - boolean (selection, schema_valid): any(rep got it right)
      - continuous (param_acc): max(per-rep value)
    We score only `kind == 'tool_call'` records with truth_malformed=False
    (same filter the per-rep aggregator uses for the mean ± stdev numbers).
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for rep_idx, f in file_by_rep.items():
        for rec in _load_turns(f):
            if rec.get("kind") != "tool_call":
                continue
            if rec.get("truth_malformed"):
                continue
            key = (rec["session"], rec["turn"])
            by_key[key].append(rec)

    n_turns = len(by_key)
    if n_turns == 0:
        return {"n_turns": 0}

    sel_pass = sum(1 for recs in by_key.values() if any(r.get("selection") for r in recs))
    schema_pass = sum(1 for recs in by_key.values() if any(r.get("schema_valid") for r in recs))
    param_max_sum = sum(max(r.get("param_acc", 0.0) for r in recs) for recs in by_key.values())
    avg_reps = sum(len(recs) for recs in by_key.values()) / n_turns

    return {
        "tool_selection_passat5": sel_pass / n_turns,
        "schema_valid_passat5":   schema_pass / n_turns,
        "param_acc_passat5":      param_max_sum / n_turns,
        "n_turns": n_turns,
        "avg_reps_per_turn": avg_reps,
    }


def main() -> int:
    mapping = _map_files_to_rows()
    print(f"Mapped {len(mapping)} (model, rep) → JSONL files", file=sys.stderr)

    by_model: dict[str, dict[int, Path]] = defaultdict(dict)
    for (model, rep), f in mapping.items():
        by_model[model][rep] = f

    results: list[dict] = []
    for model in sorted(by_model.keys()):
        reps = by_model[model]
        if len(reps) < 2:
            print(f"  skip {model}: only {len(reps)} rep(s)", file=sys.stderr)
            continue
        metrics = _pass_at_5(reps)
        metrics["model"] = model
        metrics["n_reps"] = len(reps)
        results.append(metrics)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if results:
        keys = ["model", "n_reps", "n_turns", "avg_reps_per_turn",
                "tool_selection_passat5", "param_acc_passat5", "schema_valid_passat5"]
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in keys})

    # Markdown table
    print("\n| model | n_reps | n_turns | tool_sel pass@5 | param_acc pass@5 | schema_valid pass@5 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['n_reps']} | {r['n_turns']} | "
              f"{r['tool_selection_passat5']:.4f} | "
              f"{r['param_acc_passat5']:.4f} | "
              f"{r['schema_valid_passat5']:.4f} |")

    print(f"\nwrote {OUT_CSV.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Do quant-tuner's existing quality gates see safety drift at all?

Every metric this repo produces answers *"is the quant still capable?"* — BPW,
KLD, perplexity, tok/s, tool-call accuracy, MMLU-Pro, SWE-rebench pass rate.
This script asks whether any of them **predicts** the refusal drift measured by
``scripts/redteam_ladder.py``.

The expected — and most useful — answer is *no*. Refusal is low-probability-mass
behaviour, so an averaged divergence like mean-KLD or a perplexity ratio washes
it out; and the pipeline's own guardrails (``ppl_max_ratio``, ``sanity_max_rel``
in ``calibrate/gptq.py``) are deliberately *relaxed* at 2-3 bits, exactly where
alignment is most fragile. If the correlation is absent, then "we validated the
quant" as currently practised says nothing about whether the safety properties
survived — which is a measured claim, not an assumed one.

Pure CSV analysis: no model, no server, no GPU. Join key is the model label
(basename of the quant path, matching ``leaderboard.aggregate.merge_toolcall``).

Examples
--------
    PYTHONPATH=src .venv/bin/python scripts/redteam_vs_quality.py \
        --ladder out/redteam/ladder/ladder.csv \
        --bench out/run/results.csv \
        --toolcall out/toolcall_reps_aggregated.csv \
        --out out/redteam/ladder/vs_quality.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.red_team import spearman_rho  # noqa: E402

# Quality columns worth testing against safety drift, and which direction "better"
# runs in. `higher_is_better` only shapes the interpretation line we print — the
# correlation itself is direction-agnostic.
QUALITY_COLUMNS: dict[str, bool] = {
    "bpw": True,
    "size_gib": True,
    "mean_kld": False,
    "median_kld": False,
    "ppl": False,
    "ppl_ratio": False,
    "same_top_p": True,
    "tool_selection_acc": True,
    "param_acc_mean": True,
    "schema_valid_rate": True,
    "rollout_complete_rate": True,
    "accuracy": True,
    "pass_rate": True,
}

# Safety columns from ladder.csv to correlate against.
SAFETY_COLUMNS = ["net_drift", "pass_rate_delta", "n_flip_unsafe"]

OUT_COLUMNS = ["safety_metric", "quality_metric", "n_models", "spearman_rho", "reading"]


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _model_key(row: dict) -> str:
    """Join key: the bare model label, tolerating a full quant_path."""
    raw = row.get("model") or row.get("quant_path") or ""
    return os.path.basename(raw).removesuffix(".gguf")


def _as_float(value) -> float | None:
    if value in (None, "", "NA", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge(ladder: list[dict], sidecars: list[list[dict]]) -> list[dict]:
    """Pair each ladder row with the quality columns found for it in the sidecars.

    Accepts both single-rep CSVs (bare column names) and multi-rep aggregated
    CSVs (``<key>_mean``), the same two shapes ``merge_toolcall`` handles.

    Quality values are kept in a separate ``quality`` dict rather than merged
    into the ladder row. ``ladder.csv`` has its own ``pass_rate`` column (the
    red-team one) and so does the SWE-rebench aggregate; flattening them together
    would silently correlate the safety metric with itself.
    """
    by_key: dict[str, dict] = {}
    for rows in sidecars:
        for row in rows:
            by_key.setdefault(_model_key(row), {}).update(row)

    merged: list[dict] = []
    for row in ladder:
        key = _model_key(row)
        extra = by_key.get(key, {})
        quality: dict[str, str] = {}
        for col in QUALITY_COLUMNS:
            value = extra.get(col)
            if value in (None, ""):
                value = extra.get(f"{col}_mean")
            if value not in (None, ""):
                quality[col] = value
        merged.append({"safety": row, "quality": quality, "_key": key,
                       "_matched": bool(quality)})
    return merged


def _reading(rho: float | None) -> str:
    if rho is None:
        return "undefined (need >=3 rungs with both metrics, and some variance)"
    magnitude = abs(rho)
    if magnitude < 0.3:
        return "no usable relationship — this gate is blind to the drift"
    if magnitude < 0.7:
        return "weak/partial — not a substitute for measuring safety directly"
    return "tracks the drift — worth investigating as a cheap proxy"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ladder", type=Path, required=True,
                   help="ladder.csv from scripts/redteam_ladder.py")
    p.add_argument("--bench", type=Path, action="append", default=[],
                   help="results.csv (BPW/KLD/PPL/speed). Repeatable.")
    p.add_argument("--toolcall", type=Path, action="append", default=[],
                   help="Tool-call aggregated CSV. Repeatable.")
    p.add_argument("--mmlu", type=Path, action="append", default=[],
                   help="MMLU-Pro aggregated CSV. Repeatable.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the correlation table here (default: next to --ladder).")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    if not args.ladder.exists():
        raise SystemExit(f"no ladder CSV at {args.ladder} — run scripts/redteam_ladder.py first")

    sidecar_paths = [*args.bench, *args.toolcall, *args.mmlu]
    missing = [str(p) for p in sidecar_paths if not p.exists()]
    if missing:
        raise SystemExit("missing sidecar CSV(s): " + ", ".join(missing))
    if not sidecar_paths:
        raise SystemExit("pass at least one of --bench / --toolcall / --mmlu")

    ladder = _read_csv(args.ladder)
    merged = _merge(ladder, [_read_csv(p) for p in sidecar_paths])

    unmatched = [r["_key"] for r in merged if not r["_matched"]]
    if unmatched:
        print(f"⚠ no quality row matched for: {', '.join(unmatched)}")
    if len(merged) < 3:
        print(f"⚠ only {len(merged)} rung(s) — Spearman needs >=3; "
              f"add more quants to the ladder before reading anything into this.")

    rows: list[dict] = []
    for safety in SAFETY_COLUMNS:
        for quality in QUALITY_COLUMNS:
            pairs = [
                (_as_float(r["quality"].get(quality)), _as_float(r["safety"].get(safety)))
                for r in merged
            ]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            rho = spearman_rho([x for x, _ in pairs], [y for _, y in pairs])
            rows.append(
                {
                    "safety_metric": safety,
                    "quality_metric": quality,
                    "n_models": len(pairs),
                    "spearman_rho": "" if rho is None else rho,
                    "reading": _reading(rho),
                }
            )

    out = args.out or args.ladder.with_name("vs_quality.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 92)
    print("  DOES ANY EXISTING QUALITY GATE PREDICT SAFETY DRIFT?")
    print("=" * 92)
    print(f"  {'safety':<18}{'quality':<24}{'n':>4}{'rho':>8}   reading")
    print("  " + "-" * 88)
    for r in rows:
        rho = r["spearman_rho"]
        print(f"  {r['safety_metric']:<18}{r['quality_metric']:<24}{r['n_models']:>4}"
              f"{(f'{rho:+.2f}' if rho != '' else '   —'):>8}   {r['reading']}")
    print("=" * 92)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

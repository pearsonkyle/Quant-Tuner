"""Aggregate a results.csv into a sorted markdown leaderboard with SQS scores.

SQS (Squashed Quality Score) is a weighted geometric mean of three ratios
against the FP16 reference row:

    comp = size(f16) / size(quant)               compression  ≥ 1 for any quant
    fid  = same_top_p(quant) / 100               fidelity     ≤ 1 always
    spd  = decode_tok_s(quant) / decode_tok_s(f16) speedup   ≥ 1 means faster

    SQS = (comp^α · fid^β · spd^γ) ^ (1 / (α+β+γ))

With default weights α=1, β=2, γ=1, the score weights fidelity twice as
heavily as compression and speed. The FP16 row always lands at exactly 1.000
regardless of weights, so SQS values above 1 indicate "net better than F16
under the weighting" and values below indicate "net worse".

Tool-call columns are merged in optionally from a sidecar CSV produced by
``scripts/eval_toolcall.py`` (vendored). The join key is ``basename(quant_path)``
on the bench side ↔ ``model`` on the toolcall side; rows without a match
render as ``-``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# (csv_key, display_header). Order = column order in the rendered table.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("model", "Model"),
    ("size_gib", "Size (GiB)"),
    ("bpw", "BPW"),
    ("ppl", "PPL"),
    ("ppl_ratio", "PPL Ratio"),
    ("mean_kld", "Mean KLD"),
    ("median_kld", "Median KLD"),
    ("same_top_p", "Same Top p"),
    ("rms_dp", "RMS Δp"),
    # In-distribution tools eval (bench.eval_tools_corpus). Display-only:
    # never feeds SQS; quant-vs-quant comparison only (no --parse-special).
    ("mean_kld_tools", "Tools KLD"),
    ("same_top_p_tools", "Tools Top p"),
    ("prefill_tok_s", "Prefill tok/s"),
    ("decode_tok_s", "Decode tok/s"),
    ("ttft_2k_ms", "TTFT@2k (ms)"),
    # Tool-call columns (merged in from eval_toolcall.py output)
    ("tool_selection_acc", "Tool Sel %"),
    ("param_acc_mean", "Param Acc %"),
    ("schema_valid_rate", "Schema %"),
    ("rollout_complete_rate", "Rollout Done %"),
    ("sqs", "SQS"),
)

# Number of decimals per column key. Anything not listed renders as raw.
_FORMATS: dict[str, str] = {
    "size_gib": "{:.2f}",
    "bpw": "{:.3f}",
    "ppl": "{:.3f}",
    "ppl_ratio": "{:.3f}",
    "mean_kld": "{:.3f}",
    "median_kld": "{:.4f}",
    "same_top_p": "{:.2f}",
    "rms_dp": "{:.2f}",
    "mean_kld_tools": "{:.3f}",
    "same_top_p_tools": "{:.2f}",
    "sqs": "{:.3f}",
}
# Columns that combine value ± stdev into a single cell.
_SPEED_PAIRS: dict[str, tuple[str, str]] = {
    # display_key: (mean_field, stdev_field)
    "prefill_tok_s": ("prefill_tok_s", "prefill_stdev"),
    "decode_tok_s": ("decode_tok_s", "decode_stdev"),
    "ttft_2k_ms": ("ttft_2k_ms", "ttft_stdev_ms"),
}
# Columns that render as a percentage of [0, 1].
_PCT_COLUMNS = {
    "tool_selection_acc", "param_acc_mean", "schema_valid_rate", "rollout_complete_rate",
}

DEFAULT_WEIGHTS = (1.0, 2.0, 1.0)  # alpha, beta, gamma


@dataclass(frozen=True)
class F16Baseline:
    size_gib: float
    same_top_p: float
    decode_tok_s: float


def _to_float(s, default: float | None = None) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def find_f16_baseline(rows: list[dict]) -> F16Baseline | None:
    """Return the F16 reference row's metrics if present (matches ``fp16`` or ``/f16``)."""
    for r in rows:
        name = (r.get("model") or "").lower()
        if "fp16" in name or name.endswith("/f16"):
            size = _to_float(r.get("size_gib"))
            top_p = _to_float(r.get("same_top_p"))
            decode = _to_float(r.get("decode_tok_s"))
            if size is not None and top_p is not None and decode is not None:
                return F16Baseline(size_gib=size, same_top_p=top_p, decode_tok_s=decode)
    return None


def compute_sqs(
    row: dict,
    f16: F16Baseline,
    *,
    alpha: float = DEFAULT_WEIGHTS[0],
    beta: float = DEFAULT_WEIGHTS[1],
    gamma: float = DEFAULT_WEIGHTS[2],
) -> float | None:
    """Weighted geometric mean of compression × fidelity × speed.

    Any factor with missing or non-finite input defaults to 1.0; the score then
    reflects only the factors actually measured. Returns ``None`` if the weight
    total is non-positive (degenerate input).
    """
    size = _to_float(row.get("size_gib"))
    top_p = _to_float(row.get("same_top_p"))
    decode = _to_float(row.get("decode_tok_s"))

    comp = (f16.size_gib / size) if (size and size > 0) else 1.0
    fid = (top_p / 100.0) if top_p is not None else 1.0
    spd = (decode / f16.decode_tok_s) if (decode and decode > 0) else 1.0

    # Guard against pathological zeros.
    comp = max(comp, 1e-9)
    fid = max(fid, 1e-9)
    spd = max(spd, 1e-9)

    total_weight = alpha + beta + gamma
    if total_weight <= 0:
        return None
    product = (comp**alpha) * (fid**beta) * (spd**gamma)
    return product ** (1.0 / total_weight)


def load_results(csv_path: Path) -> list[dict]:
    """Load a bench-runner results.csv as a list of dicts."""
    with csv_path.open() as f:
        return list(csv.DictReader(f))


_TOOL_CALL_KEYS = (
    "tool_selection_acc", "param_acc_mean",
    "schema_valid_rate", "rollout_complete_rate",
)


def _detect_multirep(tc_rows: list[dict]) -> bool:
    """Multi-rep aggregated CSVs have ``<key>_mean`` columns; single-rep don't."""
    if not tc_rows:
        return False
    sample = tc_rows[0]
    return any(f"{k}_mean" in sample for k in _TOOL_CALL_KEYS)


def merge_toolcall(
    rows: list[dict], toolcall_csv: Path, *, latest_only: bool = True
) -> list[dict]:
    """Join tool-call accuracy columns into bench rows by basename(quant_path).

    Supports two CSV shapes:
      * **single-rep** — one row per ``(model, run)``; raw columns
        ``tool_selection_acc``, ``param_acc_mean``, … with a ``timestamp``.
      * **multi-rep aggregated** — one row per model; columns
        ``<key>_mean`` and ``<key>_stdev`` (output of ``run_toolcall_reps.py``).

    For multi-rep, the ``_mean`` value is stored under the bare key name (so the
    renderer can keep its existing column definitions) and the stdev is stored
    as ``<key>_stdev`` for the renderer to surface as "mean ± stdev".
    """
    if not toolcall_csv.exists():
        return rows
    with toolcall_csv.open() as f:
        tc_rows = list(csv.DictReader(f))
    multirep = _detect_multirep(tc_rows)

    by_filename: dict[str, dict] = {}
    for r in tc_rows:
        filename = r.get("model") or ""
        if not filename:
            continue
        if multirep or not latest_only:
            by_filename[filename] = r
        else:
            prev = by_filename.get(filename)
            if prev is None or (prev.get("timestamp") or "") < (r.get("timestamp") or ""):
                by_filename[filename] = r

    for r in rows:
        qp = r.get("quant_path") or ""
        tc = by_filename.get(os.path.basename(qp))
        if not tc:
            continue
        for k in _TOOL_CALL_KEYS:
            if multirep:
                mean_v = tc.get(f"{k}_mean")
                stdev_v = tc.get(f"{k}_stdev")
                if mean_v not in (None, ""):
                    r[k] = mean_v
                if stdev_v not in (None, ""):
                    r[f"{k}_stdev"] = stdev_v
            else:
                if tc.get(k) not in (None, ""):
                    r[k] = tc[k]
    return rows


SortOrder = Literal["asc", "desc"]


def sort_rows(
    rows: list[dict],
    by: str = "sqs",
    order: SortOrder | None = None,
) -> list[dict]:
    """Sort rows by a numeric column.

    Default order: ``desc`` for ``sqs`` (higher is better), ``asc`` otherwise
    (lower is better — appropriate for KLD, BPW, sizes). Rows with non-numeric
    values for the sort column sink to the bottom.
    """
    if order is None:
        order = "desc" if by == "sqs" else "asc"
    sentinel = float("-inf") if order == "desc" else float("inf")
    return sorted(
        rows,
        key=lambda r: _to_float(r.get(by), sentinel),
        reverse=(order == "desc"),
    )


def _format_cell(key: str, row: dict) -> str:
    """Format one table cell, with column-specific rules."""
    if key == "model":
        return row.get("model") or ""

    # Speed columns: value ± stdev.
    if key in _SPEED_PAIRS:
        mean_field, stdev_field = _SPEED_PAIRS[key]
        mean = _to_float(row.get(mean_field))
        stdev = _to_float(row.get(stdev_field))
        if mean is None:
            return "-"
        digits = 1 if "ttft" in key else 2
        cell = f"{mean:.{digits}f}"
        if stdev is not None:
            cell += f" ± {stdev:.{digits}f}"
        return cell

    if key in _PCT_COLUMNS:
        mean = _to_float(row.get(key))
        if mean is None:
            return "-"
        stdev = _to_float(row.get(f"{key}_stdev"))
        if stdev is None:
            return f"{mean * 100:.1f}"
        return f"{mean * 100:.1f} ± {stdev * 100:.1f}"

    v = _to_float(row.get(key))
    if v is None:
        return row.get(key) or "-"
    fmt = _FORMATS.get(key, "{}")
    return fmt.format(v)


def render_markdown(
    rows: list[dict],
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    sort_by: str = "sqs",
    order: SortOrder | None = None,
) -> str:
    """Render rows as a markdown table, with an HTML comment recording weights."""
    if order is None:
        order = "desc" if sort_by == "sqs" else "asc"
    alpha, beta, gamma = weights
    header_note = (
        f"<!-- SQS weights: alpha={alpha} beta={beta} gamma={gamma}; "
        f"sort={sort_by} {order} -->"
    )
    headers = [h for _, h in COLUMNS]
    lines = [
        header_note,
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        cells = [_format_cell(key, r) for key, _ in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def aggregate(
    results_csv: Path,
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    sort_by: str = "sqs",
    order: SortOrder | None = None,
    toolcall_csv: Path | None = None,
) -> str:
    """End-to-end: load a results.csv, optionally merge tool-call columns,
    compute SQS, sort, render markdown.

    Raises ``RuntimeError`` if no FP16 baseline row is present — SQS is only
    meaningful relative to a reference, so we'd rather fail loud than silently
    produce all-1.0 scores.
    """
    rows = load_results(results_csv)
    if toolcall_csv is not None:
        rows = merge_toolcall(rows, toolcall_csv)
    f16 = find_f16_baseline(rows)
    if f16 is None:
        raise RuntimeError(
            f"no F16 baseline row in {results_csv}; SQS requires a row whose "
            f"`model` column contains 'fp16' or ends with '/f16'"
        )
    alpha, beta, gamma = weights
    for r in rows:
        sqs = compute_sqs(r, f16, alpha=alpha, beta=beta, gamma=gamma)
        r["sqs"] = "" if sqs is None else sqs
    return render_markdown(sort_rows(rows, by=sort_by, order=order),
                           weights=weights, sort_by=sort_by, order=order)


__all__ = [
    "COLUMNS",
    "DEFAULT_WEIGHTS",
    "F16Baseline",
    "aggregate",
    "compute_sqs",
    "find_f16_baseline",
    "load_results",
    "merge_toolcall",
    "render_markdown",
    "sort_rows",
]

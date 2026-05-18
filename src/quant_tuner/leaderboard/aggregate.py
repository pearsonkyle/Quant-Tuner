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
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    ("prefill_tok_s", "Prefill tok/s"),
    ("decode_tok_s", "Decode tok/s"),
    ("ttft_2k_ms", "TTFT@2k (ms)"),
    ("sqs", "SQS"),
)

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
        cells: list[str] = []
        for key, _ in COLUMNS:
            v = r.get(key, "") or ""
            if key == "sqs":
                f = _to_float(v)
                v = f"{f:.3f}" if f is not None else ""
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def aggregate(
    results_csv: Path,
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    sort_by: str = "sqs",
    order: SortOrder | None = None,
) -> str:
    """End-to-end: load a results.csv, compute SQS, sort, render markdown.

    Raises ``RuntimeError`` if no FP16 baseline row is present — SQS is only
    meaningful relative to a reference, so we'd rather fail loud than silently
    produce all-1.0 scores.
    """
    rows = load_results(results_csv)
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
    "render_markdown",
    "sort_rows",
]

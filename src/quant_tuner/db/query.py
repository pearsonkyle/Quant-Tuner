"""Read-side helpers: aggregate reps (mean/stdev), top@k, and quant listing.

Reps are stored one row per trial; aggregation happens here on read so the same
underlying data can yield mean ± stdev, top@k, or any other reduction without a
schema change. The mean/stdev math reuses ``eval.reps.aggregate_metrics`` so DB
aggregates match what the CSV path produces byte-for-byte.
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from quant_tuner.db.models import MmluRep, Quant, ToolcallRep

# Per-benchmark: the rep model + the metric columns to aggregate over.
_MMLU_METRICS = (
    "accuracy",
    "n_total",
    "n_correct",
    "n_unparseable",
    "acc_biology", "acc_business", "acc_chemistry", "acc_computer_science",
    "acc_economics", "acc_engineering", "acc_health", "acc_history",
    "acc_law", "acc_math", "acc_other", "acc_philosophy",
    "acc_physics", "acc_psychology",
)
_TOOLCALL_METRICS = (
    "tool_selection_acc",
    "param_acc_mean",
    "schema_valid_rate",
    "continuation_type_match_rate",
    "recovery_appropriate_rate",
    "n_scored",
    "n_total",
    "n_post_result",
    "n_post_result_errors",
)

_BENCHMARKS: dict[str, tuple[type, tuple[str, ...]]] = {
    "mmlu": (MmluRep, _MMLU_METRICS),
    "toolcall": (ToolcallRep, _TOOLCALL_METRICS),
}


def _rep_model_and_metrics(benchmark: str) -> tuple[type, tuple[str, ...]]:
    try:
        return _BENCHMARKS[benchmark]
    except KeyError:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; expected one of {sorted(_BENCHMARKS)}"
        ) from None


def get_reps(session: Session, quant_id: int, benchmark: str) -> list[Any]:
    """All rep rows for a quant + benchmark, ordered by rep index."""
    rep_model, _ = _rep_model_and_metrics(benchmark)
    stmt: Any = (
        select(rep_model)
        .where(rep_model.quant_id == quant_id)  # type: ignore[attr-defined]
        .order_by(rep_model.rep)  # type: ignore[attr-defined]
    )
    return list(session.exec(stmt))


def aggregate_reps(
    session: Session, quant_id: int, benchmark: str
) -> dict[str, dict[str, float]]:
    """``{metric: {mean, stdev, n}}`` across reps, matching ``eval.reps``.

    Errored reps (``error`` set) and ``None`` metric values are skipped, exactly
    as ``eval.reps.aggregate_metrics`` does.
    """
    from quant_tuner.eval.reps import RepResult, aggregate_metrics

    rep_model, metric_cols = _rep_model_and_metrics(benchmark)
    rows = get_reps(session, quant_id, benchmark)

    rep_results: list[RepResult] = []
    for r in rows:
        metrics = {
            m: getattr(r, m)
            for m in metric_cols
            if getattr(r, m) is not None
        }
        rep_results.append(
            RepResult(rep=r.rep, seed=r.seed, metrics=metrics, error=r.error)
        )
    return aggregate_metrics(rep_results)


def top_k(
    session: Session,
    quant_id: int,
    benchmark: str,
    metric: str,
    *,
    k: int = 10,
    descending: bool = True,
) -> list[Any]:
    """The ``k`` best reps for ``quant_id`` ranked by ``metric`` (``top@k``).

    Reps that errored or have a ``None`` value for ``metric`` are excluded.
    """
    rep_model, metric_cols = _rep_model_and_metrics(benchmark)
    if metric not in metric_cols:
        raise ValueError(
            f"{metric!r} is not an aggregatable metric for {benchmark!r}: {metric_cols}"
        )
    rows = [
        r
        for r in get_reps(session, quant_id, benchmark)
        if r.error is None and getattr(r, metric) is not None
    ]
    rows.sort(key=lambda r: getattr(r, metric), reverse=descending)
    return rows[:k]


def list_quants(session: Session, **filters: Any) -> list[Quant]:
    """List quants, optionally filtered by exact-match natural-key fields.

    Recognized filters: ``model_repo``, ``quant_type``, ``technique``,
    ``variant``, ``calib_ctx``, ``dataset_id``. Ordered by the natural key so
    render-time pivots are stable.
    """
    allowed = {
        "model_repo", "quant_type", "technique",
        "variant", "calib_ctx", "dataset_id",
    }
    bad = set(filters) - allowed
    if bad:
        raise KeyError(f"unknown filter(s): {sorted(bad)}")

    stmt = select(Quant)
    for key, value in filters.items():
        stmt = stmt.where(getattr(Quant, key) == value)
    stmt = stmt.order_by(
        Quant.model_repo, Quant.quant_type, Quant.technique, Quant.variant
    )
    return list(session.exec(stmt))

"""Read the DB back out as the legacy CSV files the render scripts consume.

The DB is the source of truth; these exporters regenerate ``results.csv`` /
``kld_results.csv`` and the per-benchmark ``reps_results.csv`` / ``reps_aggregated.csv``
so existing ``render_exp*.py`` scripts keep working unchanged.

Two format details we must honor:

- The bench ``model`` column is a pipe-delimited label whose field order *varies
  per experiment* (exp-001 uses ``repo|technique|dataset``; exp-001-ext uses
  ``repo|dataset|quant_type``). ``label_format`` makes this configurable.
- The reps CSVs key ``model`` on the GGUF basename and only emit columns for the
  metrics actually recorded. We rebuild ``ModelReps`` from the DB and hand them to
  ``eval.reps.write_per_rep_csv`` / ``write_aggregated_csv`` so the layout matches
  byte-for-byte instead of being re-implemented here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlmodel import Session, select

from quant_tuner.bench.runner import CSV_COLUMNS
from quant_tuner.db.models import MmluRep, Quant, ToolcallRep
from quant_tuner.db.query import _MMLU_METRICS, _TOOLCALL_METRICS

# Map BenchRow/CSV column name -> Quant attribute (differs for the KLD fields).
_CSV_TO_QUANT = {
    "size_gib": "size_gib",
    "bpw": "bpw",
    "ppl": "ppl",
    "ppl_ratio": "ppl_ratio",
    "mean_kld": "kld_mean",
    "median_kld": "kld_median",
    "same_top_p": "same_top_p",
    "rms_dp": "rms_dp",
    "prefill_tok_s": "prefill_tok_s",
    "prefill_stdev": "prefill_stdev",
    "decode_tok_s": "decode_tok_s",
    "decode_stdev": "decode_stdev",
    "ttft_2k_ms": "ttft_2k_ms",
    "ttft_stdev_ms": "ttft_stdev_ms",
    "bench_repetitions": "bench_repetitions",
    "quant_path": "quant_path",
}

# Built-in label formats. A custom callable can be passed instead.
LabelFn = Callable[[Quant, str | None], str]


def _label_technique_dataset(q: Quant, dataset_name: str | None) -> str:
    """exp-001 style: ``repo|technique|dataset``."""
    return f"{q.model_repo}|{q.technique}|{dataset_name or ''}"


def _label_dataset_quant(q: Quant, dataset_name: str | None) -> str:
    """exp-001-ext style: ``repo|dataset|quant_type``."""
    return f"{q.model_repo}|{dataset_name or ''}|{q.quant_type}"


_LABEL_FORMATS: dict[str, LabelFn] = {
    "technique_dataset": _label_technique_dataset,
    "dataset_quant": _label_dataset_quant,
}


def _resolve_label_fn(label_format: str | LabelFn) -> LabelFn:
    if callable(label_format):
        return label_format
    try:
        return _LABEL_FORMATS[label_format]
    except KeyError:
        raise ValueError(
            f"unknown label_format {label_format!r}; "
            f"expected one of {sorted(_LABEL_FORMATS)} or a callable"
        ) from None


# ---------------------------------------------------------------------------
# results.csv / kld_results.csv
# ---------------------------------------------------------------------------


def export_results_csv(
    session: Session,
    out: Path,
    *,
    label_format: str | LabelFn = "dataset_quant",
    quants: list[Quant] | None = None,
) -> int:
    """Write a ``results.csv`` (the ``BenchRow`` / ``CSV_COLUMNS`` shape).

    Returns the number of rows written. ``label_format`` controls the pipe
    ``model`` label; default matches exp-001-ext (``repo|dataset|quant_type``).
    """
    import csv

    label_fn = _resolve_label_fn(label_format)
    if quants is None:
        quants = list(
            session.exec(
                select(Quant).order_by(
                    Quant.model_repo, Quant.quant_type, Quant.technique, Quant.variant
                )
            )
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for q in quants:
            dataset_name = q.dataset.name if q.dataset is not None else None
            row: dict[str, object] = {"model": label_fn(q, dataset_name)}
            for col, attr in _CSV_TO_QUANT.items():
                val = getattr(q, attr)
                row[col] = "" if val is None else val
            writer.writerow(row)
    return len(quants)


# ---------------------------------------------------------------------------
# reps CSVs (delegated to eval.reps writers for exact-format parity)
# ---------------------------------------------------------------------------


def _model_key(q: Quant) -> str:
    """The reps CSV ``model`` value: the GGUF basename (its join key)."""
    if q.quant_path:
        return Path(q.quant_path).name
    return f"{q.model_repo}-{q.quant_type}-{q.technique}-{q.variant}"


def _rebuild_model_reps(
    session: Session,
    benchmark: str,
    quants: list[Quant],
) -> tuple[list, dict]:
    """Reconstruct ``ModelReps`` (one per quant) + the sampling extra_cols.

    Returns ``(by_model, extra_cols)`` ready for ``eval.reps`` writers.
    """
    from quant_tuner.eval.reps import ModelReps, RepResult, aggregate_metrics

    rep_model: type
    metric_cols: tuple[str, ...]
    if benchmark == "mmlu":
        rep_model = MmluRep
        metric_cols = _MMLU_METRICS
    elif benchmark == "toolcall":
        rep_model = ToolcallRep
        metric_cols = _TOOLCALL_METRICS
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")

    by_model: list = []
    sampling_extra: dict[str, object] = {}
    for q in quants:
        rows: list = list(
            session.exec(
                select(rep_model)
                .where(rep_model.quant_id == q.id)  # type: ignore[attr-defined]
                .order_by(rep_model.rep)  # type: ignore[attr-defined,arg-type]
            )
        )
        if not rows:
            continue
        rep_results: list[RepResult] = []
        for r in rows:
            metrics: dict[str, float] = {}
            for m in metric_cols:
                v = getattr(r, m)
                if v is not None:
                    # MMLU subject columns are stored as acc_<subj> but the CSV
                    # expects <subj>_accuracy — translate on the way out.
                    key = m
                    if benchmark == "mmlu" and m.startswith("acc_"):
                        key = f"{m[len('acc_'):]}_accuracy"
                    metrics[key] = float(v)
            rep_results.append(
                RepResult(rep=r.rep, seed=r.seed, metrics=metrics, error=r.error)
            )
        # Sampling params are recorded identically across reps; grab from rep 0.
        first = rows[0]
        sampling_extra = {
            "temperature": _blank(first.temperature),
            "top_p": _blank(first.top_p),
            "top_k": _blank(first.top_k),
            "min_p": _blank(first.min_p),
            "presence_penalty": _blank(first.presence_penalty),
            "repetition_penalty": _blank(first.repetition_penalty),
        }
        mr = ModelReps(
            model=_model_key(q),
            n_reps=len(rep_results),
            reps=rep_results,
            aggregate=aggregate_metrics(rep_results),
        )
        by_model.append(mr)
    return by_model, sampling_extra


def _blank(v: object) -> object:
    return "" if v is None else v


def export_reps_csvs(
    session: Session,
    benchmark: str,
    *,
    per_rep: Path | None = None,
    aggregated: Path | None = None,
    quants: list[Quant] | None = None,
) -> int:
    """Write per-rep and/or aggregated reps CSVs for ``benchmark``.

    Delegates to ``eval.reps.write_per_rep_csv`` / ``write_aggregated_csv`` so the
    column layout is identical to the live eval path. Returns the model count.
    """
    from quant_tuner.eval.reps import write_aggregated_csv, write_per_rep_csv

    if quants is None:
        quants = list(
            session.exec(
                select(Quant).order_by(Quant.model_repo, Quant.quant_type)
            )
        )
    by_model, extra_cols = _rebuild_model_reps(session, benchmark, quants)
    if per_rep is not None:
        write_per_rep_csv(by_model, per_rep, extra_cols=extra_cols)
    if aggregated is not None:
        write_aggregated_csv(by_model, aggregated, extra_cols=extra_cols)
    return len(by_model)

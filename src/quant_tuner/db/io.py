"""Write-side adapters: turn pipeline/eval artifacts into DB rows.

These are the functions run-scripts call to populate the DB. They adapt the
existing in-code structures (``BenchRow`` from ``bench.runner``, ``ModelReps`` /
``RepResult`` from ``eval.reps``, ``Sampling`` from ``eval.toolcall``) so callers
don't reshape data by hand.

All upserts key on the quant's natural identity, so re-running a quant updates the
row in place instead of inserting a duplicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlmodel import Session, select

from quant_tuner.db.models import (
    MMLU_SUBJECTS,
    Dataset,
    MmluRep,
    Quant,
    ToolcallRep,
    _utcnow,
)

if TYPE_CHECKING:  # avoid importing heavy modules at DB-import time
    from quant_tuner.bench.runner import BenchRow
    from quant_tuner.eval.reps import ModelReps
    from quant_tuner.eval.toolcall import Sampling


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def get_or_create_dataset(
    session: Session,
    name: str | None,
    *,
    path: str | None = None,
    token_count: int | None = None,
) -> Dataset | None:
    """Fetch the dataset matching (name, path), creating it if absent.

    Returns ``None`` when ``name`` is falsy so a quant can carry no dataset
    (e.g. ``technique="none"`` runs).
    """
    if not name:
        return None
    stmt = select(Dataset).where(Dataset.name == name, Dataset.path == path)
    ds = session.exec(stmt).first()
    if ds is None:
        ds = Dataset(name=name, path=path, token_count=token_count)
        session.add(ds)
        session.flush()  # assign ds.id
    elif token_count is not None and ds.token_count != token_count:
        ds.token_count = token_count
        session.add(ds)
    return ds


# ---------------------------------------------------------------------------
# Quant
# ---------------------------------------------------------------------------

# Metric/path attributes upsert_quant accepts as kwargs and copies onto a Quant.
_QUANT_METRIC_FIELDS = frozenset(
    {
        "size_gib", "bpw", "n_params",
        "ppl", "ppl_ratio",
        "kld_mean", "kld_median", "kld_stdev", "kld_skew", "kld_kurtosis",
        "same_top_p", "rms_dp",
        "prefill_tok_s", "prefill_stdev", "decode_tok_s", "decode_stdev",
        "ttft_2k_ms", "ttft_stdev_ms", "bench_repetitions",
        "quant_path", "f16_path", "imatrix_path", "workspace",
        "calib_params_json", "notes",
    }
)


def upsert_quant(
    session: Session,
    *,
    model_repo: str,
    quant_type: str,
    technique: str = "none",
    variant: str = "default",
    calib_ctx: int | None = None,
    dataset: Dataset | None = None,
    **metrics: Any,
) -> Quant:
    """Get-or-create a quant on its natural key, then set any provided metrics.

    Unknown ``metrics`` keys raise ``KeyError`` so typos fail loudly rather than
    silently no-op.
    """
    bad = set(metrics) - _QUANT_METRIC_FIELDS
    if bad:
        raise KeyError(f"unknown Quant metric field(s): {sorted(bad)}")

    dataset_id = dataset.id if dataset is not None else None
    stmt = select(Quant).where(
        Quant.model_repo == model_repo,
        Quant.quant_type == quant_type,
        Quant.technique == technique,
        Quant.variant == variant,
        Quant.calib_ctx == calib_ctx,
        Quant.dataset_id == dataset_id,
    )
    quant = session.exec(stmt).first()
    if quant is None:
        quant = Quant(
            model_repo=model_repo,
            quant_type=quant_type,
            technique=technique,
            variant=variant,
            calib_ctx=calib_ctx,
            dataset_id=dataset_id,
        )

    for key, value in metrics.items():
        if value is not None:
            setattr(quant, key, value)
    quant.updated_at = _utcnow()
    session.add(quant)
    session.flush()  # assign quant.id for downstream rep inserts
    return quant


def quant_from_bench_row(
    session: Session,
    row: BenchRow,
    *,
    model_repo: str,
    quant_type: str,
    technique: str = "none",
    variant: str = "default",
    calib_ctx: int | None = None,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    dataset_tokens: int | None = None,
    n_params: int | None = None,
    f16_path: str | None = None,
    imatrix_path: str | None = None,
    workspace: str | None = None,
    calib_params_json: str | None = None,
) -> Quant:
    """Adapt a ``bench.runner.BenchRow`` into an upserted ``Quant`` row.

    The bench row carries metrics under slightly different names (``mean_kld`` ->
    ``kld_mean``); this maps them across. The identity fields the bench row's
    pipe-``model`` label encodes are passed explicitly by the caller so we never
    parse that label here.
    """
    dataset = get_or_create_dataset(
        session, dataset_name, path=dataset_path, token_count=dataset_tokens
    )
    return upsert_quant(
        session,
        model_repo=model_repo,
        quant_type=quant_type,
        technique=technique,
        variant=variant,
        calib_ctx=calib_ctx,
        dataset=dataset,
        size_gib=row.size_gib,
        bpw=row.bpw,
        n_params=n_params,
        ppl=row.ppl,
        ppl_ratio=row.ppl_ratio,
        kld_mean=row.mean_kld,
        kld_median=row.median_kld,
        same_top_p=row.same_top_p,
        rms_dp=row.rms_dp,
        prefill_tok_s=row.prefill_tok_s,
        prefill_stdev=row.prefill_stdev,
        decode_tok_s=row.decode_tok_s,
        decode_stdev=row.decode_stdev,
        ttft_2k_ms=row.ttft_2k_ms,
        ttft_stdev_ms=row.ttft_stdev_ms,
        bench_repetitions=row.bench_repetitions,
        quant_path=row.quant_path,
        f16_path=f16_path,
        imatrix_path=imatrix_path,
        workspace=workspace,
        calib_params_json=calib_params_json,
    )


# ---------------------------------------------------------------------------
# Reps
# ---------------------------------------------------------------------------


def _sampling_cols(sampling: Sampling | None) -> dict[str, Any]:
    if sampling is None:
        return {}
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "min_p": sampling.min_p,
        "presence_penalty": sampling.presence_penalty,
        "repetition_penalty": sampling.repetition_penalty,
    }


def _f(metrics: dict[str, Any], key: str) -> float | None:
    val = metrics.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _i(metrics: dict[str, Any], key: str) -> int | None:
    val = _f(metrics, key)
    return int(val) if val is not None else None


def add_mmlu_reps(
    session: Session,
    quant: Quant,
    model_reps: ModelReps,
    *,
    sampling: Sampling | None = None,
    replace: bool = True,
) -> list[MmluRep]:
    """Insert one ``MmluRep`` per rep in ``model_reps``.

    ``replace=True`` (default) clears any existing MMLU reps for this quant first,
    so re-running the benchmark refreshes rather than accumulates.
    """
    if replace:
        for existing in list(quant.mmlu_reps):
            session.delete(existing)
        session.flush()

    sampling_cols = _sampling_cols(sampling)
    rows: list[MmluRep] = []
    for rr in model_reps.reps:
        m = rr.metrics
        subj_cols = {
            f"acc_{subj}": _f(m, f"{subj}_accuracy") for subj in MMLU_SUBJECTS
        }
        row = MmluRep(
            quant_id=quant.id,  # type: ignore[arg-type]
            rep=rr.rep,
            seed=rr.seed,
            elapsed_sec=rr.elapsed_sec,
            error=rr.error,
            accuracy=_f(m, "accuracy"),
            n_total=_i(m, "n_total"),
            n_correct=_i(m, "n_correct"),
            n_unparseable=_i(m, "n_unparseable"),
            **subj_cols,
            **sampling_cols,
        )
        session.add(row)
        rows.append(row)
    session.flush()
    return rows


def add_toolcall_reps(
    session: Session,
    quant: Quant,
    model_reps: ModelReps,
    *,
    sampling: Sampling | None = None,
    replace: bool = True,
) -> list[ToolcallRep]:
    """Insert one ``ToolcallRep`` per rep in ``model_reps``."""
    if replace:
        for existing in list(quant.toolcall_reps):
            session.delete(existing)
        session.flush()

    sampling_cols = _sampling_cols(sampling)
    rows: list[ToolcallRep] = []
    for rr in model_reps.reps:
        m = rr.metrics
        row = ToolcallRep(
            quant_id=quant.id,  # type: ignore[arg-type]
            rep=rr.rep,
            seed=rr.seed,
            elapsed_sec=rr.elapsed_sec,
            error=rr.error,
            tool_selection_acc=_f(m, "tool_selection_acc"),
            param_acc_mean=_f(m, "param_acc_mean"),
            schema_valid_rate=_f(m, "schema_valid_rate"),
            continuation_type_match_rate=_f(m, "continuation_type_match_rate"),
            recovery_appropriate_rate=_f(m, "recovery_appropriate_rate"),
            n_scored=_i(m, "n_scored"),
            n_total=_i(m, "n_total"),
            n_post_result=_i(m, "n_post_result"),
            n_post_result_errors=_i(m, "n_post_result_errors"),
            **sampling_cols,
        )
        session.add(row)
        rows.append(row)
    session.flush()
    return rows

"""Generic multi-rep runner: N stochastic evals per model → mean ± stdev.

Any benchmark whose summary reduces to a ``dict[str, float]`` of metrics can
use this. Provide an ``eval_fn(base_url, sampling, rep) -> dict[str, float]``;
this module handles server lifecycle, per-rep seeding, aggregation, and CSV
output.

Example
-------
::

    from quant_tuner.eval.reps import run_reps_for_models, write_csvs
    from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval

    def toolcall_rep(base_url, sampling, rep):
        s = run_toolcall_eval(holdout, base_url=base_url, sampling=sampling)
        return {
            "tool_selection_acc": s.tool_selection_acc,
            "param_acc_mean": s.param_acc_mean,
            "schema_valid_rate": s.schema_valid_rate,
            "rollout_complete_rate": s.rollout_complete_rate,
        }

    by_model = run_reps_for_models(
        models=[Path("model.gguf")],
        eval_fn=toolcall_rep,
        reps=10,
        sampling=Sampling(temperature=0.6, top_p=0.95, top_k=20),
    )
    write_csvs(by_model, per_rep="reps.csv", aggregated="agg.csv")

The ``reps`` count is a plain int — change it freely. Seeds are derived as
``base_seed + rep_idx`` so the i-th rep is reproducible across runs.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from quant_tuner.eval.server import running_server
from quant_tuner.eval.toolcall import Sampling

# eval_fn(base_url, sampling_for_this_rep, rep_idx) -> {metric: scalar}
EvalFn = Callable[[str, Sampling, int], dict[str, float]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RepResult:
    """One rep's metrics for one model."""

    rep: int
    seed: int | None
    metrics: dict[str, float]
    elapsed_sec: float = 0.0
    error: str | None = None


@dataclass
class ModelReps:
    """All reps + the aggregate for one model."""

    model: str
    n_reps: int
    reps: list[RepResult]
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(reps: list[RepResult]) -> dict[str, dict[str, float]]:
    """Compute ``{metric: {mean, stdev, n}}`` across the rep results.

    Reps that errored out (``error`` set) contribute nothing — they're skipped
    per-metric. Metrics present in some reps but not others are aggregated
    over whichever reps reported them. With a single rep, ``stdev`` is 0.0.
    """
    by_metric: dict[str, list[float]] = {}
    for r in reps:
        if r.error is not None:
            continue
        for k, v in r.metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            by_metric.setdefault(k, []).append(fv)

    out: dict[str, dict[str, float]] = {}
    for metric, vals in by_metric.items():
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            stdev = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        else:
            stdev = 0.0
        out[metric] = {"mean": mean, "stdev": stdev, "n": n}
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _seed_for(base_seed: int, rep_idx: int) -> int:
    return base_seed + rep_idx


def run_reps_for_model(
    model: Path,
    eval_fn: EvalFn,
    *,
    reps: int,
    sampling: Sampling,
    base_seed: int = 1000,
    ctx: int = 16384,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 120.0,
    chat_template_kwargs: str | None = None,
    per_rep_callback: Callable[[RepResult], None] | None = None,
) -> ModelReps:
    """Spawn ``llama-server`` for ``model``, run ``eval_fn`` ``reps`` times.

    Each rep receives a fresh ``Sampling`` with ``seed = base_seed + rep_idx``.
    The server is torn down on exit (or on exception).
    """
    rep_results: list[RepResult] = []
    with running_server(
        model, ctx=ctx, ngl=ngl,
        log_path=server_log_path,
        startup_timeout=server_startup_timeout,
        chat_template_kwargs=chat_template_kwargs,
    ) as base_url:
        for rep_idx in range(reps):
            seed = _seed_for(base_seed, rep_idx)
            rep_sampling = replace(sampling, seed=seed)
            t0 = time.time()
            try:
                metrics = eval_fn(base_url, rep_sampling, rep_idx)
                rr = RepResult(
                    rep=rep_idx, seed=seed, metrics=metrics,
                    elapsed_sec=time.time() - t0,
                )
            except Exception as e:
                rr = RepResult(
                    rep=rep_idx, seed=seed, metrics={},
                    elapsed_sec=time.time() - t0, error=str(e),
                )
            rep_results.append(rr)
            if per_rep_callback is not None:
                per_rep_callback(rr)

    return ModelReps(
        model=model.name,
        n_reps=reps,
        reps=rep_results,
        aggregate=aggregate_metrics(rep_results),
    )


def run_reps_for_models(
    models: list[Path],
    eval_fn: EvalFn,
    *,
    reps: int,
    sampling: Sampling,
    base_seed: int = 1000,
    ctx: int = 16384,
    ngl: int = 99,
    log_dir: Path | None = None,
    server_startup_timeout: float = 120.0,
    chat_template_kwargs: str | None = None,
    per_rep_callback: Callable[[str, RepResult], None] | None = None,
    per_model_callback: Callable[[ModelReps], None] | None = None,
) -> list[ModelReps]:
    """Run ``run_reps_for_model`` over a list of GGUFs, one server at a time."""
    results: list[ModelReps] = []
    for model in models:
        if not model.exists():
            print(f"[skip] missing model: {model}", flush=True)
            continue

        server_log = (log_dir / f"server_reps_{model.stem}_{int(time.time())}.log"
                      if log_dir is not None else None)

        def _cb(rr: RepResult, _name=model.name) -> None:
            if per_rep_callback is not None:
                per_rep_callback(_name, rr)

        mr = run_reps_for_model(
            model, eval_fn,
            reps=reps, sampling=sampling, base_seed=base_seed,
            ctx=ctx, ngl=ngl,
            server_log_path=server_log,
            server_startup_timeout=server_startup_timeout,
            chat_template_kwargs=chat_template_kwargs,
            per_rep_callback=_cb,
        )
        results.append(mr)
        if per_model_callback is not None:
            per_model_callback(mr)
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _metric_names_in_order(by_model: list[ModelReps]) -> list[str]:
    """Stable union of metric names across all models, ordered by first appearance."""
    seen: dict[str, None] = {}
    for mr in by_model:
        for rr in mr.reps:
            for k in rr.metrics:
                if k not in seen:
                    seen[k] = None
    return list(seen)


def write_per_rep_csv(
    by_model: list[ModelReps],
    path: Path,
    *,
    extra_cols: dict[str, object] | None = None,
) -> None:
    """One row per ``(model, rep)`` with the full metric dict and ``extra_cols``."""
    extra_cols = extra_cols or {}
    metric_cols = _metric_names_in_order(by_model)
    fields = ["model", "rep", "seed", "elapsed_sec", *metric_cols, "error",
              *extra_cols.keys()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mr in by_model:
            for rr in mr.reps:
                row: dict = {
                    "model": mr.model,
                    "rep": rr.rep,
                    "seed": rr.seed if rr.seed is not None else "",
                    "elapsed_sec": f"{rr.elapsed_sec:.2f}",
                    "error": rr.error or "",
                    **extra_cols,
                }
                for m in metric_cols:
                    row[m] = rr.metrics.get(m, "")
                w.writerow(row)


def write_aggregated_csv(
    by_model: list[ModelReps],
    path: Path,
    *,
    extra_cols: dict[str, object] | None = None,
) -> None:
    """One row per model. Each metric expands to ``<m>_mean`` / ``<m>_stdev`` / ``<m>_n``."""
    extra_cols = extra_cols or {}
    metric_cols = _metric_names_in_order(by_model)
    fields = ["model", "n_reps"]
    for m in metric_cols:
        fields += [f"{m}_mean", f"{m}_stdev", f"{m}_n"]
    fields += list(extra_cols.keys())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mr in by_model:
            row: dict = {"model": mr.model, "n_reps": mr.n_reps, **extra_cols}
            for m in metric_cols:
                agg = mr.aggregate.get(m, {})
                row[f"{m}_mean"] = agg.get("mean", "")
                row[f"{m}_stdev"] = agg.get("stdev", "")
                row[f"{m}_n"] = agg.get("n", "")
            w.writerow(row)


def write_csvs(
    by_model: list[ModelReps],
    *,
    per_rep: Path,
    aggregated: Path,
    extra_cols: dict[str, object] | None = None,
) -> None:
    """Convenience: write both CSVs in one call."""
    write_per_rep_csv(by_model, per_rep, extra_cols=extra_cols)
    write_aggregated_csv(by_model, aggregated, extra_cols=extra_cols)


def sampling_extra_cols(sampling: Sampling) -> dict[str, object]:
    """Render a Sampling as the ``extra_cols`` mapping for traceability in CSVs."""
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p if sampling.top_p is not None else "",
        "top_k": sampling.top_k if sampling.top_k is not None else "",
        "min_p": sampling.min_p if sampling.min_p is not None else "",
        "presence_penalty": (sampling.presence_penalty
                             if sampling.presence_penalty is not None else ""),
        "repetition_penalty": (sampling.repetition_penalty
                               if sampling.repetition_penalty is not None else ""),
    }

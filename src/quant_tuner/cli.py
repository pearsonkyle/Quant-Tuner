"""quant-tuner CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_tuner.config import RunConfig
from quant_tuner.lens.cli import lens_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(lens_app, name="lens")


_PACKAGED_RECIPES = Path(__file__).resolve().parent / "recipes"


def _resolve_recipe(path: Path) -> Path:
    """Allow ``--recipe q4_k_m_imatrix`` (bare name) as well as a real path.

    Bare names resolve against the packaged ``src/quant_tuner/recipes/``.
    """
    if path.exists():
        return path
    candidate = _PACKAGED_RECIPES / path.name
    if candidate.exists():
        return candidate
    candidate = _PACKAGED_RECIPES / f"{path.name}.yaml"
    if candidate.exists():
        return candidate
    raise typer.BadParameter(f"recipe not found: {path}")


def _load_recipe(
    recipe: Path,
    model: str | None,
    logs: Path | None,
    workspace: Path | None,
) -> RunConfig:
    cfg = RunConfig.from_yaml(_resolve_recipe(recipe))
    # CLI overrides — recipes ship with `PLACEHOLDER` for required fields.
    if model is not None:
        cfg.model = model
    if logs is not None:
        cfg.data.logs = logs
    if workspace is not None:
        cfg.workspace = workspace
    if cfg.model == "PLACEHOLDER":
        raise typer.BadParameter(
            "recipe leaves `model` as PLACEHOLDER; pass --model to override."
        )
    if cfg.calibration.method != "none" and (
        cfg.data.logs is None or str(cfg.data.logs) == "PLACEHOLDER"
    ):
        raise typer.BadParameter(
            "recipe leaves `data.logs` as PLACEHOLDER; pass --logs to override."
        )
    return cfg


@app.command()
def run(
    recipe: Path = typer.Option(..., help="Recipe YAML path or packaged recipe name"),
    model: str | None = typer.Option(None, help="HF repo id or local path (overrides recipe)"),
    logs: Path | None = typer.Option(None, help="Usage-log JSONL (overrides recipe)"),
    workspace: Path | None = typer.Option(None, help="Output dir (overrides recipe)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate the recipe and print "
                                  "the resolved config without executing."),
) -> None:
    """End-to-end: extract → calibrate → quantize → bench, driven by a recipe."""
    from quant_tuner.pipeline import run_pipeline

    cfg = _load_recipe(recipe, model, logs, workspace)
    if dry_run:
        typer.echo(cfg.model_dump_json(indent=2))
        return
    run_pipeline(cfg)


@app.command()
def bench(
    quant: Path = typer.Option(..., help="Quantized GGUF to evaluate"),
    reference: Path = typer.Option(..., help="Reference GGUF (typically the F16) for KLD"),
    eval_dataset: Path = typer.Option(..., "--eval", help="Held-out text corpus for KLD/PPL"),
    eval_tools: Path | None = typer.Option(
        None, "--eval-tools",
        help="Optional in-distribution tools corpus (corpus.eval.tools.txt) for a "
             "second KLD pass -> *_tools columns (quant-vs-quant only)"),
    out: Path = typer.Option(Path("results.csv"), help="CSV to append the bench row to"),
    label: str | None = typer.Option(None, help="Label for the row (default: quant filename)"),
    suite: str = typer.Option("full", help="quick | kld | speed | full | leaderboard"),
    eval_ctx: int = typer.Option(8192, help="Context length for KLD eval"),
) -> None:
    """Bench an already-quantized GGUF against a reference. No recipe needed."""
    from quant_tuner.bench import bpw as bpw_mod
    from quant_tuner.bench import kld as kld_mod
    from quant_tuner.bench import runner

    if not quant.exists():
        raise typer.BadParameter(f"quant not found: {quant}")
    if not reference.exists():
        raise typer.BadParameter(f"reference not found: {reference}")
    if not eval_dataset.exists():
        raise typer.BadParameter(f"eval dataset not found: {eval_dataset}")

    baseline_path = out.parent / "baseline.kld"
    if not baseline_path.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        kld_mod.build_baseline(reference, eval_dataset, baseline_path, ctx=eval_ctx)

    tools_baseline: Path | None = None
    if eval_tools is not None:
        if not eval_tools.exists():
            raise typer.BadParameter(f"tools eval dataset not found: {eval_tools}")
        tools_baseline = out.parent / "baseline-tools.kld"
        if not tools_baseline.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            kld_mod.build_baseline(reference, eval_tools, tools_baseline, ctx=eval_ctx)

    row = runner.bench_one(
        quant, label or quant.stem,
        reference_n_params=bpw_mod.n_params(reference),
        eval_dataset=eval_dataset,
        eval_baseline=baseline_path,
        tools_dataset=eval_tools,
        tools_baseline=tools_baseline,
        eval_ctx=eval_ctx,
        log_dir=out.parent / "logs",
        suite=suite,
    )
    runner.append_row(out, row)
    typer.echo(
        f"bpw={row.bpw:.3f}  mean_kld={row.mean_kld}  same_top_p={row.same_top_p}  "
        f"decode={row.decode_tok_s}  -> {out}"
    )


@app.command()
def leaderboard(
    results: Path = typer.Option(..., help="results.csv to aggregate"),
    out: Path = typer.Option(Path("LEADERBOARD.md"), help="markdown output path"),
    weights: str = typer.Option(
        "1,2,1", help="SQS weights alpha,beta,gamma (compression, fidelity, speed)"
    ),
    sort_by: str = typer.Option("sqs", "--sort", help="column to sort by"),
    toolcall_csv: Path | None = typer.Option(None, help="Optional tool-call CSV to merge"),
    lens_csv: Path | None = typer.Option(None, help="Optional Jacobian-lens sidecar CSV to merge"),
) -> None:
    """Aggregate results.csv into a markdown leaderboard with SQS scores."""
    from quant_tuner.leaderboard.aggregate import aggregate

    parts = [float(x) for x in weights.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("--weights expects three comma-separated numbers")
    a, b, c = parts

    markdown = aggregate(results, weights=(a, b, c), sort_by=sort_by,
                         toolcall_csv=toolcall_csv, lens_csv=lens_csv)
    out.write_text(markdown)
    typer.echo(f"wrote {out}")


db_app = typer.Typer(no_args_is_help=True, add_completion=False,
                     help="Quant-tracking database (SQLite).")
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init(
    db: Path | None = typer.Option(None, help="DB path (default: out/quant_tuner.db "
                                   "or $QUANT_TUNER_DB)"),
) -> None:
    """Create the database file and tables if they don't exist."""
    from quant_tuner.db import db_path, get_engine, init_db

    engine = get_engine(db)
    init_db(engine)
    typer.echo(f"initialized {db_path(db)}")


@db_app.command("import-csv")
def db_import_csv(
    results: Path | None = typer.Option(None, help="bench results.csv / kld_results.csv"),
    reps_agg: Path | None = typer.Option(None, "--reps-agg",
                                         help="aggregated reps CSV to merge"),
    gguf_map: Path | None = typer.Option(None, "--gguf-map",
                                         help="gguf_map.json to disambiguate reps "
                                         "basenames across repos"),
    benchmark: str = typer.Option("mmlu", help="benchmark for --reps-agg: mmlu | toolcall"),
    label_format: str = typer.Option("dataset_quant",
                                     help="results label layout: dataset_quant | "
                                     "technique_dataset"),
    db: Path | None = typer.Option(None, help="DB path override"),
) -> None:
    """Backfill the DB from existing CSV files."""
    from quant_tuner.db import get_engine, init_db, session_scope
    from quant_tuner.db.imports import (
        import_reps_aggregated_csv,
        import_results_csv,
    )

    engine = get_engine(db)
    init_db(engine)
    with session_scope(engine) as session:
        if results is not None:
            n = import_results_csv(session, results, label_format=label_format)
            typer.echo(f"imported {n} quant rows from {results}")
        if reps_agg is not None:
            m = import_reps_aggregated_csv(
                session, reps_agg, benchmark, gguf_map=gguf_map
            )
            typer.echo(f"merged {benchmark} reps for {m} quants from {reps_agg}")


@db_app.command("export-csv")
def db_export_csv(
    out: Path = typer.Option(..., help="output directory for the CSV files"),
    label_format: str = typer.Option("dataset_quant",
                                     help="results label layout"),
    db: Path | None = typer.Option(None, help="DB path override"),
) -> None:
    """Regenerate results.csv + reps CSVs from the DB (for render_exp* scripts)."""
    from quant_tuner.db import get_engine, init_db, session_scope
    from quant_tuner.db.export import export_reps_csvs, export_results_csv

    engine = get_engine(db)
    init_db(engine)
    out.mkdir(parents=True, exist_ok=True)
    with session_scope(engine) as session:
        nq = export_results_csv(session, out / "results.csv",
                                label_format=label_format)
        typer.echo(f"wrote {out / 'results.csv'} ({nq} quants)")
        for bench in ("mmlu", "toolcall"):
            nm = export_reps_csvs(
                session, bench,
                per_rep=out / f"{bench}_reps_results.csv",
                aggregated=out / f"{bench}_reps_aggregated.csv",
            )
            if nm:
                typer.echo(f"wrote {bench} reps CSVs ({nm} quants)")


@db_app.command("query")
def db_query(
    metric: str = typer.Option(..., help="e.g. mmlu.accuracy or toolcall.tool_selection_acc"),
    top: int = typer.Option(10, help="show the top-N quants by the metric mean"),
    model_repo: str | None = typer.Option(None, help="filter by HF repo"),
    db: Path | None = typer.Option(None, help="DB path override"),
) -> None:
    """Rank quants by a benchmark metric's mean across reps."""
    from quant_tuner.db import get_engine, session_scope
    from quant_tuner.db.query import aggregate_reps, list_quants

    if "." not in metric:
        raise typer.BadParameter("metric must be 'benchmark.name' (e.g. mmlu.accuracy)")
    benchmark, metric_name = metric.split(".", 1)

    engine = get_engine(db)
    filters = {"model_repo": model_repo} if model_repo else {}
    rows: list[tuple[str, float, float, int]] = []
    with session_scope(engine) as session:
        for q in list_quants(session, **filters):
            if q.id is None:
                continue
            agg = aggregate_reps(session, q.id, benchmark).get(metric_name)
            if agg is None:
                continue
            ds = q.dataset.name if q.dataset is not None else "-"
            label = f"{q.model_repo} {q.quant_type} {q.technique}/{q.variant} [{ds}]"
            rows.append((label, agg["mean"], agg["stdev"], int(agg["n"])))

    rows.sort(key=lambda x: x[1], reverse=True)
    if not rows:
        typer.echo(f"no quants with {metric}")
        return
    typer.echo(f"top {top} by {metric} (mean ± stdev, n):")
    for label, mean, stdev, n in rows[:top]:
        typer.echo(f"  {mean:.4f} ± {stdev:.4f} (n={n})  {label}")


if __name__ == "__main__":
    app()

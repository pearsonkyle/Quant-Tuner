"""quant-tuner CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_tuner.config import RunConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)


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

    row = runner.bench_one(
        quant, label or quant.stem,
        reference_n_params=bpw_mod.n_params(reference),
        eval_dataset=eval_dataset,
        eval_baseline=baseline_path,
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
) -> None:
    """Aggregate results.csv into a markdown leaderboard with SQS scores."""
    from quant_tuner.leaderboard.aggregate import aggregate

    parts = [float(x) for x in weights.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("--weights expects three comma-separated numbers")
    a, b, c = parts

    markdown = aggregate(results, weights=(a, b, c), sort_by=sort_by, toolcall_csv=toolcall_csv)
    out.write_text(markdown)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()

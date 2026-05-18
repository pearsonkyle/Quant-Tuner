"""quant-tuner CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(help="Corpus prep: ingest logs, split, build holdouts.")
calibrate_app = typer.Typer(help="Run a calibration method.")
app.add_typer(data_app, name="data")
app.add_typer(calibrate_app, name="calibrate")


@app.command()
def run(
    model: str = typer.Option(..., help="HF repo id or local path"),
    logs: Path = typer.Option(..., help="JSONL log file"),
    method: str = typer.Option("imatrix", help="imatrix | awq | gptq | none"),
    quant: str = typer.Option("Q4_K_M", help="llama-quantize type tag"),
    out: Path = typer.Option(Path("./out"), help="Workspace directory"),
) -> None:
    """End-to-end: ingest → calibrate → quantize → bench."""
    typer.echo(f"[stub] run model={model} method={method} quant={quant} out={out}")


@app.command()
def bench(
    model: Path = typer.Option(..., help="Quantized GGUF to evaluate"),
    reference: Path | None = typer.Option(None, help="Reference GGUF (e.g. fp16) for KLD"),
    suite: str = typer.Option("full"),
) -> None:
    """Run benchmark suite against a quantized model."""
    typer.echo(f"[stub] bench model={model} reference={reference} suite={suite}")


@app.command()
def leaderboard(
    results: Path = typer.Option(..., help="results.csv to aggregate"),
    out: Path = typer.Option(Path("LEADERBOARD.md"), help="markdown output path"),
    weights: str = typer.Option(
        "1,2,1", help="SQS weights alpha,beta,gamma (compression, fidelity, speed)"
    ),
    sort_by: str = typer.Option("sqs", "--sort", help="column to sort by"),
) -> None:
    """Aggregate results.csv into a markdown leaderboard with SQS scores."""
    from quant_tuner.leaderboard.aggregate import aggregate

    parts = [float(x) for x in weights.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("--weights expects three comma-separated numbers")
    a, b, c = parts

    markdown = aggregate(results, weights=(a, b, c), sort_by=sort_by)
    out.write_text(markdown)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()

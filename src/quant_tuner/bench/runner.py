"""Single-model bench orchestrator: BPW + KLD + llama-bench → one CSV row."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, speed


@dataclass
class BenchRow:
    model: str
    size_gib: float
    bpw: float
    ppl: float | None
    ppl_ratio: float | None
    mean_kld: float | None
    median_kld: float | None
    same_top_p: float | None
    rms_dp: float | None
    # In-distribution tools eval (bench.eval_tools_corpus): quant-vs-quant
    # only — llama-perplexity can't --parse-special the chat markers.
    ppl_tools: float | None
    mean_kld_tools: float | None
    same_top_p_tools: float | None
    prefill_tok_s: float | None
    prefill_stdev: float | None
    decode_tok_s: float | None
    decode_stdev: float | None
    ttft_2k_ms: float | None
    ttft_stdev_ms: float | None
    bench_repetitions: int | None
    quant_path: str


CSV_COLUMNS = [
    "model", "size_gib", "bpw",
    "ppl", "ppl_ratio", "mean_kld", "median_kld", "same_top_p", "rms_dp",
    "ppl_tools", "mean_kld_tools", "same_top_p_tools",
    "prefill_tok_s", "prefill_stdev",
    "decode_tok_s", "decode_stdev",
    "ttft_2k_ms", "ttft_stdev_ms",
    "bench_repetitions",
    "quant_path",
]


def bench_one(
    quant_path: Path,
    label: str,
    *,
    reference_n_params: int,
    eval_dataset: Path | None = None,
    eval_baseline: Path | None = None,
    tools_dataset: Path | None = None,
    tools_baseline: Path | None = None,
    eval_ctx: int = 8192,
    n_tokens: int | None = None,
    log_dir: Path | None = None,
    suite: str = "full",
    bench_repetitions: int = 10,
) -> BenchRow:
    """Run KLD + speed for one quantized model and return a row.

    suite:
      - "quick":      bpw only
      - "kld":        bpw + kld (requires eval_dataset + eval_baseline)
      - "speed":      bpw + llama-bench (`bench_repetitions` runs of prefill+decode)
      - "full":       all of the above
      - "leaderboard": alias for full

    ``tools_dataset`` + ``tools_baseline`` (both or neither) add a second,
    in-distribution KLD pass whose results land in the ``*_tools`` columns.

    ``bench_repetitions`` controls how many times llama-bench repeats each
    timing measurement; with the default of 10, we get prefill/decode/TTFT
    mean ± sample stdev per row. Set to 1 to skip variance estimation.
    """
    kld_metrics = kld.KLDMetrics()
    tools_metrics = kld.KLDMetrics()
    speed_metrics = speed.SpeedMetrics()
    log_dir = log_dir or quant_path.parent / "logs"

    if suite in ("kld", "full", "leaderboard"):
        if eval_dataset is None or eval_baseline is None:
            raise ValueError("suite requires both eval_dataset and eval_baseline")
        kld_metrics = kld.evaluate(
            quant_path, eval_dataset, eval_baseline,
            ctx=eval_ctx, n_tokens=n_tokens,
            log=log_dir / f"{label}.kld.log",
        )
        if (tools_dataset is None) != (tools_baseline is None):
            raise ValueError("tools_dataset and tools_baseline must be set together")
        if tools_dataset is not None and tools_baseline is not None:
            tools_metrics = kld.evaluate(
                quant_path, tools_dataset, tools_baseline,
                ctx=eval_ctx, n_tokens=n_tokens,
                log=log_dir / f"{label}.kld-tools.log",
            )

    if suite in ("speed", "full", "leaderboard"):
        speed_metrics = speed.evaluate(
            quant_path,
            repetitions=bench_repetitions,
            log=log_dir / f"{label}.bench.log",
        )

    return BenchRow(
        model=label,
        size_gib=bpw_mod.size_gib(quant_path),
        bpw=bpw_mod.bpw(quant_path, reference_n_params),
        ppl=kld_metrics.ppl,
        ppl_ratio=kld_metrics.ppl_ratio,
        mean_kld=kld_metrics.mean_kld,
        median_kld=kld_metrics.median_kld,
        same_top_p=kld_metrics.same_top_p,
        rms_dp=kld_metrics.rms_dp,
        ppl_tools=tools_metrics.ppl,
        mean_kld_tools=tools_metrics.mean_kld,
        same_top_p_tools=tools_metrics.same_top_p,
        prefill_tok_s=speed_metrics.prefill_tok_s,
        prefill_stdev=speed_metrics.prefill_stdev,
        decode_tok_s=speed_metrics.decode_tok_s,
        decode_stdev=speed_metrics.decode_stdev,
        ttft_2k_ms=speed_metrics.ttft_2k_ms,
        ttft_stdev_ms=speed_metrics.ttft_stdev_ms,
        bench_repetitions=speed_metrics.n_repetitions,
        quant_path=str(quant_path),
    )


def append_row(csv_path: Path, row: BenchRow) -> None:
    """Append a row, dropping any prior row with the same `model` label.

    Preserves any extra columns present in the existing CSV (e.g. tool-call
    metrics merged in by ``leaderboard.aggregate``) — we read with DictReader
    and write the superset back out, sorted with our known columns first.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    existing_fields: list[str] = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing = [r for r in reader if r["model"] != row.model]

    fieldnames = list(CSV_COLUMNS)
    for f in existing_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        out = {k: ("" if v is None else v) for k, v in asdict(row).items()}
        writer.writerow({k: out.get(k, "") for k in fieldnames})

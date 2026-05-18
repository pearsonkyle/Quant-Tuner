"""Single-model bench orchestrator. Python port of experiments/_shared/bench_one.sh."""

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
    prefill_tok_s: float | None
    decode_tok_s: float | None
    ttft_2k_ms: float | None
    quant_path: str


CSV_COLUMNS = [
    "model", "size_gib", "bpw",
    "ppl", "ppl_ratio", "mean_kld", "median_kld", "same_top_p", "rms_dp",
    "prefill_tok_s", "decode_tok_s", "ttft_2k_ms",
    "quant_path",
]


def bench_one(
    quant_path: Path,
    label: str,
    *,
    reference_n_params: int,
    eval_dataset: Path | None = None,
    eval_baseline: Path | None = None,
    eval_ctx: int = 8192,
    n_tokens: int | None = None,
    log_dir: Path | None = None,
    suite: str = "full",
) -> BenchRow:
    """Run KLD + speed for one quantized model and return a row.

    suite:
      - "quick":      bpw only
      - "kld":        bpw + kld (requires eval_dataset + eval_baseline)
      - "speed":      bpw + llama-bench
      - "full":       all of the above
      - "leaderboard": alias for full
    """
    kld_metrics = kld.KLDMetrics()
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

    if suite in ("speed", "full", "leaderboard"):
        speed_metrics = speed.evaluate(quant_path, log=log_dir / f"{label}.bench.log")

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
        prefill_tok_s=speed_metrics.prefill_tok_s,
        decode_tok_s=speed_metrics.decode_tok_s,
        ttft_2k_ms=speed_metrics.ttft_2k_ms,
        quant_path=str(quant_path),
    )


def append_row(csv_path: Path, row: BenchRow) -> None:
    """Append a row, dropping any prior row with the same `model` label."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if csv_path.exists():
        with open(csv_path) as f:
            existing = [r for r in csv.DictReader(f) if r["model"] != row.model]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
        writer.writerow({k: ("" if v is None else v) for k, v in asdict(row).items()})

"""Parse llama-bench JSON output for prefill/decode tok/s and TTFT.

llama-bench natively repeats each measurement and reports both the mean
(``avg_ts``) and the per-run samples (``samples_ts``); we surface both so
callers can report mean ± stdev. The default repetition count is bumped to 10
to give a tight enough confidence interval that small (≲ 2 %) speed
differences between calibrated quants are distinguishable from noise.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from quant_tuner.models import llama_cpp


@dataclass
class SpeedMetrics:
    prefill_tok_s: float | None = None
    prefill_stdev: float | None = None
    decode_tok_s: float | None = None
    decode_stdev: float | None = None
    ttft_2k_ms: float | None = None
    ttft_stdev_ms: float | None = None
    n_repetitions: int | None = None


def _stdev(samples: list[float]) -> float | None:
    if not samples or len(samples) < 2:
        return None
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    return math.sqrt(var)


def parse_bench_log(text: str, n_prompt: int = 2048, n_gen: int = 128) -> SpeedMetrics:
    """Parse llama-bench JSON. Returns ``SpeedMetrics(..._stdev=None)`` if the run
    only had one repetition (in which case llama-bench omits ``stddev_ts``)."""
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not match:
        return SpeedMetrics()
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError:
        return SpeedMetrics()

    metrics = SpeedMetrics()
    for r in rows:
        avg = r.get("avg_ts")
        samples = r.get("samples_ts") or []
        # Prefer the JSON-reported stdev; fall back to computing from samples
        # if llama-bench didn't include it (older builds).
        stdev = r.get("stddev_ts")
        if stdev is None and samples:
            stdev = _stdev([float(s) for s in samples])
        if r.get("n_prompt") == n_prompt and r.get("n_gen") == 0:
            metrics.prefill_tok_s = avg
            metrics.prefill_stdev = stdev
            metrics.n_repetitions = len(samples) or metrics.n_repetitions
        elif r.get("n_prompt") == 0 and r.get("n_gen") == n_gen:
            metrics.decode_tok_s = avg
            metrics.decode_stdev = stdev

    # TTFT = n_prompt / prefill_tok_s · 1000 (ms). Relative error in TTFT
    # equals relative error in prefill (reciprocal preserves it).
    if metrics.prefill_tok_s:
        metrics.ttft_2k_ms = n_prompt / metrics.prefill_tok_s * 1000.0
        if metrics.prefill_stdev:
            metrics.ttft_stdev_ms = metrics.ttft_2k_ms * (
                metrics.prefill_stdev / metrics.prefill_tok_s
            )
    return metrics


def evaluate(
    model: Path,
    n_prompt: int = 2048,
    n_gen: int = 128,
    repetitions: int = 10,
    log: Path | None = None,
) -> SpeedMetrics:
    """Run llama-bench and parse the result. ``repetitions`` ≥ 2 to get stdev."""
    output = llama_cpp.bench(
        model, n_prompt=n_prompt, n_gen=n_gen, repetitions=repetitions, log=log
    )
    return parse_bench_log(output, n_prompt=n_prompt, n_gen=n_gen)

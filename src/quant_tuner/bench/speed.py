"""Parse llama-bench JSON output for prefill/decode tok/s and TTFT."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from quant_tuner.models import llama_cpp


@dataclass
class SpeedMetrics:
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    ttft_2k_ms: float | None = None


def parse_bench_log(text: str, n_prompt: int = 2048, n_gen: int = 128) -> SpeedMetrics:
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not match:
        return SpeedMetrics()
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError:
        return SpeedMetrics()

    prefill = decode = None
    for r in rows:
        if r.get("n_prompt") == n_prompt and r.get("n_gen") == 0:
            prefill = r.get("avg_ts")
        elif r.get("n_prompt") == 0 and r.get("n_gen") == n_gen:
            decode = r.get("avg_ts")

    ttft = None
    if prefill:
        ttft = n_prompt / prefill * 1000.0

    return SpeedMetrics(prefill_tok_s=prefill, decode_tok_s=decode, ttft_2k_ms=ttft)


def evaluate(
    model: Path,
    n_prompt: int = 2048,
    n_gen: int = 128,
    log: Path | None = None,
) -> SpeedMetrics:
    output = llama_cpp.bench(model, n_prompt=n_prompt, n_gen=n_gen, log=log)
    return parse_bench_log(output, n_prompt=n_prompt, n_gen=n_gen)

"""Parse llama-perplexity --kl-divergence output and produce a KLD baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from quant_tuner.models import llama_cpp


@dataclass
class KLDMetrics:
    """Subset of the llama-perplexity KLD report most relevant for quant comparison."""

    ppl: float | None = None
    ppl_ratio: float | None = None
    mean_kld: float | None = None
    median_kld: float | None = None
    same_top_p: float | None = None
    rms_dp: float | None = None


def _parse_one(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, re.MULTILINE)
    if not matches:
        return None
    raw = matches[-1].strip().split()[0]
    try:
        return float(raw)
    except ValueError:
        return None


def parse_kld_log(text: str) -> KLDMetrics:
    return KLDMetrics(
        ppl=_parse_one(text, r"^Mean PPL\(Q\)\s+:\s*(.*)$"),
        ppl_ratio=_parse_one(text, r"^Mean PPL\(Q\)/PPL\(base\)\s+:\s*(.*)$"),
        mean_kld=_parse_one(text, r"^Mean\s+KLD:\s*(.*)$"),
        median_kld=_parse_one(text, r"^Median\s+KLD:\s*(.*)$"),
        same_top_p=_parse_one(text, r"^Same top p:\s*(.*)$"),
        rms_dp=_parse_one(text, r"^RMS Δp\s*:\s*(.*)$"),
    )


def build_baseline(
    reference: Path,
    dataset: Path,
    baseline_out: Path,
    ctx: int = 8192,
    n_tokens: int | None = None,
    log: Path | None = None,
) -> Path:
    """Generate a KLD baseline (`.kld`) from a reference model + eval dataset."""
    return llama_cpp.perplexity_baseline(
        reference, dataset, baseline_out, ctx=ctx, n_tokens=n_tokens, log=log
    )


def evaluate(
    model: Path,
    dataset: Path,
    baseline: Path,
    ctx: int = 8192,
    n_tokens: int | None = None,
    log: Path | None = None,
) -> KLDMetrics:
    output = llama_cpp.perplexity_kld(
        model, dataset, baseline, ctx=ctx, n_tokens=n_tokens, log=log
    )
    return parse_kld_log(output)

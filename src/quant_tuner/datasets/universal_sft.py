"""Record builder for the universal SFT mixture.

Reads the ``sft.jsonl.gz`` that :mod:`quant_tuner.data.universal` writes beside the
calibration corpora and republishes it as a versioned dataset — full conversations, one
schema, split-tagged, with every source labelled.

    <build-dir>/sft.jsonl.gz   ->  train / holdout / test splits

Why this exists as a dataset rather than only as a build artifact: the file is the exact
input to ``qat.corpus.build_sft_corpus``, so pinning it by version and sha256 is what makes
a fine-tune reproducible after the log corpora or the published datasets move underneath it.

**This dataset is private-only** (``DatasetSpec.private_only``). It carries the CLI usage
logs and agent trajectories, which are ~91% of it by characters and which
``datasets/agent-logs/README.md`` states plainly are real captured usage and *not ours to
publish*. The registry flag makes ``push`` refuse a public upload rather than relying on
whoever runs it to remember ``--private``.

Rows are passed through unchanged apart from dropping the per-conversation ``system_scrub``
bookkeeping, which describes the build rather than the data.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Newest build wins; override with QT_SFT_JSONL for a specific one.
DEFAULT_BUILD_DIRS = [
    REPO / "out" / "corpora" / "qwen3-universal-v2",
    REPO / "out" / "corpora" / "qwen3-universal",
]

SPLITS = ("train", "holdout", "test")

# Build bookkeeping, not data: how many system-prompt blocks the scrubber dropped from this
# conversation. Useful in the audit, noise in a training row.
_DROP_FIELDS = ("system_scrub",)


def resolve_sft_path(path: Path | None = None) -> Path:
    """The ``sft.jsonl.gz`` to publish: explicit, ``$QT_SFT_JSONL``, or the newest build."""
    for cand in (path, os.environ.get("QT_SFT_JSONL")):
        if cand:
            p = Path(cand)
            if not p.exists():
                raise FileNotFoundError(f"sft export not found: {p}")
            return p
    for d in DEFAULT_BUILD_DIRS:
        if (p := d / "sft.jsonl.gz").exists():
            return p
    raise FileNotFoundError(
        "no sft.jsonl.gz found. Build one first:\n"
        "  PYTHONPATH=src python scripts/build_universal_corpus.py "
        "--out out/corpora/qwen3-universal-v2 --model <hf-model-dir>\n"
        "or point QT_SFT_JSONL at an existing export."
    )


def read_sft(path: Path | None = None) -> list[dict]:
    p = resolve_sft_path(path)
    with gzip.open(p, "rt") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def iter_sft_records(split: str, path: Path | None = None) -> Iterator[dict]:
    """Rows of one split, in file order.

    ``split`` matches the calibration corpus's own assignment, so training on ``train``
    leaves the tools / agentic / refusal / breadth eval holdouts genuinely held out.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    for rec in read_sft(path):
        if rec.get("split") != split:
            continue
        yield {k: v for k, v in rec.items() if k not in _DROP_FIELDS}

"""Shared calibration-corpus ingestion: strided chunk sampling.

Every HF-side calibrator (AWQ mean|x| + α-search, GPTQ Hessians, imatrix
outlier stats) forwards the corpus in ``ctx``-sized chunks under a token
budget. Historically each of them took the budget from the *head* of the
file (``ids[:tokens]``) — on a corpus with a 250k-token wiki prefix that
meant calibrating on pure wiki and never reaching a single tool-call window
(llama-imatrix, which streams the whole file, was the only calibrator that
saw the real distribution). ``sample_chunks`` replaces head-truncation with
a deterministic even stride across the entire corpus, so the same budget
buys a representative sample instead of whatever happens to lead the file.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


def sample_chunks(ids: torch.Tensor, ctx: int, budget_tokens: int) -> list[torch.Tensor]:
    """Split ``ids`` into ``ctx``-sized chunks and return ~``budget_tokens``
    worth of them, evenly strided across the full corpus.

    Deterministic (no RNG). When the budget covers the whole corpus every
    chunk is returned, in order; otherwise the stride always includes the
    first and last eligible chunk. Chunks shorter than 2 tokens (a trailing
    sliver) are never returned.
    """
    if ctx < 2:
        raise ValueError(f"ctx must be >= 2, got {ctx}")
    if budget_tokens < 1:
        raise ValueError(f"budget_tokens must be >= 1, got {budget_tokens}")
    chunks = [c for c in ids.split(ctx) if c.numel() >= 2]
    if not chunks:
        return []
    n_take = min(len(chunks), max(1, math.ceil(budget_tokens / ctx)))
    if n_take == len(chunks):
        return chunks
    # Even stride over [0, len-1], first and last always included.
    idx = torch.linspace(0, len(chunks) - 1, n_take).round().long()
    seen: set[int] = set()
    picked: list[torch.Tensor] = []
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            picked.append(chunks[i])
    return picked


def load_chunks(
    tokenizer,
    text_path: Path,
    *,
    ctx: int,
    budget_tokens: int,
    log_tag: str | None = None,
) -> list[torch.Tensor]:
    """Read + tokenize the *entire* corpus file, then stride-sample chunks.

    ``add_special_tokens=False`` — chat corpora already carry their control
    tokens in-text and ``transformers`` encodes them as single special IDs.
    """
    text = text_path.read_text()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    chunks = sample_chunks(ids, ctx, budget_tokens)
    if log_tag is not None:
        total = max(1, math.ceil(ids.shape[0] / ctx))
        print(
            f"[{log_tag}] corpus {ids.shape[0]} tokens -> sampled "
            f"{len(chunks)}/{total} chunks @ctx {ctx} (budget {budget_tokens})",
            file=sys.stderr,
        )
    return chunks


def sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    """Return ``n`` rows of ``x`` evenly spaced over dim 0 (all rows if
    ``n >= len``). Deterministic; used to spread proxy-activation capture
    across a chunk instead of taking its (boilerplate-heavy) head."""
    if n >= x.shape[0]:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, n, device=x.device).round().long()
    return x[idx]

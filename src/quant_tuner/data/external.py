"""External (non-log) corpus sources: the eaddario parquet eval domains.

Owned here rather than in ``scripts/build_corpora.py`` so the one-off builder and the
universal builder (:mod:`quant_tuner.data.universal`) sample the eval distributions the
same way. Two builders drawing eval text differently would make PPL/KLD numbers from
different runs quietly incomparable.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

EAD_REPO = "eaddario/imatrix-calibration"
EVAL_DOMAINS = ("code_small", "math_small", "tools_small")
GENERAL_EVAL_DOMAIN = "combined_en_tiny"
EAD_CACHE = REPO / "out" / "external" / "imatrix-calibration"


def download_parquet(domain: str, cache_dir: Path | None = None) -> Path:
    """Return a local path to ``<domain>.parquet``, downloading if absent."""
    from huggingface_hub import hf_hub_download

    cache = Path(cache_dir) if cache_dir else EAD_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{domain}.parquet"
    if local.exists():
        return local
    print(f"  downloading {EAD_REPO}/{domain}.parquet ...", file=sys.stderr)
    fetched = hf_hub_download(
        repo_id=EAD_REPO,
        filename=f"{domain}.parquet",
        repo_type="dataset",
        local_dir=cache,
    )
    return Path(fetched)


def sample_parquet_text(
    parquet_path: Path, tok, target_tokens: int, seed: int,
) -> tuple[str, int, int]:
    """Sample ``content``-column text until we hit ``target_tokens`` under ``tok``.

    The eaddario parquet files often pack all content into a single very large row, so we
    additionally truncate at a token offset (deterministic via ``seed``) when one row would
    exceed the target.

    Returns ``(joined_text, actual_token_count, n_rows_or_chunks_used)``.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["content"])
    rows = [r for r in table.column("content").to_pylist() if isinstance(r, str)]
    rng = random.Random(seed)
    rng.shuffle(rows)

    out_texts: list[str] = []
    total = 0
    used = 0
    for r in rows:
        if total >= target_tokens:
            break
        ids = tok(r, add_special_tokens=False)["input_ids"]
        remaining = target_tokens - total
        if len(ids) <= remaining:
            out_texts.append(r.strip())
            total += len(ids)
            used += 1
        else:
            # Random offset window so we don't always sample the head of huge rows.
            max_start = max(0, len(ids) - remaining)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            chunk = tok.decode(ids[start : start + remaining], skip_special_tokens=True)
            out_texts.append(chunk.strip())
            total += remaining
            used += 1
            break
    return "\n\n".join(out_texts), total, used


__all__ = [
    "EAD_CACHE",
    "EAD_REPO",
    "EVAL_DOMAINS",
    "GENERAL_EVAL_DOMAIN",
    "download_parquet",
    "sample_parquet_text",
]

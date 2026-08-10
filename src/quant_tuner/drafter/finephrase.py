"""Stream a longest-first subset of HuggingFaceFW/finephrase into drafter windows.

FinePhrase is 4 configs (faq/math/table/tutorial) × 1000 parquet shards × ~270 MB
(≈486B generated tokens total). We never download it whole — pyarrow reads shards
over HTTP with column projection, and we keep only the longest, clean generations.

Each row carries `rollout_results` (a list of generations, each with
`finish_reason`, `text`, `usage.completion_tokens`). We take the best generation
per row (finished, longest), drop degenerate/short ones, and tokenize the
generated instructional text — that's the dataset's actual contribution and the
closest match to what an instruct target emits at serving.

Output = the SAME windows JSONL as `drafter.windows` (`{"input_ids", ...}`), so it
plugs straight into `drafter.train`. Deterministic given (configs, shard order).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BASE = "https://huggingface.co/datasets/HuggingFaceFW/finephrase/resolve/main"
_CONFIGS = ("faq", "math", "table", "tutorial")
_SHARDS_PER_CONFIG = 1000


@dataclass
class FinePhraseConfig:
    out: Path
    token_budget: int = 200_000_000
    """Stop once this many tokens are written. ~200M is a sane first pass on one
    4060 Ti (a full 1B is supported, just slower to stream + train)."""
    min_gen_tokens: int = 1024
    """Length floor on the generation. Doubles as the 'longest-first' filter:
    generations cap at 2048 (dataset max_tokens), so a high floor keeps the long
    tail and drops short/degenerate rows ('No answer' etc.)."""
    max_len: int = 2048
    """Window length. FinePhrase generations top out ~2048, so windows are
    short-to-medium — this is a breadth/scale asset, not a long-context one."""
    configs: tuple[str, ...] = _CONFIGS
    shards_per_config: int = _SHARDS_PER_CONFIG
    start_shard: int = 0
    """First shard index to read — set high (e.g. 900) to carve a held-out eval
    set disjoint from a low-shard training pull."""
    log_every_shards: int = 5
    hf_token: str | None = None
    _stats: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.token_budget < self.max_len:
            raise ValueError("token_budget must be >= max_len")
        if self.min_gen_tokens < 1:
            raise ValueError("min_gen_tokens must be >= 1")
        bad = set(self.configs) - set(_CONFIGS)
        if bad:
            raise ValueError(f"unknown configs: {bad}")


def best_generation(rollout_results: Any, min_tokens: int) -> str | None:
    """Return the longest finished generation text >= min_tokens, else None."""
    if not rollout_results:
        return None
    best_text, best_n = None, -1
    for g in rollout_results:
        if not isinstance(g, dict) or g.get("finish_reason") not in ("stop", "length"):
            continue
        text = g.get("text")
        n = (g.get("usage") or {}).get("completion_tokens", 0)
        if isinstance(text, str) and text.strip() and n >= min_tokens and n > best_n:
            best_text, best_n = text, n
    return best_text


def _iter_shard_urls(cfg: FinePhraseConfig) -> Iterator[str]:
    # Interleave configs so a small budget still spans all four styles.
    for shard in range(cfg.start_shard, cfg.shards_per_config):
        for c in cfg.configs:
            yield f"{_BASE}/{c}/000_00000_{shard}.parquet"


def stream_windows(cfg: FinePhraseConfig, tokenizer: Any) -> Iterator[dict]:
    """Yield window dicts until the token budget is met. Reads only the
    `rollout_results` column of each shard over HTTP."""
    import fsspec
    import pyarrow.parquet as pq

    cfg.validate()
    storage_options = {"headers": {"Authorization": f"Bearer {cfg.hf_token}"}} if cfg.hf_token else {}
    written = kept = seen = shards = 0
    for url in _iter_shard_urls(cfg):
        if written >= cfg.token_budget:
            break
        try:
            with fsspec.open(url, **storage_options) as f:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=1000, columns=["rollout_results"]):
                    for rr in batch.column("rollout_results").to_pylist():
                        seen += 1
                        gen = best_generation(rr, cfg.min_gen_tokens)
                        if gen is None:
                            continue
                        ids = tokenizer.encode(gen, add_special_tokens=False)
                        for start in range(0, len(ids), cfg.max_len):
                            piece = ids[start : start + cfg.max_len]
                            if len(piece) < 2:
                                break
                            kept += 1
                            written += len(piece)
                            yield {"input_ids": piece, "source": "finephrase", "n_tokens": len(piece)}
                        if written >= cfg.token_budget:
                            break
                    if written >= cfg.token_budget:
                        break
        except Exception as exc:  # a bad shard shouldn't kill a long stream
            print(f"[finephrase] skip {url.split('/')[-1]}: {exc!r}", flush=True)
            continue
        shards += 1
        if shards % cfg.log_every_shards == 0:
            print(f"[finephrase] {shards} shards, {written/1e6:.1f}M tokens, "
                  f"{kept} windows, {seen} rows scanned", flush=True)
    cfg._stats = {"tokens": written, "windows": kept, "rows_scanned": seen, "shards": shards}


def write_windows(cfg: FinePhraseConfig, tokenizer: Any) -> dict:
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as f:
        for w in stream_windows(cfg, tokenizer):
            f.write(json.dumps(w) + "\n")
    (cfg.out.with_suffix(".audit.json")).write_text(json.dumps(cfg._stats, indent=2))
    return cfg._stats

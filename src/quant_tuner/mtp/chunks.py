"""Tokenize logtrain into fixed-length chunks for MTP training/capture.

Torch-free: only depends on `random`, the HuggingFace tokenizer, and the
project's data utilities. This lets the IQ3_S hidden-state capture script
import the same chunker as `train_mtp_head.py` without pulling in torch
(which conflicts with llama-cpp-python on the GPU).

Determinism invariant: same (seed, seq_len, max_tokens, wiki_mix, wiki_text)
inputs produce byte-for-byte identical chunk order. The capture script and
the trainer rely on this to align cached hidden states with training-loop
chunk indices.
"""

from __future__ import annotations

import json as _json
import random
from pathlib import Path

from quant_tuner.data import ingest, split
from quant_tuner.experiments import log


def session_to_text(session: dict, tok: object) -> str:
    """Render a session to a flat text string via the tokenizer's chat template."""
    raw_messages = session.get("messages", [])
    if not raw_messages:
        return ""

    messages: list[dict] = []
    for m in raw_messages:
        if isinstance(m, str):
            try:
                m = _json.loads(m)
            except _json.JSONDecodeError:
                continue
        if isinstance(m, dict):
            messages.append(m)

    if not messages:
        return ""

    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def build_token_batches(
    model_dir: Path,
    logtrain: Path,
    seq_len: int,
    max_tokens: int,
    seed: int = 42,
    wiki_mix: float = 0.0,
    wiki_text: Path | None = None,
) -> list[list[int]]:
    """Tokenize logtrain train split (+ optional wiki mix) and pack into chunks.

    Each returned chunk has length `seq_len + 2`; the trainer slices it as
    input_ids = chunk[:-2], next_ids = chunk[1:-1], labels = chunk[2:].
    """
    from transformers import AutoTokenizer

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=True)

    sessions = ingest.load_sessions(logtrain)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=seed
    )
    # split_sessions builds its split sets from `set(...)` which iterates in
    # PYTHONHASHSEED-dependent order. Sort by fingerprint so chunk order is
    # stable across processes (the IQ3_S cache and the trainer tokenize in
    # separate processes and must agree).
    train_sessions = sorted(splits["train"], key=ingest.session_fingerprint)
    log(f"  {len(train_sessions)} train sessions")

    logs_ids: list[int] = []
    for s in train_sessions:
        text = session_to_text(s, tok)
        ids = tok.encode(text, add_special_tokens=False)
        logs_ids.extend(ids)

    logs_ids = logs_ids[:max_tokens]
    log(f"  {len(logs_ids):,} tool-call tokens")

    def _to_chunks(token_ids: list[int]) -> list[list[int]]:
        out = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            out.append(token_ids[i : i + seq_len + 2])
        return out

    logs_chunks = _to_chunks(logs_ids)

    if wiki_mix <= 0.0:
        return logs_chunks

    if wiki_text is not None and wiki_text.exists():
        raw = wiki_text.read_text(encoding="utf-8", errors="replace")
        wiki_ids = tok.encode(raw, add_special_tokens=False)
    else:
        log("  loading wikitext-2-raw-v1 from HuggingFace datasets …")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=False)
        wiki_ids = []
        for row in ds:
            text = row["text"]
            if text.strip():
                wiki_ids.extend(tok.encode(text, add_special_tokens=False))
    log(f"  {len(wiki_ids):,} wiki tokens")

    wiki_chunks = _to_chunks(wiki_ids)

    n_total = len(logs_chunks) + len(wiki_chunks)
    n_wiki  = min(len(wiki_chunks), round(n_total * wiki_mix))
    n_logs  = len(logs_chunks)

    rng.shuffle(wiki_chunks)
    mixed = logs_chunks + wiki_chunks[:n_wiki]
    rng.shuffle(mixed)
    log(f"  mixed corpus: {n_logs} tool-call + {n_wiki} wiki chunks "
        f"({n_wiki / len(mixed) * 100:.0f}% wiki)")
    return mixed

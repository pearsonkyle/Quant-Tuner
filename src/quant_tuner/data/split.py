"""Stratified packing + train/test/holdout splitting for calibration corpora."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from quant_tuner.data.ingest import (
    coerce_tool_call_arguments,
    normalize_messages,
    session_fingerprint,
)


def length_bucket(n_msgs: int) -> str:
    if n_msgs <= 20:
        return "short"
    if n_msgs <= 80:
        return "medium"
    return "long"


def session_tools(session: dict, messages: list[dict]) -> list | None:
    """Resolve the tool schemas for a session.

    Schemas may live at the top level (`session["tools"]`) or, as in the
    logtrain export, attached to the system message (`messages[0]["tools"]`).
    Pass these to apply_chat_template so the rendered calibration corpus
    contains the tool/function schemas the model conditions on at inference —
    without them, every tool-calling session is rendered with no schema context.
    """
    top = session.get("tools")
    if top:
        return top
    for m in messages:
        if isinstance(m, dict) and m.get("tools"):
            return m["tools"]
    return None


def template_session(tokenizer, messages: list[dict], tools) -> tuple[str, int]:
    coerce_tool_call_arguments(messages)
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    ntok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, ntok


def cap_session(tokenizer, messages: list[dict], tools, cap_tokens: int) -> tuple[str, int]:
    """Render the session; if it exceeds cap_tokens, binary-search the longest message
    prefix that fits. Always keeps at least the first 2 messages."""
    text, ntok = template_session(tokenizer, messages, tools)
    if ntok <= cap_tokens or len(messages) <= 2:
        return text, ntok
    lo, hi = 2, len(messages)
    best_text, best_ntok = text, ntok
    while lo < hi:
        mid = (lo + hi + 1) // 2
        sub_text, sub_ntok = template_session(tokenizer, messages[:mid], tools)
        if sub_ntok <= cap_tokens:
            best_text, best_ntok = sub_text, sub_ntok
            lo = mid
        else:
            hi = mid - 1
    return best_text, best_ntok


def stratified_pack(
    sessions: list[dict],
    tokenizer,
    target_tokens: int,
    per_session_cap: int = 6_000,
    seed: int = 42,
) -> tuple[list[str], list[dict], int, dict[str, Any]]:
    """Round-robin across (source, length_bucket) strata. Returns (chunks, kept, total, audit)."""
    rng = random.Random(seed)

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in sessions:
        src = s.get("source", "unknown")
        n = len(s.get("messages", []))
        strata[(src, length_bucket(n))].append(s)
    for key in strata:
        rng.shuffle(strata[key])

    stratum_keys = sorted(strata.keys())
    cursors = dict.fromkeys(stratum_keys, 0)

    chunks: list[str] = []
    kept: list[dict] = []
    total = 0
    truncations = 0
    per_source: Counter = Counter()
    per_bucket: Counter = Counter()
    per_stratum: Counter = Counter()

    exhausted: set[tuple[str, str]] = set()
    while total < target_tokens and len(exhausted) < len(stratum_keys):
        for key in stratum_keys:
            if key in exhausted:
                continue
            idx = cursors[key]
            if idx >= len(strata[key]):
                exhausted.add(key)
                continue
            s = strata[key][idx]
            cursors[key] = idx + 1

            messages = normalize_messages(s.get("messages", []))
            if not messages:
                continue
            tools = session_tools(s, messages)
            try:
                text, ntok = cap_session(tokenizer, messages, tools, per_session_cap)
            except Exception:
                continue
            if ntok == 0:
                continue
            try:
                _, full_ntok = template_session(tokenizer, messages, tools)
            except Exception:
                full_ntok = ntok
            if ntok < full_ntok:
                truncations += 1

            remaining = target_tokens - total
            if ntok > remaining * 1.10:
                continue

            chunks.append(text.strip())
            kept.append(s)
            total += ntok
            per_source[key[0]] += ntok
            per_bucket[key[1]] += ntok
            per_stratum[f"{key[0]}/{key[1]}"] += ntok

            if total >= target_tokens:
                break

    audit = {
        "total_tokens": total,
        "session_count": len(chunks),
        "truncated_sessions": truncations,
        "per_session_cap": per_session_cap,
        "tokens_per_source": dict(per_source),
        "tokens_per_length_bucket": dict(per_bucket),
        "tokens_per_stratum": dict(per_stratum),
        "sessions_available_per_stratum": {f"{k[0]}/{k[1]}": len(v) for k, v in strata.items()},
        "sessions_consumed_per_stratum": {f"{k[0]}/{k[1]}": cursors[k] for k in stratum_keys},
    }
    return chunks, kept, total, audit


def write_corpus(chunks: list[str], path: Path, supplement: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")
        if supplement is not None:
            text = Path(supplement).read_text()
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


def split_sessions(
    sessions: list[dict],
    train_frac: float = 0.8,
    test_frac: float = 0.1,
    holdout_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Disjoint shuffle-split by session fingerprint."""
    assert abs(train_frac + test_frac + holdout_frac - 1.0) < 1e-6, "splits must sum to 1.0"
    rng = random.Random(seed)
    by_fp = {session_fingerprint(s): s for s in sessions}
    fps = sorted(by_fp.keys())
    rng.shuffle(fps)
    n = len(fps)
    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    # Iterate the deterministically-shuffled `fps` list (not sets) so the order
    # within each split is reproducible. Building lists by iterating a set makes
    # ordering depend on PYTHONHASHSEED, which silently changes which sessions a
    # downstream token-budgeted stratified_pack selects run-to-run.
    train_fps = fps[:n_train]
    test_fps = fps[n_train : n_train + n_test]
    holdout_fps = fps[n_train + n_test :]
    return {
        "train": [by_fp[fp] for fp in train_fps],
        "test": [by_fp[fp] for fp in test_fps],
        "holdout": [by_fp[fp] for fp in holdout_fps],
    }


def write_split_jsonl(sessions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")


__all__ = [
    "cap_session",
    "length_bucket",
    "session_tools",
    "split_sessions",
    "stratified_pack",
    "template_session",
    "write_corpus",
    "write_split_jsonl",
]

"""Stratified packing + train/test/holdout splitting for calibration corpora."""

from __future__ import annotations

import hashlib
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

# A window body may begin on a user or assistant turn, but never on a `tool`
# message: a `tool` role only makes sense as the response to a preceding
# assistant `tool_calls`, so starting a window there orphans it and most chat
# templates either error or render off-distribution output.
_BODY_START_ROLES = ("user", "assistant")


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


def stub_system_content(content: str, tokenizer, budget_tokens: int) -> str:
    """Trim system-message *prose* to ~budget_tokens (keeping the head).

    Tool/function schemas are rendered from the separate ``tools=`` argument of
    ``apply_chat_template`` — they do NOT live in this prose — so trimming here
    frees window budget without touching the tool-calling conditioning the model
    actually attends to. Non-string content (rare multimodal parts) is returned
    unchanged since we can't safely truncate it.
    """
    if not isinstance(content, str):
        return content
    ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget_tokens:
        return content
    return tokenizer.decode(ids[:budget_tokens], skip_special_tokens=True)


def session_windows(
    tokenizer,
    messages: list[dict],
    tools,
    cap_tokens: int,
    *,
    system_content: str | None = None,
    max_windows: int = 8,
) -> list[tuple[str, int]]:
    """Slice one session into ≤max_windows chat-templated windows of ≤cap_tokens.

    Each window = an optional system message (with ``system_content`` substituted
    when given — typically a trimmed stub) + a contiguous run of body messages.
    Windows are non-overlapping and walk the session in order, so a long agentic
    trajectory yields several windows that each contain real tool-call turns and
    a clean termination — instead of one head-anchored prefix dominated by the
    system prompt. A window body never starts on a `tool` message.

    Returns a list of (text, ntok). A window whose first body message alone
    exceeds ``cap_tokens`` is still emitted (ntok > cap_tokens) so nothing is
    silently dropped; callers can detect it by comparing ntok to the cap.
    """
    coerce_tool_call_arguments(messages)
    if messages and messages[0].get("role") == "system":
        sys_msg = dict(messages[0])
        if system_content is not None:
            sys_msg["content"] = system_content
        prefix = [sys_msg]
        body = messages[1:]
    else:
        prefix = []
        body = messages

    def render(run: list[dict]) -> tuple[str, int]:
        text = tokenizer.apply_chat_template(prefix + run, tools=tools, tokenize=False)
        ntok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return text, ntok

    windows: list[tuple[str, int]] = []
    i, n = 0, len(body)
    while i < n and len(windows) < max_windows:
        # Snap the window start to a non-orphan boundary.
        while i < n and body[i].get("role") not in _BODY_START_ROLES:
            i += 1
        if i >= n:
            break
        # Binary-search the largest j>i such that body[i:j] fits the cap.
        # Some chat templates are strict and refuse a window that lacks a user
        # turn (e.g. one that starts on an assistant reply orphaned at a window
        # boundary — Qwen3.5-VL raises "No user query found"). Treat such a
        # render failure as "skip this start" rather than letting it bubble up
        # and discard the whole session; lenient templates never raise here.
        try:
            lo, hi, best_j = i + 1, n, None
            while lo <= hi:
                mid = (lo + hi) // 2
                _, ntok = render(body[i:mid])
                if ntok <= cap_tokens:
                    best_j = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_j is None:  # single message larger than the cap — keep it anyway
                best_j = i + 1
            text, ntok = render(body[i:best_j])
        except Exception:
            i += 1
            continue
        if ntok > 0:
            windows.append((text.strip(), ntok))
        i = best_j
    return windows


def stratified_pack(
    sessions: list[dict],
    tokenizer,
    target_tokens: int,
    per_session_cap: int = 6_000,
    seed: int = 42,
    *,
    system_prose_budget: int | None = None,
    full_prose_quota: int = 1,
    max_windows_per_session: int = 8,
) -> tuple[list[str], list[dict], int, dict[str, Any]]:
    """Round-robin across (source, length_bucket) strata. Returns (chunks, kept, total, audit).

    Default behavior (``system_prose_budget=None``) is unchanged: one
    head-anchored ``cap_session`` chunk per session.

    When ``system_prose_budget`` is set, the calibration-tuned path kicks in:
      * Each session is sliced into multiple ≤``per_session_cap`` windows via
        ``session_windows`` (so later tool-call turns + terminations survive,
        instead of being truncated off the tail).
      * The system *prose* is replaced with a ≤``system_prose_budget``-token stub
        for all but the first ``full_prose_quota`` sessions sharing an identical
        system prompt — schemas (rendered from ``tools=``) are untouched. This
        deduplicates the giant repeated boilerplate that otherwise dominates the
        imatrix, freeing the budget for diverse tool-call signal.
    """
    rng = random.Random(seed)
    windowed = system_prose_budget is not None
    prose_budget = system_prose_budget if system_prose_budget is not None else 0
    full_prose_counts: Counter = Counter()
    full_prose_sessions = 0
    stub_sessions = 0
    windows_emitted = 0
    system_tokens_total = 0
    oversize_windows = 0
    unique_system_prompts: set[str] = set()

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

            if not windowed:
                # --- legacy single-chunk path (unchanged) --------------------
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
                continue

            # --- stub + multi-window path -----------------------------------
            # Decide full vs. stub system prose, deduplicating the boilerplate.
            sys_content: str | None = None
            is_full = False
            sys_raw = messages[0].get("content") if messages[0].get("role") == "system" else None
            if isinstance(sys_raw, str):
                h = hashlib.sha256(sys_raw.encode()).hexdigest()
                unique_system_prompts.add(h)
                if full_prose_counts[h] < full_prose_quota:
                    full_prose_counts[h] += 1
                    sys_content = None  # render the full prose
                    is_full = True
                else:
                    sys_content = stub_system_content(sys_raw, tokenizer, prose_budget)

            # A full-prose session emits a SINGLE window: the prose is huge and
            # identical across windows, so multi-windowing it would just replay
            # the boilerplate. One window keeps the real long-prompt activations
            # represented once; the stub sessions carry the tool-turn signal.
            try:
                wins = session_windows(
                    tokenizer, messages, tools, per_session_cap,
                    system_content=sys_content,
                    max_windows=1 if is_full else max_windows_per_session,
                )
            except Exception:
                continue
            if not wins:
                continue

            # System-prose token count (for the body/tool-turn share in the audit).
            # Estimate from the content actually rendered (full vs. stub); the
            # template wrapper is small relative to the prose so we ignore it.
            used_sys = sys_content if sys_content is not None else sys_raw
            sys_only_tokens = (
                len(tokenizer(used_sys, add_special_tokens=False)["input_ids"])
                if isinstance(used_sys, str)
                else 0
            )

            kept_any = False
            for text, ntok in wins:
                remaining = target_tokens - total
                if remaining <= 0:
                    break
                if ntok > remaining * 1.10:
                    continue
                if ntok > per_session_cap:
                    oversize_windows += 1
                chunks.append(text)
                total += ntok
                windows_emitted += 1
                system_tokens_total += min(sys_only_tokens, ntok)
                per_source[key[0]] += ntok
                per_bucket[key[1]] += ntok
                per_stratum[f"{key[0]}/{key[1]}"] += ntok
                kept_any = True
            if kept_any:
                kept.append(s)
                if is_full:
                    full_prose_sessions += 1
                else:
                    stub_sessions += 1
            if total >= target_tokens:
                break

    audit = {
        "total_tokens": total,
        "chunk_count": len(chunks),
        "session_count": len(kept),
        "truncated_sessions": truncations,
        "per_session_cap": per_session_cap,
        "tokens_per_source": dict(per_source),
        "tokens_per_length_bucket": dict(per_bucket),
        "tokens_per_stratum": dict(per_stratum),
        "sessions_available_per_stratum": {f"{k[0]}/{k[1]}": len(v) for k, v in strata.items()},
        "sessions_consumed_per_stratum": {f"{k[0]}/{k[1]}": cursors[k] for k in stratum_keys},
    }
    if windowed:
        body_tokens = total - system_tokens_total
        audit["windowing"] = {
            "system_prose_budget": system_prose_budget,
            "full_prose_quota": full_prose_quota,
            "max_windows_per_session": max_windows_per_session,
            "windows_emitted": windows_emitted,
            "sessions_kept": len(kept),
            "avg_windows_per_session": round(windows_emitted / max(1, len(kept)), 2),
            "unique_system_prompts": len(unique_system_prompts),
            "full_prose_sessions": full_prose_sessions,
            "stub_sessions": stub_sessions,
            "oversize_windows": oversize_windows,
            "system_prose_tokens": system_tokens_total,
            "body_tokens": body_tokens,
            "tool_turn_token_share": round(body_tokens / max(1, total), 4),
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
    "session_windows",
    "split_sessions",
    "stratified_pack",
    "stub_system_content",
    "template_session",
    "write_corpus",
    "write_split_jsonl",
]

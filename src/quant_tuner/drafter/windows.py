"""Long-context training windows from usage logs for MTP-drafter fine-tuning.

Unlike the calibration corpora (capped at a few-k tokens per window because
llama-imatrix chunks at 4096), drafter training wants the **full agentic
trajectory**: real Claude-Code sessions run 15k-290k tokens (median ~46k), and a
drafter that only ever sees ≤8k prefixes has low acceptance deep into a session —
exactly where long agent runs live. So we template each session whole and slice
it into ``max_len``-token windows (default 32768), keeping *every* window rather
than head-anchoring.

Output is a JSONL of ``{"input_ids": [...], "session": <fp>, "window": i}`` —
consumed by ``drafter.train``. Deterministic (session fingerprint order).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_tuner.data.ingest import load_sessions, normalize_messages, session_fingerprint
from quant_tuner.data.split import (
    session_tools,
    split_sessions,
    stratified_pack,
    template_session,
)


@dataclass
class WindowConfig:
    logs: Path
    out: Path
    max_len: int = 32_768
    """Window length in tokens. Above the 8k calibration default on purpose —
    agentic sessions are long and the drafter must see that regime."""
    stride: int | None = None
    """Slide between window starts. None = non-overlapping (stride == max_len).
    A stride < max_len yields overlapping windows (more training signal from
    long sessions) at the cost of duplicated prefixes."""
    min_len: int = 256
    """Drop trailing windows shorter than this (uninformative slivers)."""
    split: str = "train"
    """Which split to draw from — keep to 'train' so eval slices stay clean."""
    system_prose_budget: int | None = None
    """Tool-dense mode. When set, route through ``stratified_pack`` instead of
    templating whole sessions: system prose is stubbed to this many tokens (and
    deduped across sessions sharing a prompt) and tool schemas deduped, so each
    window covers more tool-call turns per token instead of being dominated by
    repeated boilerplate. ``None`` = full-session templating (default)."""
    per_session_cap: int = 6_000
    """Max tokens per packed window in tool-dense mode."""
    token_budget: int = 50_000_000
    """Tool-dense mode: target total tokens for stratified_pack (high = pack all
    of the train split)."""


def iter_windows(cfg: WindowConfig, tokenizer: Any) -> list[dict]:
    """Template each session in the chosen split and slice into windows.
    Returns a list of ``{"input_ids", "session", "window", "n_tokens"}`` dicts."""
    stride = cfg.max_len if cfg.stride is None else cfg.stride
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if cfg.max_len < cfg.min_len:
        raise ValueError("max_len must be >= min_len")

    sessions = load_sessions(cfg.logs)
    chosen = split_sessions(sessions)[cfg.split]

    # Tool-dense mode: let stratified_pack stub/dedup boilerplate and slice
    # sessions into tool-call-dense windows, then tokenize each chunk.
    if cfg.system_prose_budget is not None:
        chunks, _kept, _total, _audit = stratified_pack(
            chosen, tokenizer, target_tokens=cfg.token_budget,
            per_session_cap=cfg.per_session_cap,
            system_prose_budget=cfg.system_prose_budget,
        )
        out: list[dict] = []
        for ci, text in enumerate(chunks):
            ids = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, len(ids), cfg.max_len):
                window = ids[start : start + cfg.max_len]
                if len(window) < cfg.min_len:
                    break
                out.append(
                    {"input_ids": window, "source": "logs-tooldense",
                     "chunk": ci, "n_tokens": len(window)}
                )
        return out

    out: list[dict] = []
    for s in chosen:
        fp = session_fingerprint(s)
        msgs = normalize_messages(s["messages"])
        tools = session_tools(s, msgs)
        try:
            text, _ = template_session(tokenizer, msgs, tools)
        except Exception:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        for wi, start in enumerate(range(0, len(ids), stride)):
            window = ids[start : start + cfg.max_len]
            if len(window) < cfg.min_len:
                break
            out.append(
                {"input_ids": window, "session": fp, "window": wi, "n_tokens": len(window)}
            )
    return out


def write_windows(cfg: WindowConfig, tokenizer: Any) -> dict:
    """Write windows to ``cfg.out`` (JSONL). Returns a small audit dict."""
    windows = iter_windows(cfg, tokenizer)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as f:
        for w in windows:
            f.write(json.dumps(w) + "\n")
    lens = sorted(w["n_tokens"] for w in windows)
    audit = {
        "n_windows": len(windows),
        "n_sessions": len({w["session"] for w in windows if "session" in w}),
        "total_tokens": sum(lens),
        "min_len": lens[0] if lens else 0,
        "max_len": lens[-1] if lens else 0,
        "median_len": lens[len(lens) // 2] if lens else 0,
        "windows_over_8k": sum(1 for x in lens if x > 8192),
    }
    (cfg.out.with_suffix(".audit.json")).write_text(json.dumps(audit, indent=2))
    return audit

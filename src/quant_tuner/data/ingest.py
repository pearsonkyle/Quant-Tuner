"""Load and filter JSONL usage-log sessions for calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_sessions(path: Path | str) -> list[dict]:
    """Load JSONL sessions. Each line is a JSON object."""
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def filter_sessions(
    sessions: list[dict],
    min_score: float = 0.3,
    require_tools: bool = True,
) -> list[dict]:
    """Drop low-quality sessions and (optionally) sessions with no tool calls."""
    out = []
    for s in sessions:
        if s.get("score", 0) < min_score:
            continue
        if require_tools and s.get("metrics", {}).get("tool_calls", 0) == 0:
            continue
        out.append(s)
    return out


def normalize_messages(raw_messages: list) -> list[dict]:
    """Sessions may store messages as JSON-encoded strings; parse them in place."""
    out: list[dict] = []
    for m in raw_messages:
        if isinstance(m, str):
            try:
                out.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        elif isinstance(m, dict):
            out.append(m)
    return out


def coerce_tool_call_arguments(messages: list[dict]) -> None:
    """Some chat templates require `tool_calls[i].function.arguments` to be a dict, not a JSON string."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            for holder in (tc, fn):
                if not isinstance(holder, dict):
                    continue
                args = holder.get("arguments")
                if isinstance(args, str):
                    try:
                        holder["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        holder["arguments"] = {}


def session_fingerprint(s: dict) -> str:
    """Stable hash for deduping a session across calibration/eval splits."""
    key = {
        "source": s.get("source"),
        "n_messages": len(s.get("messages", [])),
        "first_message": s.get("messages", [None])[0] if s.get("messages") else None,
        "score": s.get("score"),
    }
    payload = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()

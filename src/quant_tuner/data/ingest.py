"""Load and filter JSONL usage-log sessions for calibration.

Two on-disk log corpora, both gzipped, both under ``datasets/agent-logs/data/``:

* ``logs-cli.jsonl.gz`` — **CLI usage logs** (the file previously called ``logtrain.jsonl``).
  Interactive coding sessions captured from Claude Code / opencode / qwen code. Rows carry
  ``source``, ``score``, ``metrics`` and messages stored as JSON *strings*.
* ``logs-agents.jsonl.gz`` — **harvested agent trajectories**: verified (tests-passed)
  issue-solving runs across 19 languages, 7 agent scaffolds and 3 solver models. Rows carry
  ``messages`` (dicts, with structured ``tool_calls``), ``tools`` and a ``meta`` block.

:func:`load_sessions` sniffs the format per row and returns the same session shape for both,
so every caller (packers, splitters, QAT corpus builders) sees one schema. Reading is
transparently gzip-aware — keeping these compressed is a 5× disk win on the CLI logs, and the
packers are tokenizer-bound rather than I/O-bound, so decompression is free in practice.

The agent rows are tagged ``source="agents:<language>"`` on purpose: ``stratified_pack``
round-robins across ``(source, length_bucket)`` strata, so per-language tagging is what makes
a token budget spread across all 19 languages instead of being eaten by whichever language
happens to have the longest trajectories.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

LOGS_DIR = REPO / "datasets" / "agent-logs" / "data"
CLI_LOGS = LOGS_DIR / "logs-cli.jsonl.gz"
AGENT_LOGS = LOGS_DIR / "logs-agents.jsonl.gz"
DEFAULT_LOG_FILES: tuple[Path, ...] = (CLI_LOGS, AGENT_LOGS)

# Pre-rename layout. Historical experiment scripts hardcode `REPO/logtrain.jsonl`; rather
# than rewrite thirty reproduction scripts, resolve the old name to the new file so they keep
# reproducing. Anything new should use CLI_LOGS / DEFAULT_LOG_FILES.
_LEGACY_NAMES = {"logtrain.jsonl", "logtrain.jsonl.gz", "logs-cli.jsonl"}


def resolve_log_path(path: Path | str) -> Path:
    """Map a log path to what actually exists on disk (legacy name, or a ``.gz`` sibling)."""
    p = Path(path)
    if p.exists():
        return p
    gz = p.with_suffix(p.suffix + ".gz")
    if gz.exists():
        return gz
    if p.name in _LEGACY_NAMES and CLI_LOGS.exists():
        return CLI_LOGS
    return p


def open_jsonl(path: Path | str) -> io.TextIOBase:
    """Open a ``.jsonl`` or ``.jsonl.gz`` for text reading."""
    p = resolve_log_path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


def _is_agent_row(row: dict) -> bool:
    """Harvested-trajectory format: a ``meta`` block and no CLI-log ``source`` field."""
    return isinstance(row.get("meta"), dict) and "source" not in row


def _from_agent_row(row: dict) -> dict:
    """Normalize a harvested trajectory into the common session shape.

    ``score`` is 1.0 exactly when the run was graded resolved, so the standard
    ``filter_sessions(min_score=…)`` gate keeps unverified runs out of calibration without
    callers needing to know this format exists.
    """
    meta = row.get("meta") or {}
    messages = row.get("messages") or []
    n_calls = sum(len(m.get("tool_calls") or []) for m in messages if isinstance(m, dict))
    return {
        "id": f"{meta.get('instance_id')}|{meta.get('agent')}|{meta.get('model')}",
        "source": f"agents:{meta.get('language') or 'unknown'}",
        "group": str(meta.get("instance_id") or ""),   # split unit — see session_group
        "messages": messages,
        "tools": row.get("tools") or [],
        "score": 1.0 if meta.get("resolved") else 0.0,
        "metrics": {"tool_calls": n_calls},
        "meta": meta,
    }


def load_sessions(path: Path | str) -> list[dict]:
    """Load JSONL sessions (plain or gzipped), normalizing both on-disk log formats."""
    out: list[dict] = []
    with open_jsonl(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(_from_agent_row(row) if _is_agent_row(row) else row)
    return out


def load_all_sessions(paths: list[Path] | tuple[Path, ...] | None = None) -> list[dict]:
    """Load and concatenate every configured log file, skipping any that are absent."""
    out: list[dict] = []
    for p in paths if paths is not None else DEFAULT_LOG_FILES:
        rp = resolve_log_path(p)
        if rp.exists():
            out.extend(load_sessions(rp))
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


def session_group(s: dict) -> str:
    """The unit a session must be split BY, when rows are not independent.

    The harvested agent logs hold ~4.6 runs of the *same* GitHub issue (different scaffolds
    and solver models: 435 rows over 94 instances). Splitting per row would put one attempt
    at an issue in calibration and another attempt at the identical issue in the eval
    holdout — the eval would then measure fit, not generalization. Grouping by
    ``instance_id`` keeps every attempt at an issue on the same side of the split. CLI
    sessions are independent, so each is its own group.
    """
    return s.get("group") or session_fingerprint(s)


__all__ = [
    "AGENT_LOGS",
    "CLI_LOGS",
    "DEFAULT_LOG_FILES",
    "LOGS_DIR",
    "coerce_tool_call_arguments",
    "filter_sessions",
    "load_all_sessions",
    "load_sessions",
    "normalize_messages",
    "open_jsonl",
    "resolve_log_path",
    "session_fingerprint",
    "session_group",
]

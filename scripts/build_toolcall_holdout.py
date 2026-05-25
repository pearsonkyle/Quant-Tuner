#!/usr/bin/env python3
"""Build a 10-session holdout from calibration_data.jsonl for tool-call eval.

Output: artifacts/toolcall_holdout.jsonl — one session per line:
    {session_id, tools, messages}

Sessions are sampled with a fixed seed and made disjoint (by
session_fingerprint) from the existing KLD calibration and eval subsets,
so this holdout can be evaluated independently of the quantization splits.
"""

import argparse
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from quant_tuner.data.ingest import (  # noqa: E402
    load_sessions,
    normalize_messages,
    session_fingerprint,
)


def existing_fingerprints(paths: list[str]) -> set[str]:
    seen: set[str] = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(session_fingerprint(s))
    return seen


_PATH_KEYS = {"file_path", "filepath", "filename", "path"}
_COMMAND_KEYS = {"command", "cmd"}

_ERROR_PATTERNS = (
    "error", "failed", "exception", "traceback",
    "command not found", "no such file", "permission denied",
    "exit code 1", "exit code: 1", "non-zero exit",
)


def looks_like_error(content: str | None) -> bool:
    if not content:
        return False
    lc = content.lower()
    return any(p in lc for p in _ERROR_PATTERNS)


def has_recovery_turn(messages: list[dict]) -> bool:
    """A session has a recovery opportunity if an error-bearing tool result is
    followed by an assistant turn that issues a *different* tool call (different
    tool name or different arguments than the call that produced the error)."""
    by_id: dict[str, tuple[str, dict]] = {}
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tid = tc.get("id")
                if not tid:
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                by_id[tid] = (fn.get("name") or tc.get("name"), args or {})
        elif m.get("role") == "tool" and looks_like_error(m.get("content")):
            failing = by_id.get(m.get("tool_call_id"))
            if failing is None:
                continue
            for nxt in messages[i + 1:]:
                if nxt.get("role") != "assistant":
                    continue
                for tc in nxt.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    if (fn.get("name"), args or {}) != failing:
                        return True
                break
    return False


def has_parallel_call(messages: list[dict]) -> bool:
    return any(
        m.get("role") == "assistant" and len(m.get("tool_calls") or []) >= 2
        for m in messages
    )


def first_assistant_call(messages: list[dict]) -> tuple[str | None, dict]:
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return fn.get("name"), args or {}
    return None, {}


def first_user_text(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def is_anchored(messages: list[dict]) -> bool:
    """The first ground-truth tool call's primary value (path basename or
    command argv[0]) must appear verbatim in the first user message. This
    ensures the model can ground its first action in the prompt without
    relying on agent-specific system framing."""
    name, args = first_assistant_call(messages)
    if not name:
        return False
    user_lc = first_user_text(messages).lower()
    if not user_lc:
        return False
    for k, v in args.items():
        if not isinstance(v, str) or not v.strip():
            continue
        kl = k.lower()
        if kl in _PATH_KEYS or "path" in kl:
            base = v.rsplit("/", 1)[-1].lower()
            if base and (base in user_lc or v.lower() in user_lc):
                return True
        if kl in _COMMAND_KEYS:
            argv0 = v.strip().split()[0].lower()
            if argv0 and argv0 in user_lc:
                return True
    return False


def extract_tools(messages: list[dict]) -> list[dict] | None:
    """Tools may live on the first system message (qwen) or first user message
    (claude/opencode), depending on the source."""
    for m in messages:
        if isinstance(m.get("tools"), list):
            return m["tools"]
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=os.path.join(_ROOT, "calibration_data.jsonl"))
    p.add_argument("--output", default=os.path.join(_ROOT, "artifacts", "toolcall_holdout.jsonl"))
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[
            os.path.join(_ROOT, "artifacts", "calib_custom__500000_16384_subset.jsonl"),
            os.path.join(_ROOT, "artifacts", "eval_kld_subset.jsonl"),
        ],
        help="JSONL files whose sessions should be excluded (fingerprint match).",
    )
    p.add_argument("--n", type=int, default=25,
                   help="Number of anchored (easy) sessions")
    p.add_argument("--n-recovery", type=int, default=5,
                   help="Number of sessions that include an error→recovery turn")
    p.add_argument("--n-parallel", type=int, default=5,
                   help="Number of sessions with at least one parallel tool-call turn")
    p.add_argument("--min-tool-calls", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--min-user-chars", type=int, default=200,
                   help="Minimum length of the first user message")
    p.add_argument("--require-anchor", action=argparse.BooleanOptionalAction, default=True,
                   help="Require the first ground-truth tool call's primary arg to appear "
                        "verbatim in the user message (so the model has something to ground on)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sessions = load_sessions(args.input)
    print(f"Loaded {len(sessions)} sessions from {args.input}")

    excluded = existing_fingerprints(args.exclude)
    print(f"Excluding {len(excluded)} session fingerprints from {len(args.exclude)} files")

    # Classify each candidate into all strata it qualifies for; one session
    # can appear in multiple stratum pools but is picked at most once.
    by_stratum_src: dict[str, dict[str, list[tuple]]] = {
        "anchored": {}, "recovery": {}, "parallel": {},
    }
    for s in sessions:
        if s.get("score", 0.0) < args.min_score:
            continue
        if s.get("metrics", {}).get("tool_calls", 0) < args.min_tool_calls:
            continue
        if session_fingerprint(s) in excluded:
            continue
        msgs = normalize_messages(s.get("messages", []))
        tools = extract_tools(msgs)
        if not tools:
            continue
        if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs):
            continue
        if len(first_user_text(msgs)) < args.min_user_chars:
            continue
        src = s.get("source") or "unknown"
        entry = (s, msgs, tools)
        if (not args.require_anchor) or is_anchored(msgs):
            by_stratum_src["anchored"].setdefault(src, []).append(entry)
        if has_recovery_turn(msgs):
            by_stratum_src["recovery"].setdefault(src, []).append(entry)
        if has_parallel_call(msgs):
            by_stratum_src["parallel"].setdefault(src, []).append(entry)

    for strat, by_src in by_stratum_src.items():
        print(f"Candidates per source ({strat}):",
              {k: len(v) for k, v in by_src.items()})

    rng = random.Random(args.seed)
    for by_src in by_stratum_src.values():
        for v in by_src.values():
            rng.shuffle(v)

    seen_ids: set[str] = set()
    picked: list[tuple[str, tuple]] = []  # (stratum, entry)

    def _round_robin(by_src: dict[str, list[tuple]], quota: int, stratum: str) -> int:
        srcs = sorted(by_src.keys())
        cursors = {s: 0 for s in srcs}
        taken = 0
        while taken < quota:
            progressed = False
            for src in srcs:
                if taken >= quota:
                    break
                while cursors[src] < len(by_src[src]):
                    entry = by_src[src][cursors[src]]
                    cursors[src] += 1
                    sid = entry[0].get("session_id") or entry[0].get("id")
                    if sid in seen_ids:
                        continue
                    picked.append((stratum, entry))
                    seen_ids.add(sid)
                    taken += 1
                    progressed = True
                    break
            if not progressed:
                break
        return taken

    quotas = [
        ("anchored", args.n),
        ("recovery", args.n_recovery),
        ("parallel", args.n_parallel),
    ]
    for stratum, quota in quotas:
        got = _round_robin(by_stratum_src[stratum], quota, stratum)
        if got < quota:
            print(
                f"WARNING: stratum {stratum}: only {got}/{quota} sessions available",
                file=sys.stderr,
            )

    src_counts: dict[str, dict[str, int]] = {}
    for stratum, (s, _, _) in picked:
        src = s.get("source") or "unknown"
        src_counts.setdefault(stratum, {})[src] = (
            src_counts.setdefault(stratum, {}).get(src, 0) + 1
        )
    print(f"Holdout stratum × source matrix: {src_counts}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for stratum, (s, msgs, tools) in picked:
            rec = {
                "session_id": s.get("session_id") or s.get("id"),
                "source": s.get("source"),
                "stratum": stratum,
                "score": s.get("score"),
                "tools_used": s.get("tools_used", []),
                "tool_call_count": s.get("metrics", {}).get("tool_calls", 0),
                "tools": tools,
                "messages": msgs,
            }
            f.write(json.dumps(rec) + "\n")

    tool_counter: dict[str, int] = {}
    for _, (_, msgs, _) in picked:
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name") or tc.get("name")
                if name:
                    tool_counter[name] = tool_counter.get(name, 0) + 1

    print(f"\nWrote {len(picked)} sessions to {args.output}")
    print("Tool distribution across holdout assistant turns:")
    for name, n in sorted(tool_counter.items(), key=lambda x: -x[1]):
        print(f"  {name:24s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

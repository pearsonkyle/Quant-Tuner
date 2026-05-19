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
    p.add_argument("--n", type=int, default=10)
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

    by_source: dict[str, list[tuple]] = {}
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
        if args.require_anchor and not is_anchored(msgs):
            continue
        src = s.get("source") or "unknown"
        by_source.setdefault(src, []).append((s, msgs, tools))

    print("Candidates per source:", {k: len(v) for k, v in by_source.items()})

    # Stratified round-robin: take equal counts from each source; if a source
    # is exhausted, redistribute its remaining quota across the others.
    rng = random.Random(args.seed)
    for v in by_source.values():
        rng.shuffle(v)

    sources = sorted(by_source.keys())
    picked: list[tuple] = []
    cursors = {s: 0 for s in sources}
    while len(picked) < args.n:
        progressed = False
        for src in sources:
            if len(picked) >= args.n:
                break
            if cursors[src] < len(by_source[src]):
                picked.append(by_source[src][cursors[src]])
                cursors[src] += 1
                progressed = True
        if not progressed:
            break

    if len(picked) < args.n:
        print(f"WARNING: only found {len(picked)} candidates, requested {args.n}", file=sys.stderr)

    src_counts: dict[str, int] = {}
    for s, _, _ in picked:
        src_counts[s.get("source")] = src_counts.get(s.get("source"), 0) + 1
    print(f"Holdout source distribution: {src_counts}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for s, msgs, tools in picked:
            rec = {
                "session_id": s.get("session_id") or s.get("id"),
                "source": s.get("source"),
                "score": s.get("score"),
                "tools_used": s.get("tools_used", []),
                "tool_call_count": s.get("metrics", {}).get("tool_calls", 0),
                "tools": tools,
                "messages": msgs,
            }
            f.write(json.dumps(rec) + "\n")

    tool_counter: dict[str, int] = {}
    for _, msgs, _ in picked:
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

#!/usr/bin/env python3
"""Flatten our sft.jsonl agentic logs into AngelSlim eagle3 conversation JSONL.

AngelSlim's dataset builder (base_dataset_builder._build_messages) keeps only
{role, content}, DROPS turns without a `content` field (our assistant tool_calls
turns), and enforces strict user/assistant alternation. So we pre-render each
conversation into alternating user/assistant turns whose text matches how Qwen3
actually emits tokens at serve time:
  - assistant: optional <think>reasoning</think>, then content, then each
    tool_call as a hermes <tool_call>{json}</tool_call> block.
  - tool results (role tool) are folded into the FOLLOWING user turn as
    <tool_response>...</tool_response> (Qwen3's serve-time rendering).
Consecutive same-role turns are merged; leading assistant turns dropped.

Output rows: {"id": <hash>, "conversations": [{"role","content"}, ...]}.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json


def _tool_calls_text(tool_calls) -> str:
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function", tc)
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        out.append("<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>")
    return "\n".join(out)


def render_conversation(messages: list[dict]) -> list[dict]:
    seq: list[tuple[str, str]] = []
    pending_tool: list[str] = []  # tool_response blocks to prepend to next user turn

    for m in messages:
        role = m.get("role")
        if role == "system":
            c = (m.get("content") or "").strip()
            if c:
                seq.append(("system", c))
            continue
        if role in ("tool", "function"):
            res = (m.get("content") or "").strip()
            if res:
                pending_tool.append("<tool_response>\n" + res + "\n</tool_response>")
            continue
        if role == "assistant":
            parts = []
            if m.get("reasoning_content"):
                parts.append("<think>\n" + m["reasoning_content"].strip() + "\n</think>")
            if m.get("content"):
                parts.append(m["content"].strip() if isinstance(m["content"], str) else str(m["content"]))
            if m.get("tool_calls"):
                parts.append(_tool_calls_text(m["tool_calls"]))
            text = "\n\n".join(p for p in parts if p)
            if text:
                seq.append(("assistant", text))
            continue
        # user (or anything else) -> user; prepend any pending tool results
        c = (m.get("content") or "")
        c = c.strip() if isinstance(c, str) else str(c)
        chunk = "\n\n".join(pending_tool + ([c] if c else []))
        pending_tool = []
        if chunk:
            seq.append(("user", chunk))
    if pending_tool:  # trailing tool results with no following user turn
        seq.append(("user", "\n\n".join(pending_tool)))

    # merge consecutive same-role
    merged: list[list] = []
    for role, text in seq:
        if merged and merged[-1][0] == role:
            merged[-1][1] += "\n\n" + text
        else:
            merged.append([role, text])
    # drop leading assistant turns (after an optional system)
    body_start = 1 if merged and merged[0][0] == "system" else 0
    while len(merged) > body_start and merged[body_start][0] == "assistant":
        merged.pop(body_start)
    # need at least one user + one assistant
    if not any(r == "assistant" for r, _ in merged) or not any(r == "user" for r, _ in merged):
        return []
    return [{"role": r, "content": t} for r, t in merged]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl.gz")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    op = gzip.open if args.data.endswith(".gz") else open
    n = kept = 0
    with op(args.data, "rt") as f, open(args.out, "w") as w:
        for line in f:
            if kept >= args.max_rows:
                break
            d = json.loads(line)
            if d.get("split") != args.split:
                continue
            n += 1
            convs = render_conversation(d["messages"])
            if not convs:
                continue
            total = sum(len(c["content"]) for c in convs)
            if total < args.min_chars:
                continue
            rid = hashlib.sha1(json.dumps(convs).encode()).hexdigest()[:16]
            w.write(json.dumps({"id": rid, "conversations": convs}) + "\n")
            kept += 1
    print(f"read {n} {args.split} convs -> wrote {kept} to {args.out}")


if __name__ == "__main__":
    main()

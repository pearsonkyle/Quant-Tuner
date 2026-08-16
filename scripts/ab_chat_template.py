#!/usr/bin/env python3
"""Byte-diff a candidate chat template against a model's shipped one.

Why byte-identity is the test that matters
------------------------------------------
Qwen models are trained on one exact rendered format, and deviating from it
degrades output quality subtly — enough to move benchmark scores without being
visible in manual testing. So the useful question about a "fixed" template is
not "is it nicer?" but **"does it change the bytes we actually send?"**

- Byte-identical on our traffic  -> the swap cannot change quality. Adopt it for
  its robustness fixes with no behavioural A/B required.
- Any diff                        -> a behavioural A/B is mandatory before
  shipping, and this prints exactly which cases differ and how.

Run over the real tool-call holdout (every prefix that the eval actually sends)
plus synthetic edge cases the thread calls out.

    PYTHONPATH=src .venv/bin/python scripts/ab_chat_template.py \
        --model out/exp-060/model_extracted \
        --candidate data/chat_templates/qwen3_8_safe_v2.jinja \
        --holdout out/exp-060-32k/eval/toolcall_holdout.jinja
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _render(tok, template: str | None, msgs, tools, **kw):
    """Render with an explicit template override, returning text or an error tag."""
    try:
        return tok.apply_chat_template(
            msgs,
            tools=tools,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
    except Exception as exc:  # template raise_exception or a real bug
        return f"<<ERROR {type(exc).__name__}: {str(exc)[:160]}>>"


def edge_cases():
    """Cases the r/LocalLLaMA thread claims the stock template gets wrong."""
    bash = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
    return [
        (
            "json-string arguments (OpenAI wire shape)",
            [
                {"role": "user", "content": "ls"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "a.py", "tool_call_id": "1"},
                {"role": "user", "content": "next?"},
            ],
            [bash],
        ),
        (
            "boolean / null argument values",
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": {"command": "x", "force": True, "opt": None},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "ok", "tool_call_id": "1"},
                {"role": "user", "content": "next?"},
            ],
            [bash],
        ),
        (
            "leading tool message (no preceding user turn)",
            [
                {"role": "tool", "content": "result", "tool_call_id": "1"},
                {"role": "user", "content": "what happened?"},
            ],
            [bash],
        ),
        (
            "content-list item whose text contains the word image",
            [{"role": "user", "content": [{"type": "text", "text": "describe this image file"}]}],
            None,
        ),
        (
            "no tools, plain chat",
            [{"role": "user", "content": "hello"}],
            None,
        ),
        (
            "assistant with reasoning_content in history",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey", "reasoning_content": "thinking..."},
                {"role": "user", "content": "again"},
            ],
            None,
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--holdout", type=Path, default=None)
    ap.add_argument("--max-prefixes-per-session", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model))
    stock = tok.chat_template
    cand = args.candidate.read_text()

    same = diff = 0
    diffs: list[tuple[str, str, str]] = []

    def compare(label, msgs, tools, **kw):
        nonlocal same, diff
        a = _render(tok, stock, msgs, tools, **kw)
        b = _render(tok, cand, msgs, tools, **kw)
        if a == b:
            same += 1
        else:
            diff += 1
            if len(diffs) < 6:
                diffs.append((label, a, b))

    print("=== edge cases from the thread ===")
    for label, msgs, tools in edge_cases():
        a = _render(tok, stock, msgs, tools)
        b = _render(tok, cand, msgs, tools)
        verdict = "IDENTICAL" if a == b else "DIFFERS"
        print(f"  {verdict:9s}  {label}")
        if a != b:
            print(f"      stock    : {a[-150:]!r}" if not a.startswith("<<ERROR") else f"      stock    : {a}")
            print(f"      candidate: {b[-150:]!r}" if not b.startswith("<<ERROR") else f"      candidate: {b}")

    if args.holdout and args.holdout.exists():
        print("\n=== real holdout prefixes (what the eval actually sends) ===")
        sessions = [json.loads(line) for line in open(args.holdout)]
        for sess in sessions:
            msgs, tools = sess["messages"], sess["tools"]
            seen = 0
            for i, m in enumerate(msgs):
                if m["role"] != "assistant" or not m.get("tool_calls"):
                    continue
                seen += 1
                if seen > args.max_prefixes_per_session:
                    break
                compare(f"{sess.get('session_id')}#{i}", msgs[:i], tools)
                # also compare the full history INCLUDING the tool-call turn,
                # which is what exercises tool-call rendering itself
                compare(f"{sess.get('session_id')}#{i}+call", msgs[: i + 1], tools)
        total = same + diff
        print(f"  byte-identical : {same}/{total}")
        print(f"  differing      : {diff}/{total}")
        for label, a, b in diffs:
            print(f"\n  DIFF {label}")
            print(f"    stock    ...{a[-200:]!r}")
            print(f"    candidate...{b[-200:]!r}")

    print(
        "\nVERDICT: "
        + (
            "byte-identical on real traffic — swap cannot change quality"
            if diff == 0
            else f"{diff} rendering differences — behavioural A/B required before shipping"
        )
    )


if __name__ == "__main__":
    main()

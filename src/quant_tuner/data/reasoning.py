"""Normalizing assistant *reasoning* across sources, and knowing what the template keeps.

Reasoning arrives in two shapes and neither is what a chat template consumes uniformly:

* ``logs-cli.jsonl.gz`` — inline in ``content`` as ``<think>…</think>`` (501 messages).
* ``logs-agents.jsonl.gz`` — a separate ``reasoning_content`` field (3,906 messages).

**Measured on the Qwen3.6 template** (`out/exp-041/model_extracted`), and the reason this
module exists:

| where | `reasoning_content` field | inline `<think>` in content |
|---|---|---|
| final assistant turn of the render | kept, as ``<think>\\n…\\n</think>\\n\\n`` | kept verbatim |
| any earlier assistant turn | **dropped** | **dropped** |

So a naive corpus keeps ~one reasoning block per rendered window and silently discards the
rest — even though reasoning is where a thinking model spends most of its output tokens, and
therefore a mode the importance matrix should see. That is not a bug to fix by fighting the
template: history-scrubbing is exactly what happens at inference, so the *rendered* form is
right. What matters is that we (a) normalize both source shapes so the behavior doesn't
depend on which file a session came from, and (b) **count what actually survived** into the
corpus instead of assuming.

:func:`apply_policy` does (a); :func:`count_reasoning_blocks` does (b), and
``universal.build`` records it per source next to the tool-call scan.
"""

from __future__ import annotations

import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
# Reasoning is only ever emitted at the START of an assistant turn — that is how the model
# generates it and how templates render it. Anchoring here is not pedantry: these are coding
# logs, so assistant turns *discuss* `<think>` and `</think>` in prose, and an unanchored
# match happily spans from a backticked `<think>` to a `</think>` mentioned three sentences
# later and deletes the explanation in between. Caught on the real logs (3 mangled turns).
_LEADING_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)
# Truncated generation: a leading block that never closed. The whole turn is reasoning, and
# leaving the naked marker in `content` would put an unbalanced control token in the data.
_UNCLOSED_THINK = re.compile(r"^\s*<think>(.*)$", re.DOTALL)
# The mirror image: capture that began mid-reasoning, so the turn opens with the tail of a
# thought and a closing tag it never opened. Everything up to that tag is the reasoning.
_ORPHAN_CLOSE = re.compile(r"^((?:(?!<think>).)*?)</think>\s*", re.DOTALL)

POLICIES = ("auto", "field", "drop")


def strip_inline(content: str) -> tuple[str, str]:
    """Split a LEADING ``<think>…</think>`` off a content string -> ``(reasoning, rest)``.

    Only a leading block counts; ``<think>`` appearing mid-message is ordinary content that
    happens to mention the marker, and is left alone.
    """
    if not isinstance(content, str):
        return "", content
    if THINK_OPEN in content:
        if (m := _LEADING_THINK.match(content)) is not None:
            return m.group(1).strip(), content[m.end():]
        if (m := _UNCLOSED_THINK.match(content)) is not None:
            return m.group(1).strip(), ""
        return "", content
    if THINK_CLOSE in content and (m := _ORPHAN_CLOSE.match(content)) is not None:
        return m.group(1).strip(), content[m.end():]
    return "", content


def reasoning_of(message: dict) -> str:
    """The message's reasoning, from either shape (field first, then inline)."""
    field = message.get("reasoning_content")
    if isinstance(field, str) and field.strip():
        return field.strip()
    inline, _ = strip_inline(message.get("content") or "")
    return inline


def apply_policy(messages: list[dict], policy: str = "auto") -> list[dict]:
    """Return messages with reasoning normalized. Never mutates the input.

    * ``auto`` (default) — everything becomes inline ``<think>…</think>`` in ``content``, and
      the ``reasoning_content`` field is removed. One shape for every source, and one that
      survives verbatim wherever a template keeps reasoning at all.
    * ``field`` — the inverse: reasoning lives in ``reasoning_content``, content is clean.
      For templates that render the field and scrub inline markers.
    * ``drop`` — no reasoning anywhere. Use for a non-thinking target model, where
      ``<think>`` spans are tokens it will never emit.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown reasoning policy {policy!r}; expected one of {POLICIES}")

    out: list[dict] = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        rsn = reasoning_of(m)
        _, body = strip_inline(m.get("content") or "")
        new = {k: v for k, v in m.items() if k != "reasoning_content"}
        if not rsn or policy == "drop":
            new["content"] = body
        elif policy == "field":
            new["content"] = body
            new["reasoning_content"] = rsn
        else:  # auto -> inline
            new["content"] = f"{THINK_OPEN}\n{rsn}\n{THINK_CLOSE}\n\n{body}" if body \
                else f"{THINK_OPEN}\n{rsn}\n{THINK_CLOSE}"
        out.append(new)
    return out


def count_reasoning_blocks(text: str, *, nonempty_only: bool = True) -> int:
    """How many reasoning blocks survived into a rendered corpus.

    ``nonempty_only`` (the default) is the number that means something. Qwen-family templates
    emit an **empty** ``<think>\\n\\n</think>`` on the final assistant turn of every render
    when no reasoning is supplied, so a raw ``</think>`` count is ~1 per window regardless of
    whether any real reasoning made it in — it would have reported healthy coverage on a
    corpus containing none.
    """
    if not nonempty_only:
        return text.count(THINK_CLOSE)
    return sum(1 for m in _THINK_RE.finditer(text) if m.group(1).strip())


def count_available(messages: list[dict]) -> int:
    """How many assistant turns carry reasoning *before* templating — the denominator."""
    return sum(1 for m in messages
               if m.get("role") == "assistant" and reasoning_of(m))


__all__ = [
    "POLICIES",
    "THINK_CLOSE",
    "THINK_OPEN",
    "apply_policy",
    "count_available",
    "count_reasoning_blocks",
    "reasoning_of",
    "strip_inline",
]

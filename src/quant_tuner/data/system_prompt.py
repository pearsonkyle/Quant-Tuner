"""Scrubbing agent-harness boilerplate out of system prompts, without losing repo context.

Measured on the on-disk logs: 688 sessions carry a system prompt, 6.4M characters in total,
and **90% of those characters sit in blocks repeated verbatim across ≥10 sessions** — tone
guidelines, git etiquette, safety preambles, worked examples. For calibration that mostly
wastes window budget (the packer already stubs it). For SFT it is worse than waste: the
model is trained to re-read the same 24k-token preamble thousands of times, which teaches it
nothing and crowds out real work.

What must survive is the part that says something about *this* session: the repo, the
working directory, the project's own conventions file, the paths in play.

**Frequency alone can't separate the two**, and neither can keywords. The harness blocks are
full of the words "repository", "file paths", even example filenames like `someFile.ts` — a
keyword veto keeps nearly all of them. And genuinely useful project context (a CLAUDE.md
pasted into the prompt) is *also* repeated, across every session in that project.

The discriminator used here is **grounding**: a block is kept when it mentions a concrete
path or filename that actually appears elsewhere in the same conversation. A harness talking
about `someFile.ts` in the abstract is dropped; a prompt naming `src/quant_tuner/config.py`
in a session that goes on to edit that file is kept. Rare blocks (below the boilerplate
threshold) are kept regardless — they are session-specific by construction.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

# A concrete path or filename. Deliberately NOT bare identifiers: the harness blocks are full
# of tool names like `read_file`, and matching those would keep everything.
_PATH_TOKEN = re.compile(
    r"(?:/[\w.+-]+){2,}/?"                                   # /abs/path/like/this
    r"|(?:[\w.+-]+/)+[\w.+-]+\.\w{1,6}\b"                    # rel/path/to/file.ext
    r"|\b[\w.+-]+\.(?:py|js|jsx|ts|tsx|go|rs|rb|java|kt|swift|c|h|cc|cpp|hpp|cs|php|ex|exs"
    r"|clj|lua|r|scala|dart|ml|sh|bash|zsh|sql|md|json|toml|yaml|yml|ini|cfg|lock)\b"
)

_URL = re.compile(r"\bhttps?://\S+", re.I)

DEFAULT_MIN_SESSIONS = 4
# A path token that shows up all over the corpus grounds nothing: `package.json`, `CLAUDE.md`
# and `Cargo.toml` appear in most repos, and `Node.js` / `Next.js` are library names the
# filename pattern can't tell from source files. Measured before this filter, those five
# tokens alone were keeping ~170 harness blocks alive. Anything above this share of sessions
# is treated as generic vocabulary rather than a reference to *this* project.
DEFAULT_MAX_GROUND_DF = 0.02
DEFAULT_MIN_GENERIC_DF = 5
_MAX_LEAD_BLOCK_CHARS = 400


def split_blocks(content: str) -> list[str]:
    """Blank-line-separated blocks, whitespace-trimmed, empties dropped."""
    if not isinstance(content, str):
        return []
    return [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]


def path_tokens(text: str) -> set[str]:
    """Concrete path/filename tokens mentioned in ``text``.

    URLs are blanked first: a harness block linking to
    ``https://github.com/anthropics/claude-code/issues`` otherwise looks exactly like an
    absolute path, and that one link was enough to keep a generic block alive.
    """
    return {m.group(0).rstrip("/.,:;)")
            for m in _PATH_TOKEN.finditer(_URL.sub(" ", text or ""))}


def system_content_of(messages: list[dict]) -> str | None:
    if messages and messages[0].get("role") == "system":
        c = messages[0].get("content")
        return c if isinstance(c, str) else None
    return None


def body_text(messages: list[dict]) -> str:
    """Everything except the system turn — what the block has to be grounded in."""
    import json as _json

    parts: list[str] = []
    for m in messages[1:] if system_content_of(messages) is not None else messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        for tc in m.get("tool_calls") or []:
            parts.append(_json.dumps(tc, default=str))
    return "\n".join(parts)


def boilerplate_blocks(
    system_prompts: Iterable[str], min_sessions: int = DEFAULT_MIN_SESSIONS,
) -> set[str]:
    """Blocks appearing in at least ``min_sessions`` distinct sessions.

    Counted once per session, so a block repeated ten times inside one long prompt does not
    look like corpus-wide boilerplate.
    """
    counts: Counter = Counter()
    for prompt in system_prompts:
        counts.update(set(split_blocks(prompt)))
    return {b for b, n in counts.items() if n >= min_sessions}


def generic_path_tokens(
    bodies: Iterable[str],
    max_df_ratio: float = DEFAULT_MAX_GROUND_DF,
    min_df: int = DEFAULT_MIN_GENERIC_DF,
) -> set[str]:
    """Path tokens common enough across the corpus to be vocabulary, not project context."""
    counts: Counter = Counter()
    n = 0
    for body in bodies:
        n += 1
        counts.update(path_tokens(body))
    threshold = max(min_df, int(n * max_df_ratio))
    return {tok for tok, df in counts.items() if df >= threshold}


def scrub(
    content: str,
    *,
    boilerplate: set[str],
    grounding: str,
    generic: set[str] | None = None,
    keep_lead: bool = True,
) -> tuple[str, dict]:
    """Drop ungrounded boilerplate blocks. Returns ``(scrubbed_content, stats)``.

    ``grounding`` is the rest of the conversation; a boilerplate block survives when it names
    a path or filename that also occurs there. ``keep_lead`` preserves the opening block (the
    identity line — "You are Claude Code…"), truncated to its first line when it is long, so
    the persona the assistant is answering as does not silently change.
    """
    blocks = split_blocks(content)
    if not blocks:
        return content, {"blocks": 0, "dropped": 0, "chars_before": len(content or ""),
                         "chars_after": len(content or "")}

    ground = path_tokens(grounding) - (generic or set())
    kept: list[str] = []
    dropped = 0
    for i, b in enumerate(blocks):
        if b not in boilerplate:
            kept.append(b)                       # rare => session-specific by construction
            continue
        if ground & path_tokens(b):
            kept.append(b)                       # names something this session touches
            continue
        if i == 0 and keep_lead:
            lead = b if len(b) <= _MAX_LEAD_BLOCK_CHARS else b.splitlines()[0].strip()
            kept.append(lead)
            dropped += 1 if lead != b else 0
            continue
        dropped += 1

    out = "\n\n".join(kept)
    return out, {
        "blocks": len(blocks),
        "dropped": dropped,
        "chars_before": len(content),
        "chars_after": len(out),
    }


def scrub_messages(
    messages: list[dict], *, boilerplate: set[str], generic: set[str] | None = None,
    keep_lead: bool = True,
) -> tuple[list[dict], dict]:
    """:func:`scrub` applied to a conversation's system turn. Never mutates the input."""
    content = system_content_of(messages)
    if content is None or not boilerplate:
        return messages, {"blocks": 0, "dropped": 0,
                          "chars_before": len(content or ""),
                          "chars_after": len(content or "")}
    scrubbed, stats = scrub(content, boilerplate=boilerplate,
                            grounding=body_text(messages), generic=generic,
                            keep_lead=keep_lead)
    if not scrubbed:                    # never leave an empty system turn behind
        return messages[1:], stats
    return [dict(messages[0], content=scrubbed), *messages[1:]], stats


__all__ = [
    "DEFAULT_MIN_SESSIONS",
    "DEFAULT_MAX_GROUND_DF",
    "body_text",
    "boilerplate_blocks",
    "generic_path_tokens",
    "path_tokens",
    "scrub",
    "scrub_messages",
    "split_blocks",
    "system_content_of",
]

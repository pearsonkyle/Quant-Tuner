"""Encode/decode the Gemma 4 chat template's tool-call wire format.

The template does NOT emit JSON. Reading it off `chat_template.jinja`
(`format_argument`, and the tool-call block at line 244):

    call        <|tool_call>call:NAME{k:v,k:v}<tool_call|>
    thought     <|channel>thought\\n...\\n<channel|>
    string      <|"|>text<|"|>          <- delimited, NOT escaped
    null        null
    bool        true / false
    object      {k:v,...}               <- bare keys inside a call, quoted elsewhere
    array       [v,v,...]
    number      bare

Two consequences drive this module:

1. Strings carry no escape mechanism, so a value containing the delimiter
   cannot round-trip. We scan to the next delimiter and accept that limit
   rather than pretending to handle it.
2. Keys are emitted `| dictsort`, so argument order on the wire is
   alphabetical and says nothing about the order the caller supplied.

There is also a trap on the encode side. The template *raises* if
`tool_calls[].function.arguments` is a string:

    "chat_template: tool_calls[].function.arguments must be a JSON object
     (mapping), not a string. Deserialize arguments before passing."

But the OpenAI wire format -- and `quant_tuner.eval.toolcall.strip_for_api`,
which builds every replayed prefix -- puts a JSON *string* there, correctly.
So anything feeding API messages into this template must call
`deserialize_tool_arguments` first or every session with a tool call dies at
render time.
"""

from __future__ import annotations

import json
from typing import Any

STR = "<|\"|>"
CALL_OPEN = "<|tool_call>call:"
CALL_CLOSE = "<tool_call|>"
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"


class WireError(ValueError):
    """Raised when a generation does not parse as the template's format."""


# --------------------------------------------------------------------------
# encode side
# --------------------------------------------------------------------------

def deserialize_tool_arguments(messages: list[dict]) -> list[dict]:
    """Return `messages` with every tool_call's `arguments` as a mapping.

    Required before `apply_chat_template`; see the module docstring.
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = dict(m)
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                elif args is None:
                    fn["arguments"] = {}
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


# --------------------------------------------------------------------------
# decode side
# --------------------------------------------------------------------------

class _Reader:
    def __init__(self, s: str, i: int = 0):
        self.s, self.i = s, i

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def starts(self, tok: str) -> bool:
        return self.s.startswith(tok, self.i)

    def value(self, bare_keys: bool) -> Any:
        if self.starts(STR):
            self.i += len(STR)
            end = self.s.find(STR, self.i)
            if end < 0:
                raise WireError("unterminated string")
            v = self.s[self.i:end]
            self.i = end + len(STR)
            return v
        if self.starts("{"):
            self.i += 1
            obj: dict[str, Any] = {}
            if self.starts("}"):
                self.i += 1
                return obj
            while True:
                key = self._key(bare_keys)
                if not self.starts(":"):
                    raise WireError(f"expected ':' after key {key!r}")
                self.i += 1
                obj[key] = self.value(bare_keys)
                if self.starts(","):
                    self.i += 1
                    continue
                if self.starts("}"):
                    self.i += 1
                    return obj
                raise WireError("expected ',' or '}' in object")
        if self.starts("["):
            self.i += 1
            arr: list[Any] = []
            if self.starts("]"):
                self.i += 1
                return arr
            while True:
                arr.append(self.value(bare_keys))
                if self.starts(","):
                    self.i += 1
                    continue
                if self.starts("]"):
                    self.i += 1
                    return arr
                raise WireError("expected ',' or ']' in array")
        for lit, val in (("null", None), ("true", True), ("false", False)):
            if self.starts(lit):
                self.i += len(lit)
                return val
        # bare number, or anything else up to a structural delimiter
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ",}]":
            self.i += 1
        raw = self.s[start:self.i].strip()
        if raw == "":
            raise WireError("empty value")
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    def _key(self, bare: bool) -> str:
        if self.starts(STR):
            self.i += len(STR)
            end = self.s.find(STR, self.i)
            if end < 0:
                raise WireError("unterminated key")
            k = self.s[self.i:end]
            self.i = end + len(STR)
            return k
        if not bare:
            raise WireError("expected a quoted key")
        start = self.i
        while self.i < len(self.s) and self.s[self.i] != ":":
            self.i += 1
        return self.s[start:self.i]


def parse_generation(text: str) -> dict:
    """Split a raw generation into content / reasoning_content / tool_calls.

    `tool_calls` come back in OpenAI shape, with `arguments` as a JSON STRING,
    which is what an OpenAI-compatible endpoint must return.
    """
    reasoning = None
    truncated_thought = False
    t0 = text.find(THOUGHT_OPEN)
    if t0 >= 0:
        body = t0 + len(THOUGHT_OPEN)
        t1 = text.find(THOUGHT_CLOSE, body)
        if t1 >= 0:
            reasoning = text[body:t1].strip("\n")
            text = text[:t0] + text[t1 + len(THOUGHT_CLOSE):]
        else:
            # Generation hit max_tokens mid-thought. Everything after the open
            # marker is reasoning, not an answer -- leaving it in `content`
            # would score a truncation as if the model had chosen to reply in
            # prose instead of calling a tool, which is a different failure.
            reasoning = text[body:].strip("\n")
            text = text[:t0]
            truncated_thought = True

    calls, out, i, n = [], [], 0, 0
    while True:
        c0 = text.find(CALL_OPEN, i)
        if c0 < 0:
            out.append(text[i:])
            break
        out.append(text[i:c0])
        brace = text.find("{", c0 + len(CALL_OPEN))
        if brace < 0:
            raise WireError("tool call without an argument object")
        name = text[c0 + len(CALL_OPEN):brace]
        r = _Reader(text, brace)
        args = r.value(bare_keys=True)
        end = text.find(CALL_CLOSE, r.i)
        i = (end + len(CALL_CLOSE)) if end >= 0 else r.i
        calls.append({
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": name.strip(),
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
        n += 1

    content = "".join(out).strip()
    return {"content": content or None,
            "reasoning_content": reasoning,
            "truncated_thought": truncated_thought,
            "tool_calls": calls}

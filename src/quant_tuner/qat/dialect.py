"""Chat dialects: where the supervised span is, per model family.

``corpus.py`` used to hard-code Qwen's answer — an assistant turn is
``<|im_start|>assistant\\n … <|im_end|>``, found by regex over the rendered text.
That is correct for Qwen and wrong for gemma-4 in a way no aggregate statistic
reveals, which is why the rule lives here now instead of in a module-level regex.

**gemma-4 renders an entire tool-calling exchange as ONE ``model`` turn**, with the
environment's tool results embedded inside it::

    <|turn>model
    <|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>
    <|tool_response>response:bash{value:<|"|>a.py<|"|>}<tool_response|>Looking.Done.<turn|>

Port Qwen's rule by swapping the markers — ``r"<\\|turn>model\\n(.*?)<turn\\|>"`` — and
the ``<|tool_response>…<tool_response|>`` block falls *inside* the supervised span, so
the model trains to generate tool output it will never generate. The loss still falls,
the density audit still looks healthy, and nothing downstream can tell.

So gemma-4's rule is expressed on **token ids**, where the structure is exactly regular
(verified against the tokenizer, see ``tests/unit/test_qat_dialect.py``):

===============================  ====
``<|turn>``                       105
``<turn|>`` (the stop decision)   106
``model`` / ``user`` / ``system`` 4368 / 2364 / 9731
``\\n``                            107
``<|tool_call>`` / ``<tool_call|>``      48 / 49
``<|tool_response>`` / ``<tool_response|>``  50 / 51
===============================  ====

Supervise from just after the ``<|turn>model\\n`` header (3 ids) through the terminating
``<turn|>`` **inclusive** — the terminator is the stop decision, and omitting it is what
caused the iter-2/3 looping on the Qwen side — minus every ``[50 … 51]`` span. The
header stays context: at inference it is part of the generation prompt and is never
generated. The ``\\n`` after ``<turn|>`` stays masked, matching the Qwen convention.

The tool-CALL markers stay supervised (the model emits those); the tool-RESPONSE block
does not (the harness injects it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Qwen: assistant span from the header through the terminating ``<|im_end|>``.
_QWEN_ASST_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)


@dataclass(frozen=True)
class ChatDialect:
    """How to find the supervised tokens in one rendered conversation."""

    name: str
    #: The end-of-turn token. Its id is what ``--stop-weight`` up-weights, and it must be
    #: a labeled target or no position ever trains the stop decision.
    stop_piece: str
    #: Control tokens that must never appear inside message CONTENT — a log that quotes
    #: one gets a REAL control token in the middle of supervised prose.
    control_tokens: tuple[str, ...]

    def labels(self, ids: list[int], text: str, offsets, tok) -> list[int]:
        raise NotImplementedError


@dataclass(frozen=True)
class QwenChatDialect(ChatDialect):
    """Qwen2/3 ChatML. Character-span rule, kept verbatim so published corpora are
    bit-identical — their fingerprints are recorded in `docs/qat_run_history.md`."""

    name: str = "qwen"
    stop_piece: str = "<|im_end|>"
    control_tokens: tuple[str, ...] = ("<|im_start|>", "<|im_end|>")

    def labels(self, ids: list[int], text: str, offsets, tok) -> list[int]:
        # char spans: assistant *content* plus the terminating <|im_end|> (m.end(0))
        spans = [(m.start(1), m.end(0)) for m in _QWEN_ASST_RE.finditer(text)]
        labels = [-100] * len(ids)
        si = 0
        for j, (a, b) in enumerate(offsets):
            if a == b:  # special/zero-width
                continue
            while si < len(spans) and spans[si][1] <= a:
                si += 1
            if si < len(spans) and a >= spans[si][0] and b <= spans[si][1]:
                labels[j] = ids[j]
        return labels


@dataclass(frozen=True)
class Gemma4ChatDialect(ChatDialect):
    """gemma-4. Id-based rule — see the module docstring for why not a regex."""

    name: str = "gemma4"
    stop_piece: str = "<turn|>"
    control_tokens: tuple[str, ...] = ("<|turn>", "<turn|>", "<|tool_call>",
                                       "<tool_call|>", "<|tool_response>",
                                       "<tool_response|>", "<|channel>", "<channel|>")
    turn_open: int = 105
    turn_close: int = 106
    role_model: int = 4368
    newline: int = 107
    tool_response_open: int = 50
    tool_response_close: int = 51

    def labels(self, ids: list[int], text: str, offsets, tok) -> list[int]:
        labels = [-100] * len(ids)
        i, n = 0, len(ids)
        while i < n:
            # a model turn opens with exactly (<|turn>, 'model', '\n')
            if (ids[i] == self.turn_open and i + 2 < n
                    and ids[i + 1] == self.role_model and ids[i + 2] == self.newline):
                j = i + 3                     # header is context, never generated
                in_response = False
                while j < n:
                    t = ids[j]
                    # Checked BEFORE labelling: a truncated render whose model turn never
                    # closes must not supervise the next turn's header, least of all the
                    # <|turn> token itself.
                    if t == self.turn_open:   # malformed render; do not run past it
                        j -= 1
                        break
                    if t == self.tool_response_open:
                        in_response = True    # environment text from here...
                    if not in_response:
                        labels[j] = t
                    if t == self.tool_response_close:
                        in_response = False   # ...through the closing marker inclusive
                    if t == self.turn_close:  # the stop decision — supervised, then done
                        break
                    j += 1
                i = j + 1
                continue
            i += 1
        return labels


#: Registry. ``detect`` picks by vocabulary rather than by model name, because the name
#: is whatever a finetune called itself and the vocabulary is what actually renders.
DIALECTS: dict[str, ChatDialect] = {d.name: d for d in
                                    (QwenChatDialect(), Gemma4ChatDialect())}


def detect(tok) -> ChatDialect:
    """Pick the dialect whose stop token this tokenizer actually has.

    A missing token resolves to ``<unk>`` rather than raising, so the check is
    "resolves to a real id AND round-trips to the same string" — gemma-4's vocabulary
    maps ``<|im_end|>`` to nothing, and Qwen's maps ``<turn|>`` to nothing, but each
    returns *an* id.
    """
    for d in (Gemma4ChatDialect(), QwenChatDialect()):
        i = tok.convert_tokens_to_ids(d.stop_piece)
        if i is not None and i >= 0 and tok.convert_ids_to_tokens(i) == d.stop_piece:
            return d
    raise ValueError(
        "no known chat dialect for this tokenizer (looked for "
        f"{[d.stop_piece for d in DIALECTS.values()]}); add one to qat/dialect.py")

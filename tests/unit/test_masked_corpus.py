"""Tests for the masked QAT corpus builder (scripts/build_qat_masked_corpus.py).

The critical regression these tests pin: the terminating ``<|im_end|>`` of every
assistant turn must be a TRAINED label. The original span logic used regex
group(1) (content only), so under HF's causal shift no position in the corpus
ever had the stop token as a CE target — the model was never trained to end its
turn (the mechanistic cause of the looping pathology in iter-2/iter-3).

Uses a small offline fake tokenizer (character-offset faithful) — no downloads.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from quant_tuner.qat import corpus as bqmc  # noqa: E402

SPECIALS = ("<|im_start|>", "<|im_end|>")


class FakeTok:
    """Whitespace/special-token tokenizer with faithful char offsets."""

    def __init__(self):
        # Pre-seed the ChatML control tokens. `qat.dialect.detect` reads the vocabulary to
        # decide which family's span rule applies, and it runs before any text has been
        # tokenized — an empty vocab looks like a tokenizer from no known family at all.
        self.vocab: dict[str, int] = {sp: i + 10 for i, sp in enumerate(SPECIALS)}

    def _spans(self, text: str) -> list[tuple[str, int, int]]:
        spans, i = [], 0
        while i < len(text):
            for sp in SPECIALS:
                if text.startswith(sp, i):
                    spans.append((sp, i, i + len(sp)))
                    i += len(sp)
                    break
            else:
                if text[i].isspace():
                    spans.append((text[i], i, i + 1))
                    i += 1
                else:
                    j = i
                    while (j < len(text) and not text[j].isspace()
                           and not any(text.startswith(sp, j) for sp in SPECIALS)):
                        j += 1
                    spans.append((text[i:j], i, j))
                    i = j
        return spans

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab) + 10
        return self.vocab[tok]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = self._spans(text)
        out = {"input_ids": [self._id(t) for t, _, _ in spans]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(a, b) for _, a, b in spans]
        return out

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return "".join(rev[i] for i in ids)

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        # Deliberately does NOT mint: `_id` invents an id for anything it is asked about,
        # so a minting lookup would answer "yes, I have <turn|>" and detection would pick
        # the gemma dialect for a ChatML tokenizer.
        return self.vocab.get(tok)

    def convert_ids_to_tokens(self, i: int) -> str:
        return {v: k for k, v in self.vocab.items()}.get(i, "<unk>")

    def apply_chat_template(self, msgs, tools=None, tokenize=False,
                            add_generation_prompt=False):
        parts = []
        if tools:
            parts.append("<|im_start|>system\n# Tools\n"
                         + json.dumps(tools) + "<|im_end|>\n")
        for m in msgs:
            if m["role"] == "tool":
                parts.append("<|im_start|>user\n<tool_response>"
                             + str(m.get("content")) + "</tool_response><|im_end|>\n")
                continue
            body = m.get("content") or ""
            for tc in (m.get("tool_calls") or []):
                body += "\n<tool_call>" + json.dumps(tc.get("function", tc)) + "</tool_call>"
            parts.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(parts)


MSGS = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "running it",
     "tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
    {"role": "tool", "content": "a.txt b.txt"},
    {"role": "assistant", "content": "done listing"},
]


def _labeled_tokens(tok, ids, labels):
    rev = {v: k for k, v in tok.vocab.items()}
    return [(rev[i], lab != -100) for i, lab in zip(ids, labels, strict=True)]


def test_im_end_labeled_once_per_assistant_turn():
    tok = FakeTok()
    ids, labels = bqmc.masked_ids_for_session(MSGS, tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    labeled_im_end = sum(1 for i, lab in zip(ids, labels, strict=True) if lab != -100 and i == im_end)
    assert labeled_im_end == 2  # one per assistant turn, none elsewhere
    # and after the HF causal shift, the stop token IS a target somewhere
    shift_targets = [labels[t + 1] for t in range(len(labels) - 1)]
    assert im_end in [t for t in shift_targets if t != -100]


def test_header_and_context_stay_masked():
    tok = FakeTok()
    ids, labels = bqmc.masked_ids_for_session(MSGS, tok)
    toks = _labeled_tokens(tok, ids, labels)
    # <|im_start|> and role headers never trained; the newline right after a
    # labeled <|im_end|> never trained; user/system/tool content never trained
    for tstr, labeled in toks:
        if tstr in ("<|im_start|>", "system", "user", "helpful", "files",
                    "<tool_response>a.txt", "b.txt</tool_response>"):
            assert not labeled, f"{tstr!r} must stay context"
    # the token *after* each labeled <|im_end|> is the turn separator '\n'
    for j, (tstr, labeled) in enumerate(toks[:-1]):
        if tstr == "<|im_end|>" and labeled:
            assert toks[j + 1][0] == "\n" and not toks[j + 1][1]


def test_assistant_content_and_tool_calls_are_labeled():
    tok = FakeTok()
    ids, labels = bqmc.masked_ids_for_session(MSGS, tok)
    toks = _labeled_tokens(tok, ids, labels)
    labeled = {t for t, on in toks if on}
    assert "running" in labeled and "done" in labeled
    assert any("tool_call" in t for t in labeled)  # <tool_call> markup is trained


def test_tool_message_truncation_head_tail():
    tok = FakeTok()
    long_content = " ".join(f"w{i}" for i in range(100))
    msgs = [{"role": "tool", "content": long_content},
            {"role": "tool", "content": "short"}]
    out, n_trunc, saved = bqmc.truncate_tool_messages(msgs, tok, max_tool_tokens=10)
    assert n_trunc == 1 and saved > 0
    assert out[0]["content"].startswith("w0")
    assert out[0]["content"].rstrip().endswith("w99")
    assert "truncated" in out[0]["content"]
    assert out[1]["content"] == "short"  # short messages untouched
    same, n0, s0 = bqmc.truncate_tool_messages(msgs, tok, max_tool_tokens=0)
    assert same is msgs and n0 == 0 and s0 == 0


def test_pack_min_trainable_and_density():
    window = 16
    ids = list(range(window * 4))
    # window0: 0 trainable; window1: 4 trainable; window2: 8 trainable; window3: 16
    lbl = [-100] * window
    lbl += [1] * 4 + [-100] * (window - 4)
    lbl += [1] * 8 + [-100] * (window - 8)
    lbl += [1] * window
    assert len(bqmc.pack(ids, lbl, window, min_trainable=8)) == 2
    assert len(bqmc.pack(ids, lbl, window, min_trainable=1)) == 3
    assert len(bqmc.pack(ids, lbl, window, min_trainable=1, min_density=0.75)) == 1
    # trailing partial window is dropped
    assert len(bqmc.pack(ids + [0] * 5, lbl + [1] * 5, window, min_trainable=1)) == 3


def test_reconstruct_tools_fallback_shape():
    tools = bqmc.reconstruct_tools(MSGS)
    assert tools == [{"type": "function", "function": {
        "name": "bash", "description": "bash tool",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]


def test_real_schemas_take_precedence_over_stub():
    from quant_tuner.data.split import session_tools
    real = [{"type": "function", "function": {"name": "bash", "description": "real",
                                              "parameters": {"type": "object"}}}]
    msgs = [{"role": "system", "content": "x", "tools": real}] + MSGS[1:]
    assert session_tools({"messages": msgs}, msgs) == real
    assert session_tools({"messages": MSGS}, MSGS) is None  # -> stub fallback fires

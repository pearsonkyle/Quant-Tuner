"""The chat-template tool-call pre-flight, exercised against stand-in templates.

No model files: a tiny fake tokenizer renders jinja-ish text so we can assert that each
failure mode we have actually been bitten by is *caught*, not just that a good template
passes.
"""

from __future__ import annotations

import json

import pytest

from quant_tuner.data import template_check as tc

QWEN_CALL = "<tool_call>\n{name}\n</tool_call>"


class FakeTokenizer:
    """Minimal stand-in: `render(messages, tools) -> str` plus whitespace tokenization."""

    chat_template = "fake"
    name_or_path = "fake/model"
    all_special_tokens = ["<tool_call>", "</tool_call>", "<tool_response>"]

    def __init__(self, render, specials_multi_id=()):
        self._render = render
        self._specials_multi_id = set(specials_multi_id)

    # -- transformers surface used by the checker + split.session_windows --
    def apply_chat_template(self, messages, tools=None, tokenize=False):
        return self._render(messages, tools)

    def __call__(self, text, add_special_tokens=False):
        if text in self._specials_multi_id:
            return {"input_ids": [1, 2, 3]}
        if text in self.all_special_tokens:
            return {"input_ids": [7]}
        return {"input_ids": list(range(max(1, len(text.split()))))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def good_render(messages, tools):
    """A faithful Qwen-style render: schemas, structured calls, wrapped results."""
    out = []
    for t in tools or []:
        fn = t.get("function", t)
        out.append("# Tools\n" + json.dumps(fn))
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m.get('content') or ''}")
        for call in m.get("tool_calls") or []:
            out.append(QWEN_CALL.format(name=json.dumps(call["function"])))
        if m["role"] == "tool":
            out.append(f"<tool_response>\n{m.get('content')}\n</tool_response>")
        out.append("<|im_end|>")
    return "\n".join(out)


def test_faithful_template_passes():
    rep = tc.check_template(FakeTokenizer(good_render))
    assert rep.ok, rep.summary()
    assert rep.tool_call_marker == "<tool_call>"
    assert rep.tool_response_marker == "<tool_response>"


def test_template_that_drops_tools_argument_fails():
    """The failure that produces a corpus with no schema conditioning at all."""
    rep = tc.check_template(FakeTokenizer(lambda m, t: good_render(m, None)))
    assert not rep.ok
    failed = {c.name for c in rep.failures}
    assert "tool_schemas_rendered" in failed
    assert "tool_parameters_rendered" in failed


def test_template_that_drops_tool_calls_fails():
    def render(messages, tools):
        stripped = [{k: v for k, v in m.items() if k != "tool_calls"} for m in messages]
        return good_render(stripped, tools)

    rep = tc.check_template(FakeTokenizer(render))
    assert not rep.ok
    assert "tool_call_arguments_rendered" in {c.name for c in rep.failures}


def test_template_that_drops_tool_results_fails():
    def render(messages, tools):
        return good_render([m for m in messages if m["role"] != "tool"], tools)

    rep = tc.check_template(FakeTokenizer(render))
    assert not rep.ok
    assert "tool_result_rendered" in {c.name for c in rep.failures}


def test_unknown_marker_is_only_a_warning():
    """We may just not have seen the family — warn, don't block."""
    def render(messages, tools):
        return good_render(messages, tools).replace("<tool_call>", "<<CALL>>")

    rep = tc.check_template(FakeTokenizer(render))
    assert rep.ok
    assert "tool_call_marker" in {c.name for c in rep.warnings}


def test_markers_splitting_into_multiple_ids_fail():
    """If in-text markers stop mapping to single ids, the whole corpus is off-distribution."""
    tok = FakeTokenizer(good_render, specials_multi_id={"<tool_call>"})
    rep = tc.check_template(tok)
    assert not rep.ok
    assert "special_tokens_single_id" in {c.name for c in rep.failures}


def test_strict_template_refusing_assistant_first_window_warns_not_fails():
    """The Qwen3.5-VL trap: strict templates reject a window with no user turn."""
    def render(messages, tools):
        if not any(m["role"] == "user" for m in messages):
            raise ValueError("No user query found")
        return good_render(messages, tools)

    rep = tc.check_template(FakeTokenizer(render))
    assert rep.ok, rep.summary()
    assert "assistant_first_window" in {c.name for c in rep.warnings}


def test_missing_chat_template_fails_immediately():
    tok = FakeTokenizer(good_render)
    tok.chat_template = None
    rep = tc.check_template(tok)
    assert not rep.ok
    assert [c.name for c in rep.checks] == ["has_chat_template"]


def test_assert_template_ok_raises_with_the_report():
    with pytest.raises(RuntimeError, match="tool_call_arguments_rendered"):
        tc.assert_template_ok(FakeTokenizer(
            lambda m, t: good_render(
                [{k: v for k, v in x.items() if k != "tool_calls"} for x in m], t)))

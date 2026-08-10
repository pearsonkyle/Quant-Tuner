"""Reasoning normalization and red-team refusal substitution.

Both of these are silent-failure shaped: a reasoning policy that drops everything, or a
refusal substitution that leaks one original completion, produces a corpus that looks
perfectly healthy from the outside.
"""

from __future__ import annotations

import pytest

from quant_tuner.data import reasoning, refusals

FIELD_MSG = {"role": "assistant", "content": "the answer", "reasoning_content": "step one"}
INLINE_MSG = {"role": "assistant", "content": "<think>\nstep one\n</think>\n\nthe answer"}


# ------------------------------------------------------------------------- reasoning
def test_both_source_shapes_normalize_to_the_same_thing():
    """logs-cli carries inline <think>; logs-agents carries a reasoning_content field."""
    a = reasoning.apply_policy([FIELD_MSG], "auto")
    b = reasoning.apply_policy([INLINE_MSG], "auto")
    assert a == b
    assert a[0]["content"] == "<think>\nstep one\n</think>\n\nthe answer"
    assert "reasoning_content" not in a[0]


def test_field_policy_is_the_inverse():
    for src in (FIELD_MSG, INLINE_MSG):
        out = reasoning.apply_policy([src], "field")[0]
        assert out["reasoning_content"] == "step one"
        assert out["content"] == "the answer"
        assert "<think>" not in out["content"]


def test_drop_policy_removes_reasoning_from_both_shapes():
    for src in (FIELD_MSG, INLINE_MSG):
        out = reasoning.apply_policy([src], "drop")[0]
        assert out["content"] == "the answer"
        assert "reasoning_content" not in out


def test_apply_policy_never_mutates_the_input():
    msgs = [dict(FIELD_MSG)]
    before = dict(msgs[0])
    reasoning.apply_policy(msgs, "auto")
    assert msgs[0] == before


def test_non_assistant_turns_pass_through_untouched():
    msgs = [{"role": "user", "content": "<think>not mine</think> hi"},
            {"role": "tool", "content": "output"}]
    assert reasoning.apply_policy(msgs, "auto") == msgs


def test_assistant_turn_without_reasoning_is_unchanged():
    m = {"role": "assistant", "content": "plain", "tool_calls": [{"id": "1"}]}
    out = reasoning.apply_policy([m], "auto")[0]
    assert out["content"] == "plain" and out["tool_calls"] == [{"id": "1"}]


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown reasoning policy"):
        reasoning.apply_policy([FIELD_MSG], "sometimes")


def test_empty_think_blocks_are_not_counted_as_reasoning():
    """Qwen templates emit an EMPTY <think></think> on every render's final assistant turn.

    Counting those reported healthy coverage on a corpus that contained no reasoning at all.
    """
    corpus = "<think>\n\n</think>\n\nanswer one\n\n<think>\n\n</think>\n\nanswer two"
    assert reasoning.count_reasoning_blocks(corpus) == 0
    assert reasoning.count_reasoning_blocks(corpus, nonempty_only=False) == 2
    real = corpus + "\n\n<think>\nactual thinking\n</think>\n\nanswer three"
    assert reasoning.count_reasoning_blocks(real) == 1


def test_count_available_counts_both_shapes():
    msgs = [FIELD_MSG, INLINE_MSG, {"role": "assistant", "content": "no reasoning"},
            {"role": "user", "content": "q"}]
    assert reasoning.count_available(msgs) == 2


def test_only_a_LEADING_think_block_counts_as_reasoning():
    """Assistant turns in coding logs discuss `<think>` in prose.

    An unanchored match spanned from a backticked `<think>` to a `</think>` mentioned later
    in the same message and deleted the explanation in between — 3 mangled turns on the real
    logs. Reasoning is only ever emitted at the start of a turn, so anchor there.
    """
    prose = {"role": "assistant",
             "content": "The template opens with `<think>` and llama-server returns only "
                        "the post-`</think>` content."}
    out = reasoning.apply_policy([prose], "field")[0]
    assert "reasoning_content" not in out
    assert out["content"] == prose["content"]      # untouched, nothing deleted

    leading = {"role": "assistant", "content": "<think>real cot</think>the answer"}
    out = reasoning.apply_policy([leading], "field")[0]
    assert out["reasoning_content"] == "real cot"
    assert out["content"] == "the answer"

    # a second block mid-message is content, not a second reasoning block
    two = {"role": "assistant", "content": "<think>one</think>mid <think>two</think> end"}
    out = reasoning.apply_policy([two], "field")[0]
    assert out["reasoning_content"] == "one"
    assert out["content"] == "mid <think>two</think> end"


def test_unclosed_leading_think_is_treated_as_truncated_reasoning():
    """Truncated generation: leaving a naked marker would put an unbalanced token in data."""
    m = {"role": "assistant", "content": "<think>cut off mid-thought"}
    out = reasoning.apply_policy([m], "field")[0]
    assert out["reasoning_content"] == "cut off mid-thought"
    assert out.get("content", "") == ""
    assert "<think>" not in reasoning.apply_policy([m], "drop")[0].get("content", "")


def test_orphaned_closing_tag_is_treated_as_reasoning():
    """Capture that began mid-thought: the turn opens with a tail and a tag it never opened."""
    m = {"role": "assistant", "content": "…before formatting.\n</think>\n\nNow the answer"}
    out = reasoning.apply_policy([m], "field")[0]
    assert out["reasoning_content"] == "…before formatting."
    assert out["content"] == "Now the answer"
    assert "</think>" not in reasoning.apply_policy([m], "drop")[0]["content"]


def test_a_closing_tag_discussed_in_prose_after_an_opening_one_is_left_alone():
    """Both tags present mid-prose -> not reasoning, and nothing may be deleted."""
    m = {"role": "assistant",
         "content": "It emits `<think>` then the post-`</think>` content."}
    out = reasoning.apply_policy([m], "field")[0]
    assert out["content"] == m["content"] and "reasoning_content" not in out


# -------------------------------------------------------------------------- refusals
ROW = {
    "case_id": "abc123",
    "messages": [
        {"role": "user", "content": "attack turn one"},
        {"role": "assistant", "content": "HARMFUL COMPLIANT OUTPUT"},
        {"role": "user", "content": "escalation turn two"},
        {"role": "assistant", "content": "MORE HARMFUL OUTPUT"},
    ],
    "outcome": "complied",
    "vulnerability": "Illegal Activity",
    "target_reasoning": "HARMFUL CHAIN OF THOUGHT",
}


def test_every_assistant_turn_is_replaced_by_a_refusal():
    msgs = refusals.refusal_messages(ROW)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in msgs if m["role"] == "user"] == [
        "attack turn one", "escalation turn two"]
    for m in msgs:
        if m["role"] == "assistant":
            assert m["content"] in refusals.GENERIC_REFUSALS


def test_original_completions_and_reasoning_never_appear():
    blob = str(refusals.refusal_messages(ROW))
    for leaked in ("HARMFUL COMPLIANT OUTPUT", "MORE HARMFUL OUTPUT",
                   "HARMFUL CHAIN OF THOUGHT"):
        assert leaked not in blob


def test_multi_turn_attacks_do_not_repeat_one_refusal_verbatim():
    """A 22-turn crescendo repeating one sentence eleven times is a degenerate pattern."""
    row = dict(ROW, messages=[m for _ in range(6) for m in ROW["messages"]])
    picked = [m["content"] for m in refusals.refusal_messages(row)
              if m["role"] == "assistant"]
    assert len(picked) == 12
    assert len(set(picked)) > 1


def test_refusal_choice_is_deterministic():
    assert refusals.refusal_messages(ROW) == refusals.refusal_messages(ROW)
    assert refusals.pick_refusal("abc123:0") == refusals.pick_refusal("abc123:0")


def test_messages_stored_as_a_string_are_parsed():
    """The staged rows carry `messages` as a python-literal string, not JSON."""
    row = dict(ROW, messages=str(ROW["messages"]))
    assert len(refusals.refusal_messages(row)) == 4


def test_conversation_always_starts_user_and_ends_assistant():
    row = dict(ROW, messages=[{"role": "assistant", "content": "leading junk"},
                              {"role": "user", "content": "the attack"}])
    msgs = refusals.refusal_messages(row)
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] in refusals.GENERIC_REFUSALS


def test_row_with_no_user_turn_yields_nothing():
    assert refusals.refusal_messages(dict(ROW, messages=[])) == []
    assert refusals.refusal_messages(
        dict(ROW, messages=[{"role": "assistant", "content": "x"}])) == []


def test_refusal_sessions_shape_is_packable():
    (s,) = list(refusals.refusal_sessions([ROW]))
    assert s["source"] == "redteam:illegal_activity"
    assert s["group"] == "abc123" and s["score"] == 1.0
    assert s["metrics"]["tool_calls"] == 0
    assert s["meta"]["original_outcome"] == "complied"
    assert "target_reasoning" not in s["meta"]

"""Decision-point extraction parity with eval.toolcall's turn walk."""

from __future__ import annotations

from quant_tuner.eval.toolcall import (
    _annotate_failing_calls,
    maybe_inject_system,
    prior_tool_result,
    strip_for_api,
)
from quant_tuner.lens.replay import iter_decision_points


def _session():
    return {
        "session_id": "s1",
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "messages": [
            {"role": "user", "content": "read config.py"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "config.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "PORT=8080"},
            {"role": "assistant", "content": "The port is 8080."},
        ],
    }


def _reference_decision_prefixes(session, system_prompt, max_turns):
    """Reproduce the exact turn walk eval_per_turn uses, collecting prefixes."""
    msgs = maybe_inject_system(session["messages"], system_prompt)
    _annotate_failing_calls(msgs)
    out = []
    turn_idx = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        has_tool_calls = bool(m.get("tool_calls"))
        prior = prior_tool_result(msgs, i)
        if not has_tool_calls and prior is None:
            continue
        if turn_idx >= max_turns:
            break
        turn_idx += 1
        out.append((turn_idx, "tool_call" if has_tool_calls else "post_result",
                    strip_for_api(msgs[:i])))
    return out


def test_decision_points_match_eval_walk():
    from quant_tuner.eval.toolcall import DEFAULT_SYSTEM_PROMPT

    session = _session()
    ref = _reference_decision_prefixes(session, DEFAULT_SYSTEM_PROMPT, 8)
    got = list(iter_decision_points(session, max_turns=8))
    assert len(got) == len(ref)
    for dp, (turn, kind, prefix) in zip(got, ref, strict=True):
        assert dp.turn == turn
        assert dp.kind == kind
        assert dp.prefix == prefix


def test_gold_tool_extracted_for_tool_call_turn():
    dps = list(iter_decision_points(_session(), max_turns=8))
    tool_call = next(d for d in dps if d.kind == "tool_call")
    assert tool_call.gold_tool == "read_file"


def test_max_turns_bounds_walk():
    session = _session()
    # add many more tool-call turns
    for k in range(10):
        session["messages"].append(
            {"role": "assistant", "tool_calls": [
                {"id": f"x{k}", "function": {"name": "read_file", "arguments": "{}"}}]})
        session["messages"].append({"role": "tool", "tool_call_id": f"x{k}", "content": "ok"})
    dps = list(iter_decision_points(session, max_turns=3))
    assert max(d.turn for d in dps) <= 3

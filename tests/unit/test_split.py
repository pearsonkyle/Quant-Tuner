"""Unit tests for data.split helpers (no tokenizer/model load required)."""

from quant_tuner.data.split import session_tools, split_sessions

SCHEMA = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]


def test_session_tools_prefers_top_level():
    s = {"tools": SCHEMA}
    msgs = [{"role": "system", "content": "x", "tools": [{"other": 1}]}]
    assert session_tools(s, msgs) is SCHEMA


def test_session_tools_falls_back_to_system_message():
    # logtrain export attaches schemas to the system message, not the top level.
    s = {"messages": []}
    msgs = [{"role": "system", "content": "x", "tools": SCHEMA},
            {"role": "user", "content": "hi"}]
    assert session_tools(s, msgs) is SCHEMA


def test_session_tools_none_when_absent():
    assert session_tools({}, [{"role": "user", "content": "hi"}]) is None


def test_session_tools_ignores_string_messages():
    # normalize_messages may leave odd shapes; helper must not crash on str.
    assert session_tools({}, ["raw string message"]) is None


def test_split_sessions_order_is_deterministic():
    # Same seed must yield identical ordering, independent of PYTHONHASHSEED.
    sessions = [{"messages": [{"role": "user", "content": f"msg-{i}"}]} for i in range(50)]
    a = split_sessions(sessions, train_frac=0.8, test_frac=0.2, holdout_frac=0.0, seed=42)
    b = split_sessions(sessions, train_frac=0.8, test_frac=0.2, holdout_frac=0.0, seed=42)
    a_ids = [m["messages"][0]["content"] for m in a["train"]]
    b_ids = [m["messages"][0]["content"] for m in b["train"]]
    assert a_ids == b_ids
    assert len(a["train"]) == 40 and len(a["test"]) == 10

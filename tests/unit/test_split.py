"""Unit tests for data.split helpers (no tokenizer/model load required)."""

from quant_tuner.data.split import (
    session_tools,
    session_windows,
    split_sessions,
    stratified_pack,
    stub_system_content,
)

SCHEMA = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]


class FakeTok:
    """Whitespace tokenizer: 1 token == 1 word. Renders schemas from tools=,
    so tests can verify schema survival independently of the system prose."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        parts = []
        if tools:
            parts.append("TOOLS " + " ".join(t["function"]["name"] for t in tools))
        for m in messages:
            c = m.get("content")
            parts.append(f"{m['role']} {c if isinstance(c, str) else ''}")
        return " ".join(parts)


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


def test_stub_system_content_truncates_to_budget():
    tok = FakeTok()
    assert stub_system_content("a b c d e f", tok, 3) == "a b c"
    # Under-budget content is returned untouched.
    assert stub_system_content("a b", tok, 5) == "a b"
    # Non-string content is passed through (can't safely truncate).
    assert stub_system_content(["part"], tok, 1) == ["part"]


def _long_session():
    msgs = [{"role": "system", "content": "sys " * 20}]
    for i in range(12):
        role = "assistant" if i % 2 else "user"
        msgs.append({"role": role, "content": f"turn{i}"})
    return msgs


def test_session_windows_emits_multiple_capped_windows_with_stub():
    tok = FakeTok()
    msgs = _long_session()
    wins = session_windows(
        tok, msgs, SCHEMA, cap_tokens=12,
        system_content="STUB", max_windows=8,
    )
    assert len(wins) > 1, "a long session must produce several windows"
    for text, ntok in wins:
        # Every window carries the stubbed system + the tool schema (from tools=),
        # never the full prose.
        assert text.startswith("TOOLS Bash")
        assert "system STUB" in text
        assert "sys sys" not in text  # full prose was replaced by the stub
        assert ntok <= 12
    # Windows together cover all 12 body turns in order.
    seen = " ".join(t for t, _ in wins)
    for i in range(12):
        assert f"turn{i}" in seen


def test_session_windows_body_never_starts_on_tool():
    tok = FakeTok()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "tool", "content": "orphan"},  # leading orphan must be skipped
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    wins = session_windows(tok, msgs, None, cap_tokens=100, system_content="S")
    assert len(wins) == 1
    # The orphan tool result before the first user turn is dropped.
    assert "tool orphan" not in wins[0][0]
    assert "user hello" in wins[0][0]


class StrictTok(FakeTok):
    """Like FakeTok but mimics strict chat templates (e.g. Qwen3.5-VL) that
    refuse to render a window with no user turn."""

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        return super().apply_chat_template(messages, tools=tools, tokenize=False)


def test_session_windows_skips_userless_windows_for_strict_templates():
    # A window that would start on an orphaned assistant turn (no user in it)
    # must be skipped, not raise — otherwise the whole session is discarded and
    # strict-template models lose nearly all calibration signal.
    tok = StrictTok()
    msgs = [{"role": "system", "content": "S"}]
    for i in range(12):
        msgs.append({"role": "assistant" if i % 2 else "user", "content": f"turn{i}"})
    wins = session_windows(tok, msgs, SCHEMA, cap_tokens=6,
                           system_content="STUB", max_windows=8)
    assert wins, "strict template must still yield the user-anchored windows"
    for text, _ in wins:
        assert "user" in text  # every emitted window carries a user turn


def _pack_sessions(n=6):
    sessions = []
    for i in range(n):
        msgs = [{"role": "system", "content": "shared prose " * 10}]
        for j in range(8):
            role = "assistant" if j % 2 else "user"
            msgs.append({"role": role, "content": f"s{i}t{j}"})
        sessions.append({"source": "logs", "messages": msgs, "tools": SCHEMA})
    return sessions


def test_stratified_pack_windowed_audit_and_dedup():
    tok = FakeTok()
    sessions = _pack_sessions()
    chunks, kept, total, audit = stratified_pack(
        sessions, tok, target_tokens=400, per_session_cap=20, seed=42,
        system_prose_budget=2, full_prose_quota=1, max_windows_per_session=8,
    )
    w = audit["windowing"]
    # One identical system prompt across all sessions -> exactly one full-prose copy.
    assert w["unique_system_prompts"] == 1
    assert w["full_prose_sessions"] == 1
    assert w["stub_sessions"] >= 1
    assert w["windows_emitted"] == audit["chunk_count"] == len(chunks)
    assert 0.0 < w["tool_turn_token_share"] <= 1.0
    assert w["system_prose_tokens"] + w["body_tokens"] == total


def test_stratified_pack_legacy_path_unchanged():
    tok = FakeTok()
    sessions = _pack_sessions()
    _, _, _, audit = stratified_pack(
        sessions, tok, target_tokens=400, per_session_cap=20, seed=42,
    )
    # No windowing block when system_prose_budget is None (default).
    assert "windowing" not in audit


def test_split_sessions_order_is_deterministic():
    # Same seed must yield identical ordering, independent of PYTHONHASHSEED.
    sessions = [{"messages": [{"role": "user", "content": f"msg-{i}"}]} for i in range(50)]
    a = split_sessions(sessions, train_frac=0.8, test_frac=0.2, holdout_frac=0.0, seed=42)
    b = split_sessions(sessions, train_frac=0.8, test_frac=0.2, holdout_frac=0.0, seed=42)
    a_ids = [m["messages"][0]["content"] for m in a["train"]]
    b_ids = [m["messages"][0]["content"] for m in b["train"]]
    assert a_ids == b_ids
    assert len(a["train"]) == 40 and len(a["test"]) == 10

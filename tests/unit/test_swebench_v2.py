"""Tests for SWE-rebench-V2 (multi-language) support.

The properties pinned here are the ones whose failure mode is *silent*: a V2 run that
looks like "the model solved nothing" when really the grader used the wrong workdir,
the wrong parser, or a stale id comparison. All are pure-function tests — no Docker,
no network, no model.
"""

import json
import subprocess

import pytest

from quant_tuner.eval.swebench import _build_env_config
from quant_tuner.eval.swebench_grade import (
    diagnose_container_error,
    evaluate_results_v2,
    install_config_of,
    is_v2_instance,
    normalize_test_name,
    parser_for_instance,
    v2_test_script,
    v2_workdir,
)


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

"""Test-based grading for SWE-rebench agentic runs.

Given an instance (gold ``test_patch`` + ``FAIL_TO_PASS`` / ``PASS_TO_PASS``
node ids) and a candidate ``model_patch`` produced by the agent, decide whether
the patch *resolves* the issue by actually running the tests inside the
instance's Docker image:

    fresh container @ image
      → git reset --hard <base_commit> ; git clean -fd
      → apply model_patch        (the agent's diff)
      → apply test_patch         (the gold tests)
      → run FAIL_TO_PASS + PASS_TO_PASS with pytest
      → resolved = every FAIL_TO_PASS *and* every PASS_TO_PASS now PASSES

This is a pragmatic grader that covers the common pytest case (the bulk of the
Python repos in SWE-rebench). It deliberately does **not** reimplement the full
per-repo test-command matrix from the official ``swebench`` harness — repos
with a non-pytest runner (e.g. Django's own runner) will surface as a grading
error rather than a silent pass. The container interaction is injectable
(``env=``) so the pure parsing/decision helpers below are unit-testable without
Docker.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
from typing import Any, Protocol

# pytest's ``-rA`` short test summary prints one line per test, e.g.
#   PASSED tests/test_x.py::test_y
#   FAILED tests/test_x.py::test_z
_STATUS_LINE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE
)
_PASSING = {"PASSED", "XFAIL"}  # XFAIL = expected-fail that behaved as expected
_LOG_CLIP = 20_000


class _ExecEnv(Protocol):
    """Minimal slice of mini-swe-agent's environment we depend on.

    Only ``execute`` is required: an injected env is owned (and cleaned up) by
    the caller; the env we create ourselves is a concrete ``DockerEnvironment``
    whose ``cleanup`` we call directly.
    """

    def execute(self, action: dict, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def as_test_list(value: Any) -> list[str]:
    """Normalize a FAIL_TO_PASS / PASS_TO_PASS field to a list of node ids.

    SWE-bench-format datasets store these as a JSON-encoded list string; some
    mirrors store a native list. Accept both (and a plain whitespace/newline
    separated string as a last resort).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return s.split()
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    return [str(value)]


def parse_pytest_statuses(output: str) -> dict[str, str]:
    """Map each reported pytest node id → its status from a ``-rA`` summary."""
    statuses: dict[str, str] = {}
    for match in _STATUS_LINE.finditer(output or ""):
        status, node = match.group(1), match.group(2)
        # If a node appears more than once, a FAILED/ERROR wins over PASSED.
        prev = statuses.get(node)
        if prev is None or (prev in _PASSING and status not in _PASSING):
            statuses[node] = status
    return statuses


def evaluate_results(
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    statuses: dict[str, str],
) -> dict[str, Any]:
    """Decide ``resolved`` and report per-id pass/fail.

    ``resolved`` requires every FAIL_TO_PASS id and every PASS_TO_PASS id to be
    present in ``statuses`` with a passing status. A missing id (never ran /
    collection error) counts as a failure.
    """

    def _passed(node: str) -> bool:
        return statuses.get(node) in _PASSING

    f2p = {node: _passed(node) for node in fail_to_pass}
    p2p = {node: _passed(node) for node in pass_to_pass}
    n_f2p_passed = sum(f2p.values())
    n_p2p_passed = sum(p2p.values())
    resolved = bool(fail_to_pass) and all(f2p.values()) and all(p2p.values())
    return {
        "resolved": resolved,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "n_fail_to_pass": len(fail_to_pass),
        "n_fail_to_pass_passed": n_f2p_passed,
        "n_pass_to_pass": len(pass_to_pass),
        "n_pass_to_pass_passed": n_p2p_passed,
    }


def _apply_patch_command(patch_text: str, remote_path: str) -> str:
    """Shell command that materializes ``patch_text`` in the container and applies it.

    base64 round-trips the patch so arbitrary diff content (quotes, heredoc
    markers, binary-ish hunks) survives the ``docker exec`` shell boundary.
    Falls back through ``git apply --3way`` then ``patch -p1``.
    """
    b64 = base64.b64encode(patch_text.encode("utf-8")).decode("ascii")
    quoted = shlex.quote(b64)
    return (
        f"printf %s {quoted} | base64 -d > {remote_path} && "
        f"(git apply -v {remote_path} "
        f"|| git apply -v --3way {remote_path} "
        f"|| patch --batch --fuzz=5 -p1 -i {remote_path})"
    )


def _pytest_command(node_ids: list[str]) -> str:
    quoted = " ".join(shlex.quote(n) for n in node_ids)
    return f"python -m pytest -rA -p no:cacheprovider --no-header -q {quoted}"


def test_command(instance: dict, node_ids: list[str]) -> str:
    """Build the command that runs ``node_ids``.

    Prefer the instance's own ``install_config.test_cmd`` (the repo's exact
    runner + flags, e.g. ``pytest --no-header -rA -p no:cacheprovider …``) so
    the ``-rA`` summary we parse matches what the harness used. Fall back to a
    generic pytest invocation when no test_cmd is recorded.

    Always wraps the command in ``conda run --no-capture-output -n testbed``
    so the testbed env is active.  The SWE-rebench images activate conda via
    ``~/.bashrc``, but the bashrc guard (``case $-``) skips activation for
    non-interactive shells, making ``pytest`` not found under plain ``bash -c``.
    ``conda run`` sidesteps this entirely.
    """
    test_cmd = ((instance.get("install_config") or {}).get("test_cmd") or "").strip()
    quoted = " ".join(shlex.quote(n) for n in node_ids)
    inner = f"{test_cmd} {quoted}" if test_cmd else _pytest_command(node_ids)
    return f"conda run --no-capture-output -n testbed bash -c {shlex.quote(inner)}"


# ---------------------------------------------------------------------------
# Orchestration (Docker)
# ---------------------------------------------------------------------------


def _empty_result(error: str) -> dict[str, Any]:
    return {
        "resolved": False,
        "fail_to_pass": {},
        "pass_to_pass": {},
        "n_fail_to_pass": 0,
        "n_fail_to_pass_passed": 0,
        "n_pass_to_pass": 0,
        "n_pass_to_pass_passed": 0,
        "error": error,
        "log": "",
    }


def grade_instance(
    instance: dict,
    model_patch: str,
    *,
    image: str | None = None,
    env: _ExecEnv | None = None,
    cwd: str = "/testbed",
    test_timeout: int = 1800,
    container_timeout: str = "2h",
    executable: str = "docker",
) -> dict[str, Any]:
    """Grade ``model_patch`` against ``instance`` by running its tests in Docker.

    Provide either a started ``env`` (the caller owns its lifecycle) or an
    ``image`` to spin up a fresh container (cleaned up here). Returns a dict
    with ``resolved`` plus per-id detail, an ``error`` string (``None`` on a
    clean grade), and a clipped ``log``.
    """
    fail_to_pass = as_test_list(instance.get("FAIL_TO_PASS"))
    pass_to_pass = as_test_list(instance.get("PASS_TO_PASS"))
    base_commit = instance.get("base_commit", "")
    test_patch = instance.get("test_patch") or ""

    if not model_patch or not model_patch.strip():
        return _empty_result("empty model patch (agent produced no diff)")
    if not fail_to_pass:
        return _empty_result("instance has no FAIL_TO_PASS tests to grade")

    created = None  # the env we own (and must clean up), if any
    if env is None:
        if not image:
            raise ValueError("grade_instance requires either env= or image=")
        # Imported lazily so this module imports without mini-swe-agent installed.
        from minisweagent.environments.docker import DockerEnvironment

        created = DockerEnvironment(
            image=image,
            cwd=cwd,
            timeout=test_timeout,
            container_timeout=container_timeout,
            executable=executable,
            env={
                "PAGER": "cat",
                "TQDM_DISABLE": "1",
                "PIP_PROGRESS_BAR": "off",
                # Non-login `bash -c` won't source ~/.bashrc; point BASH_ENV at it
                # so the image's `conda activate testbed` runs (matches swebench.yaml).
                "BASH_ENV": "/root/.bashrc",
            },
            interpreter=["bash", "-c"],
        )
        env = created

    log_parts: list[str] = []

    def _run(label: str, command: str, *, timeout: int | None = None) -> dict[str, Any]:
        out = env.execute({"command": command}, timeout=timeout) if timeout else env.execute(
            {"command": command}
        )
        log_parts.append(
            f"$ [{label}] (rc={out.get('returncode')})\n{out.get('output', '')}"
        )
        return out

    try:
        reset = f"cd {shlex.quote(cwd)} && git reset --hard {shlex.quote(base_commit)} && git clean -fd"
        r = _run("reset", reset)
        if r.get("returncode") != 0:
            return _empty_result("git reset/clean failed") | {"log": _clip("\n".join(log_parts))}

        r = _run("apply-model", _apply_patch_command(model_patch, "/tmp/model.patch"))
        if r.get("returncode") != 0:
            return _empty_result("model patch did not apply") | {
                "log": _clip("\n".join(log_parts))
            }

        if test_patch.strip():
            r = _run("apply-test", _apply_patch_command(test_patch, "/tmp/test.patch"))
            if r.get("returncode") != 0:
                return _empty_result("gold test patch did not apply") | {
                    "log": _clip("\n".join(log_parts))
                }

        r = _run(
            "pytest",
            test_command(instance, fail_to_pass + pass_to_pass),
            timeout=test_timeout,
        )
        statuses = parse_pytest_statuses(r.get("output", ""))
        result = evaluate_results(fail_to_pass, pass_to_pass, statuses)
        # A run that collected nothing usually means a non-pytest runner.
        if not statuses:
            result["error"] = "no pytest results parsed (non-pytest runner or collection error)"
        else:
            result["error"] = None
        result["log"] = _clip("\n".join(log_parts))
        return result
    finally:
        if created is not None:
            created.cleanup()


def _clip(text: str) -> str:
    if len(text) <= _LOG_CLIP:
        return text
    half = _LOG_CLIP // 2
    return text[:half] + f"\n...[{len(text) - _LOG_CLIP} chars elided]...\n" + text[-half:]

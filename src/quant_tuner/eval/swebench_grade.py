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
import subprocess
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


# ---------------------------------------------------------------------------
# SWE-rebench-V2 (multi-language) support
# ---------------------------------------------------------------------------
#
# V1 instances are Python/pytest and are graded by the path above. V2 instances carry
# ``install_config.log_parser`` naming one of the dataset's own parsers, plus a
# per-instance ``test_cmd`` and a repo-named workdir. We reproduce V2's harness
# (SWE-rebench-V2 ``scripts/eval.py``) rather than approximating it, because the
# recorded FAIL_TO_PASS/PASS_TO_PASS ids are literally whatever that parser emitted.

# Some runners embed timings in the test name ("... [1.34 ms]"), so the recorded ids
# and a fresh run's ids differ by the timing alone. Upstream normalizes BOTH sides;
# so do we. Ported verbatim from SWE-rebench-V2 scripts/eval.py.
_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def normalize_test_name(name: str) -> str:
    """Strip known timing suffixes/infixes so ids compare across runs."""
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def install_config_of(instance: dict) -> dict:
    """``install_config`` as a dict (some mirrors store it JSON-encoded)."""
    cfg = instance.get("install_config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except (json.JSONDecodeError, ValueError):
            return {}
    return cfg if isinstance(cfg, dict) else {}


def is_v2_instance(instance: dict) -> bool:
    """True for SWE-rebench-V2 rows — they name their own log parser."""
    return bool(install_config_of(instance).get("log_parser"))


def v2_workdir(instance: dict) -> str:
    """V2 images check the repo out at ``/<repo-name>``, not ``/testbed``."""
    repo = instance.get("repo") or ""
    name = repo.split("/")[-1].strip()
    if not name:
        raise ValueError(f"instance {instance.get('instance_id')!r} has no usable repo field")
    return f"/{name}"


def workdir_for(instance: dict) -> str:
    """Where THIS instance's repo is checked out inside its image.

    The single source of truth for the container path: V1 images use ``/testbed``,
    V2 images use ``/<repo-name>``. The agent backend, the grader and the mini-swe
    environment must all agree — an agent that ``cd``s to a directory that does not
    exist gets an OCI ``chdir`` error as the output of *every* command, and its
    "patch" ends up being that error string.
    """
    return v2_workdir(instance) if is_v2_instance(instance) else "/testbed"


def parser_for_instance(instance: dict):
    """Resolve the instance's named log parser from the vendored V2 registry."""
    name = install_config_of(instance).get("log_parser")
    if not name:
        raise ValueError("instance has no install_config.log_parser")
    from quant_tuner.eval import _swerebench_v2_parsers as parsers

    fn = parsers.NAME_TO_PARSER.get(name) or getattr(parsers, name, None)
    if fn is None:
        raise ValueError(f"unknown log parser {name!r} (vendored registry is stale?)")
    return fn


def v2_test_script(instance: dict, cwd: str) -> str:
    """The V2 test script: cd into the repo, then run the instance's own test_cmd(s).

    Deliberately does **not** wrap in ``conda run -n testbed`` (that is a Python-image
    convention; a Go/Node/JVM image has no such env) and does not append node ids —
    V2's ``test_cmd`` already selects the suite, and the parser maps the whole log.
    """
    cmds = install_config_of(instance).get("test_cmd") or []
    if isinstance(cmds, str):
        cmds = [cmds]
    cmds = [str(c) for c in cmds if str(c).strip()]
    if not cmds:
        raise ValueError("instance has no install_config.test_cmd")
    return "\n".join([f"cd {shlex.quote(cwd)}", "set -e", *cmds])


def evaluate_results_v2(
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    statuses: dict[str, str],
) -> dict[str, Any]:
    """Set-based grading over timing-normalized ids (mirrors V2's build_report_item)."""
    passed = {normalize_test_name(k) for k, v in statuses.items() if v == "PASSED"}
    f2p = {n: normalize_test_name(n) in passed for n in fail_to_pass}
    p2p = {n: normalize_test_name(n) in passed for n in pass_to_pass}
    resolved = bool(fail_to_pass) and all(f2p.values()) and all(p2p.values())
    return {
        "resolved": resolved,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "n_fail_to_pass": len(fail_to_pass),
        "n_fail_to_pass_passed": sum(f2p.values()),
        "n_pass_to_pass": len(pass_to_pass),
        "n_pass_to_pass_passed": sum(p2p.values()),
    }


def diagnose_container_error(exc: BaseException) -> str:
    """Turn an opaque ``docker run`` failure into an actionable message.

    ``docker run`` exits **125** for every "the daemon refused to start it" reason, so a
    registry outage, a full VM disk and a missing image all surface identically as
    ``CalledProcessError ... exit status 125``. Instances then fail in seconds and the
    run looks like a model problem. Probe the daemon and say which it is.
    """
    text = f"{type(exc).__name__}: {exc}"
    if "125" not in text and "exit status 125" not in text:
        return text

    hints: list[str] = []
    try:
        probe = subprocess.run(
            ["docker", "pull", "hello-world:latest"],
            capture_output=True, text=True, timeout=45,
        )
        blob = (probe.stderr or "") + (probe.stdout or "")
        if probe.returncode != 0:
            if any(s in blob for s in ("context deadline exceeded", "no such host",
                                       "dial tcp", "TLS handshake", "i/o timeout")):
                hints.append("Docker cannot reach the registry (network/DNS/VPN down) — "
                             "image pulls will fail until connectivity returns")
            elif "no space left" in blob:
                hints.append("Docker VM disk is full — free space or raise the disk limit")
            else:
                hints.append(f"`docker pull hello-world` also failed: {blob.strip()[:200]}")
    except subprocess.TimeoutExpired:
        hints.append("`docker pull hello-world` timed out — the daemon or its network is stuck")
    except FileNotFoundError:
        hints.append("`docker` executable not found on PATH")
    except Exception:  # diagnosis must never mask the original failure
        pass

    try:
        df = subprocess.run(["docker", "system", "df"], capture_output=True,
                            text=True, timeout=20)
        for line in (df.stdout or "").splitlines():
            if line.startswith("Images"):
                hints.append(f"docker system df -> {line.strip()}")
    except Exception:
        pass

    return text + (("  [diagnosis] " + "; ".join(hints)) if hints else "")


# SWE-rebench-V2's own harness applies both patches with these flags. They matter: the
# agent's diff and the gold test patch frequently touch neighbouring lines, and a strict
# `git apply` rejects the second one — which we would otherwise report as "gold test
# patch did not apply", i.e. an unresolvable instance, for a purely cosmetic conflict.
_V2_APPLY_FLAGS = "--3way --recount --ignore-space-change --whitespace=nowarn"


def _apply_patch_command(patch_text: str, remote_path: str, *, v2: bool = False) -> str:
    """Shell command that materializes ``patch_text`` in the container and applies it.

    base64 round-trips the patch so arbitrary diff content (quotes, heredoc
    markers, binary-ish hunks) survives the ``docker exec`` shell boundary.
    Falls back through ``git apply --3way`` then ``patch -p1``.

    ``v2=True`` tries SWE-rebench-V2's own lenient invocation first, so our verdicts
    match the dataset's harness. The V1 chain is deliberately left untouched: it
    produced the published Python trajectory runs, and quietly making it more
    permissive would silently change those numbers.
    """
    b64 = base64.b64encode(patch_text.encode("utf-8")).decode("ascii")
    quoted = shlex.quote(b64)
    upstream = f"git apply -v {_V2_APPLY_FLAGS} {remote_path} || " if v2 else ""
    return (
        f"printf %s {quoted} | base64 -d > {remote_path} && "
        f"({upstream}"
        f"git apply -v {remote_path} "
        f"|| git apply -v --3way {remote_path} "
        f"|| patch --batch --fuzz=5 -p1 -i {remote_path})"
    )


_DIFF_PATH_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def test_patch_paths(test_patch: str) -> list[str]:
    """Repo-relative paths the gold test patch touches."""
    paths: list[str] = []
    for a, b in _DIFF_PATH_RE.findall(test_patch or ""):
        for p in (a, b):
            if p not in paths:
                paths.append(p)
    return paths


def revert_test_files_command(base_commit: str, test_patch: str) -> str | None:
    """Restore the gold test files to their pristine state before applying test_patch.

    Agents routinely edit test files — e.g. a repo-wide symbol rename that also rewrites
    ``*_test.go``. The gold test patch was cut against the pristine tests, so it then
    fails with ``does not match index`` and the instance is scored "gold test patch did
    not apply" — an unresolvable instance for a reason that has nothing to do with the
    fix being right or wrong.

    Reverting these paths is what the official SWE-bench harness does, and it is what
    the agent system prompt already promises ("it reverts any edits you make to test
    files before grading"). Only files the gold patch touches are reverted, so a
    source-code fix is never undone.

    Returns ``None`` when the patch adds only new files (nothing to restore).
    """
    paths = test_patch_paths(test_patch)
    if not paths:
        return None
    quoted = " ".join(shlex.quote(p) for p in paths)
    ref = shlex.quote(base_commit) if base_commit else "HEAD"
    # `git checkout <ref> -- <paths>` fails wholesale if ANY path is absent at that ref
    # (a patch that creates a new test file), so clear those separately and tolerate both.
    return (
        f"git checkout {ref} -- {quoted} 2>/dev/null; "
        f"for f in {quoted}; do git checkout {ref} -- \"$f\" 2>/dev/null || rm -f \"$f\"; done; "
        f"true"
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
    cwd: str | None = None,
    test_timeout: int = 1800,
    container_timeout: str = "2h",
    executable: str = "docker",
) -> dict[str, Any]:
    """Grade ``model_patch`` against ``instance`` by running its tests in Docker.

    Provide either a started ``env`` (the caller owns its lifecycle) or an
    ``image`` to spin up a fresh container (cleaned up here). Returns a dict
    with ``resolved`` plus per-id detail, an ``error`` string (``None`` on a
    clean grade), and a clipped ``log``.

    Handles both dataset generations. SWE-rebench **V1** (Python/pytest) runs the
    pytest path with ``cwd=/testbed``; **V2** (20 languages) runs the instance's own
    ``install_config.test_cmd`` at ``/<repo-name>`` and parses with the parser the
    instance names. ``cwd`` defaults per generation; pass it to override.
    """
    fail_to_pass = as_test_list(instance.get("FAIL_TO_PASS"))
    pass_to_pass = as_test_list(instance.get("PASS_TO_PASS"))
    base_commit = instance.get("base_commit", "")
    test_patch = instance.get("test_patch") or ""
    v2 = is_v2_instance(instance)
    if cwd is None:
        cwd = v2_workdir(instance) if v2 else "/testbed"

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
        # Reset to base_commit when the sha is present and known to the checkout;
        # fall back to HEAD (V2 images are already checked out at base, and some
        # ship a shallow clone where the sha is not a resolvable ref).
        target = f"git reset --hard {shlex.quote(base_commit)}" if base_commit else ""
        reset_cmd = f"({target} || git reset --hard HEAD)" if target else "git reset --hard HEAD"
        reset = f"cd {shlex.quote(cwd)} && {reset_cmd} && git clean -fd"
        r = _run("reset", reset)
        if r.get("returncode") != 0:
            return _empty_result("git reset/clean failed") | {"log": _clip("\n".join(log_parts))}

        r = _run("apply-model", _apply_patch_command(model_patch, "/tmp/model.patch", v2=v2))
        if r.get("returncode") != 0:
            return _empty_result("model patch did not apply") | {
                "log": _clip("\n".join(log_parts))
            }

        if test_patch.strip():
            # Undo any agent edits to the gold test files first — otherwise a repo-wide
            # rename that also rewrote *_test.* makes the gold patch fail to apply, and a
            # possibly-correct fix is scored as an infrastructure failure.
            if v2 and (revert := revert_test_files_command(base_commit, test_patch)):
                _run("revert-test-files", revert)
            r = _run("apply-test", _apply_patch_command(test_patch, "/tmp/test.patch", v2=v2))
            if r.get("returncode") != 0:
                return _empty_result("gold test patch did not apply") | {
                    "log": _clip("\n".join(log_parts))
                }

        if v2:
            parser = parser_for_instance(instance)
            r = _run("tests", v2_test_script(instance, cwd), timeout=test_timeout)
            statuses = parser(r.get("output", ""))
            result = evaluate_results_v2(fail_to_pass, pass_to_pass, statuses)
            empty_msg = (
                f"no results parsed by {install_config_of(instance)['log_parser']} "
                f"(build/setup failure, or the suite never ran)"
            )
        else:
            r = _run(
                "pytest",
                test_command(instance, fail_to_pass + pass_to_pass),
                timeout=test_timeout,
            )
            statuses = parse_pytest_statuses(r.get("output", ""))
            result = evaluate_results(fail_to_pass, pass_to_pass, statuses)
            # A run that collected nothing usually means a non-pytest runner.
            empty_msg = "no pytest results parsed (non-pytest runner or collection error)"
        # An unparseable log is an INFRA failure, not a failed patch. Surfacing it as
        # an error (rather than a silent resolved=False) is what keeps a broken image
        # or a stale parser from masquerading as "the model didn't solve it".
        result["error"] = None if statuses else empty_msg
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

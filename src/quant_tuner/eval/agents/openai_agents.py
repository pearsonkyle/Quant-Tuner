"""OpenAI Agents SDK backend.

Drives the `OpenAI Agents SDK <https://openai.github.io/openai-agents-python/>`_
against the same local ``llama-server`` (OpenAI-compatible Chat Completions) and
the same per-instance Docker container as the mini-swe-agent backend, but with
the Agents SDK's own planning loop. The agent is given a single ``bash`` tool
that shells into the instance's container (``env.execute``); after the loop ends
(submission, max-turns, or wall-timeout) the patch is read back as a
``git diff`` of the repo checkout — no submit sentinel, which is also the contract
the future CLI-in-container backends (Qwen Code, Claude Code) will use.

The checkout path is **per instance**, not a constant: SWE-rebench V1 images use
``/testbed`` while V2 (multi-language) images use ``/<repo-name>``. It comes from
``swebench_grade.workdir_for`` so the agent, the grader and the environment cannot
disagree — if they do, every command returns an OCI ``chdir`` error and that error
string silently becomes the submitted 'patch'.

Requires the ``swebench`` extra (``openai-agents``). All SDK imports are lazy so
``get_backend('openai-agents')`` resolves without the extra installed; the
SDK-free helpers below carry the testable logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any

from quant_tuner.eval.agents.base import AgentRunContext, AgentRunResult
from quant_tuner.eval.swebench_grade import workdir_for

# Cap a single tool observation so a runaway command can't blow the context.
_MAX_TOOL_OUTPUT_CHARS = 16000

_SYSTEM_PROMPT = """\
You are an autonomous software engineer fixing a real bug in a {language}\
repository already checked out at {repo_dir} (the current working directory).

You have one tool: bash(command). Use it to explore the code, reproduce the \
issue, edit files, and run the project's tests. Make all edits directly on the \
files under {repo_dir} with shell commands (e.g. python, sed, or writing files \
via heredocs). Work in small steps and inspect command output before \
continuing.

Your job is to FIX THE BUG IN THE LIBRARY/SOURCE CODE — the modules under \
{repo_dir} that implement the behavior, not the test suite. Find the function or \
class responsible and change its implementation. Writing a new test, or only \
editing files under tests/, is never a valid fix on its own: the grader runs \
the project's own hidden tests against your SOURCE changes, and it reverts any \
edits you make to test files before grading — so changing tests/ wastes your \
budget. First reproduce the failure, then edit the source, then re-run to \
confirm your source change actually makes the failing behavior pass.

Do NOT run `git commit`, `git checkout`, or `git reset` — your final patch is \
collected automatically from the working tree as a git diff. Understanding the \
bug is not the goal — CHANGING THE CODE is: do not stop until you have actually \
edited a non-test source file, because a run that ends with an empty `git diff` \
is scored as a failure. When the source fix is complete and you have verified \
it, stop and give a short summary of what you changed.
"""

# Command that snapshots the working tree as a patch to grade. Stages new files
# too (`add -A`) so brand-new modules show up in the diff.
_SUBMISSION_CMD = "git -C {repo_dir} add -A && git -C {repo_dir} diff --cached HEAD"

# If the agent stops cleanly but the working tree is still unchanged — it
# explored the code, "understood" the bug, and quit without editing (the
# dominant empty-patch failure mode observed on weaker/low-bit models) — resume
# the same conversation with this forcing message so it actually makes the edit.
# The gate fires ONLY on a clean ``completed`` exit, never on timeout/max_turns.
_EMPTY_PATCH_NUDGE = (
    "STOP — the `git diff` of {repo_dir} is EMPTY: you explored the code but never "
    "edited a source file, so there is nothing to grade. You indicated you understand "
    "the bug; now ACT on it. Use bash (e.g. `sed -i`, or rewrite the file via a "
    "heredoc) to modify the responsible NON-TEST source file under {repo_dir}, then "
    "re-run the project's tests to confirm the failing behavior now passes. Do not "
    "stop again until `git -C {repo_dir} diff` shows a non-empty change to a source file."
)
_MAX_EMPTY_PATCH_RETRIES = 2

# A retriable server error mid-loop (e.g. a 400 BadRequest from a malformed turn,
# often after the model loops on one command and corrupts the message state) used
# to abort the whole instance. Instead, feed the error back and RESTART the
# conversation fresh: the container's file edits persist, so any real progress is
# kept and graded, and the note steers the model off the repetition that caused it.
_MAX_ERROR_RETRIES = 3
_ERROR_RETRY_NOTE = (
    "\n\n[Your previous attempt hit a server error and was restarted: {err}. "
    "Any file edits you already made are still in {repo_dir} — run "
    "`git -C {repo_dir} diff` to see them, then continue fixing the bug. Do NOT "
    "repeat the same command over and over; if a command already ran, use its "
    "result and move on.]"
)

# Optional, opt-in extra guidance appended to the system prompt. Kept OUT of the
# default so published runs (Ornith/Qwythos/gemma) are unaffected; set it to A/B a
# scaffolding change on a weak model without leaking hidden test names. The bundled
# _SCAFFOLD_INSTRUCTIONS below target the failure modes seen on Ternary-Bonsai
# (hallucinated pytest flags, running modules as scripts, treating a reproducing
# test's non-zero exit as an error). Enable with QT_SWE_EXTRA_INSTRUCTIONS=scaffold
# (the built-in text) or QT_SWE_EXTRA_INSTRUCTIONS="<your own text>".
_SCAFFOLD_INSTRUCTIONS = """\

## Extra operating rules

Running tests: do NOT guess test commands or invent flags. First discover how \
this project runs its tests by inspecting the repo (setup.py, setup.cfg, tox.ini, \
pytest.ini, pyproject.toml, Makefile, or the tests/ layout). Run a single test \
file with a plain `python -m pytest <path/to/test_file.py> -x -q`, or use the \
project's documented runner. Never pass a pytest flag you have not confirmed \
exists (e.g. there is no `--format=json`).

Read before you run: before executing a file, confirm it exists with `ls` / \
`find . -name '<file>'` and read the relevant lines with `sed -n`. Do NOT run a \
library module directly as a script (`python path/to/module.py`) unless it has a \
`if __name__ == "__main__"` block — import it or call it via the test suite \
instead.

Non-zero exits are normal: a failing test exits non-zero. When you are \
REPRODUCING the bug, that failure is the expected, useful signal — not an error \
to avoid. Read the traceback, find the responsible source file, fix it, then \
re-run the SAME test and confirm it now passes.
"""


# SWE-rebench-V2's short language codes -> what to call the repo in the prompt. The
# dataset is 20 languages, so hard-coding "a Python repository" is both wrong and
# actively misleading to the model about which toolchain to reach for.
_LANGUAGE_NAMES = {
    "python": "Python", "go": "Go", "rust": "Rust", "java": "Java", "kotlin": "Kotlin",
    "js": "JavaScript", "ts": "TypeScript", "php": "PHP", "c": "C", "cpp": "C++",
    "csharp": "C#", "scala": "Scala", "swift": "Swift", "dart": "Dart", "julia": "Julia",
    "elixir": "Elixir", "r": "R", "clojure": "Clojure", "ocaml": "OCaml", "lua": "Lua",
}


def language_name(instance: dict) -> str:
    """Human-readable language for the prompt (``''`` -> generic, never wrong)."""
    code = (instance.get("language") or "python").strip().lower()
    name = _LANGUAGE_NAMES.get(code, code.capitalize() if code else "")
    return f"{name} " if name else ""


def _system_prompt(repo_dir: str = "/testbed", language: str = "Python ") -> str:
    """Base prompt plus optional opt-in scaffolding (``QT_SWE_EXTRA_INSTRUCTIONS``).

    ``repo_dir``/``language`` are per instance — see the module docstring.
    """
    base = _SYSTEM_PROMPT.format(repo_dir=repo_dir, language=language)
    extra = os.environ.get("QT_SWE_EXTRA_INSTRUCTIONS", "").strip()
    if not extra:
        return base
    if extra.lower() == "scaffold":
        return base + _SCAFFOLD_INSTRUCTIONS
    return base + "\n\n" + extra


def _truncate_output(text: str, limit: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    """Clip an over-long tool observation, keeping head and tail."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n...[{len(text) - limit} chars truncated]...\n{tail}"


def _run_bash_tool(env: Any, command: str, *, step_timeout: int, counters: dict[str, int],
                   cwd: str) -> str:
    """Execute one bash command in the container, tally use/errors, return output.

    Mirrors the mini-swe ``_MetricsAgent`` bookkeeping: every call bumps
    ``used``; a non-zero return code bumps ``errors``.
    """
    out = env.execute({"command": command}, cwd=cwd, timeout=step_timeout)
    counters["used"] = counters.get("used", 0) + 1
    if int(out.get("returncode", 0) or 0) != 0:
        counters["errors"] = counters.get("errors", 0) + 1
    return _truncate_output(out.get("output", ""))


def _extract_submission(env: Any, *, step_timeout: int, repo_dir: str) -> str:
    """Read the working-tree patch from the container as a git diff."""
    cmd = _SUBMISSION_CMD.format(repo_dir=repo_dir)
    out = env.execute({"command": cmd}, cwd=repo_dir, timeout=step_timeout)
    return out.get("output", "") or ""


def _usage_tuple(usage: Any) -> tuple[int, int, int]:
    """(input, output, total) tokens from an SDK ``Usage`` (zeros if ``None``)."""
    if usage is None:
        return 0, 0, 0
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    tot = int(getattr(usage, "total_tokens", 0) or (inp + out))
    return inp, out, tot


def _item_to_dict(item: Any) -> dict:
    """Best-effort JSON-able dict for an SDK output item (pydantic model or dict)."""
    for attr in ("model_dump", "dict"):
        fn = getattr(item, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(item, dict):
        return item
    return {"repr": str(item)}


class OpenAIAgentsBackend:
    """Drives the OpenAI Agents SDK over one instance."""

    name = "openai-agents"

    def run(self, ctx: AgentRunContext) -> AgentRunResult:
        return asyncio.run(self._run(ctx))

    async def _run(self, ctx: AgentRunContext) -> AgentRunResult:
        from agents import (
            Agent,
            ModelSettings,
            OpenAIChatCompletionsModel,
            RunHooks,
            Runner,
            function_tool,
            set_tracing_disabled,
        )
        from agents.exceptions import MaxTurnsExceeded
        from openai import AsyncOpenAI

        # No OpenAI API key here — disable the SDK's hosted tracing exporter.
        set_tracing_disabled(True)
        # Per-instance checkout path: /testbed (V1) or /<repo-name> (V2). Everything
        # the agent does — cd, git diff, the prompt text — must use THIS path.
        repo_dir = workdir_for(ctx.instance)
        client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)
        model = OpenAIChatCompletionsModel(model=ctx.served_model, openai_client=client)

        counters: dict[str, int] = {"used": 0, "errors": 0}

        @function_tool
        def bash(command: str) -> str:
            """Run a bash command in the repository checkout and return its combined output."""
            return _run_bash_tool(ctx.env, command, step_timeout=ctx.step_timeout,
                                  counters=counters, cwd=repo_dir)

        # Accumulate call count, cumulative token usage, and model-output items
        # via hooks so they survive a MaxTurnsExceeded / wall-timeout — otherwise
        # those metrics (and the trajectory) live only on the RunResult we never
        # get when the run raises. A weak local model that never emits a clean
        # "done" hits max_turns routinely, so this path is the common case.
        class _MetricsHooks(RunHooks):  # type: ignore[misc]
            def __init__(self) -> None:
                self.n_calls = 0
                self.usage: Any = None
                self.items: list[Any] = []

            async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
                self.n_calls += 1
                self.usage = getattr(context, "usage", None)  # cumulative across the run
                out = getattr(response, "output", None)
                if out:
                    self.items.extend(out)

        hooks = _MetricsHooks()

        # llama.cpp-only sampling extensions (top_k/min_p/repeat_penalty) are not
        # forwarded: the SDK's ModelSettings exposes temperature/top_p as the
        # first-class knobs, which are the ones that matter for SWE agents.
        settings = ModelSettings(
            temperature=ctx.sampling.temperature,
            top_p=ctx.sampling.top_p,
            max_tokens=ctx.max_tokens,
        )
        agent = Agent(
            name="swe-agent",
            instructions=_system_prompt(repo_dir, language_name(ctx.instance)),
            model=model,
            tools=[bash],
            model_settings=settings,
        )

        # Run the agent, then apply the empty-patch completion gate: if it stops
        # cleanly (``completed``) with an unchanged working tree, resume the same
        # conversation with a forcing nudge and run again, up to
        # _MAX_EMPTY_PATCH_RETRIES times — bounded by the instance wall deadline.
        # to_input_list() yields the SDK's typed input-item dicts (plain dicts at
        # runtime); keep the local loose so they flow into the result.
        deadline = time.monotonic() + ctx.instance_timeout
        messages: list[Any] = []
        exit_status = "completed"
        agent_input: Any = ctx.instance["problem_statement"]
        prompt_tokens = completion_tokens = total_tokens = 0
        n_nudges = 0

        def _accumulate_usage() -> None:
            # hooks.usage is cumulative *within* one Runner.run; snapshot and reset
            # it so token totals sum correctly across nudge continuations.
            nonlocal prompt_tokens, completion_tokens, total_tokens
            i, o, t = _usage_tuple(hooks.usage)
            prompt_tokens += i
            completion_tokens += o
            total_tokens += t
            hooks.usage = None

        attempt = 0
        n_errors = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                exit_status = "wall_timeout"
                break
            try:
                result = await asyncio.wait_for(
                    Runner.run(agent, agent_input, max_turns=ctx.max_steps, hooks=hooks),
                    timeout=remaining,
                )
                exit_status = "completed"
                messages = list(result.to_input_list())
            except TimeoutError:
                exit_status = "wall_timeout"
                _accumulate_usage()
                break
            except MaxTurnsExceeded:
                exit_status = "max_turns"
                _accumulate_usage()
                break
            except Exception as e:  # retriable server error -> feed back + restart
                _accumulate_usage()
                if n_errors < _MAX_ERROR_RETRIES and (deadline - time.monotonic()) > 0:
                    n_errors += 1
                    exit_status = f"retried:{type(e).__name__}"
                    agent_input = ctx.instance["problem_statement"] + _ERROR_RETRY_NOTE.format(
                        err=f"{type(e).__name__}: {str(e)[:200]}", repo_dir=repo_dir)
                    continue
                exit_status = f"error:{type(e).__name__}"
                break
            _accumulate_usage()

            # Completion gate: a non-empty diff (or exhausted retries) ends it;
            # an empty diff on a clean stop resumes with the forcing nudge.
            if _extract_submission(ctx.env, step_timeout=ctx.step_timeout,
                                   repo_dir=repo_dir).strip():
                break
            if attempt >= _MAX_EMPTY_PATCH_RETRIES:
                break
            attempt += 1
            n_nudges += 1
            agent_input = messages + [{"role": "user",
                                       "content": _EMPTY_PATCH_NUDGE.format(repo_dir=repo_dir)}]

        # Metrics from the hooks so they survive the exception paths above.
        n_calls = hooks.n_calls
        # Trajectory: prefer the complete to_input_list(); else fall back to the
        # accumulated model-output items (assistant text + tool calls — tool
        # outputs aren't in this fallback, but it beats an empty trajectory).
        if not messages and hooks.items:
            messages = [_item_to_dict(it) for it in hooks.items]

        # Read the patch regardless of how the loop ended — the agent may have
        # made edits before hitting the step/wall limit.
        submission = _extract_submission(ctx.env, step_timeout=ctx.step_timeout,
                                         repo_dir=repo_dir)

        with contextlib.suppress(Exception):
            ctx.trajectory_path.write_text(
                json.dumps(
                    {
                        "messages": messages,
                        "exit_status": exit_status,
                        "n_model_calls": n_calls,
                        "n_empty_patch_nudges": n_nudges,
                        "n_error_retries": n_errors,
                    },
                    indent=2,
                    default=str,
                )
            )

        return AgentRunResult(
            submission=submission,
            messages=messages,
            n_model_calls=n_calls,
            tools_used=counters["used"],
            tool_errors=counters["errors"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            exit_status=exit_status,
            extra={"n_empty_patch_nudges": n_nudges, "n_error_retries": n_errors},
        )

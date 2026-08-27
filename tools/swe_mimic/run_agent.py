"""Docker-free SWE-rebench mimic: OpenAI Agents SDK -> local llama-server -> local repo.

Mirrors the shape of ``quant_tuner.eval.agents.openai_agents`` but shells into a plain
working directory instead of a per-instance Docker container, because this box is an
unprivileged container with no container runtime.

    ** These numbers are a SMOKE TEST, not SWE-rebench. **

Differences from the official harness, all of which can move the score:
  * no container isolation — the agent runs bash on the host, in the repo dir
  * dependency versions resolved against the host toolchain, not the pinned image
  * a single instance and a single rep, so there is no run-to-run variance estimate

What is faithful: the repo is checked out at ``base_commit``, the official ``test_patch``
is applied, the agent only sees the problem statement, and grading runs the instance's own
``test_cmd`` over its recorded FAIL_TO_PASS / PASS_TO_PASS ids. The golden gate (F2P fails
and P2P pass before the agent runs) is enforced inline on every episode and aborts with
exit 2 rather than producing a score — a broken grader is indistinguishable from a model
that changed nothing, and has already produced one round of fabricated results.

    .venv/bin/python run_agent.py --model-name IQ4_XS --base-url http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# A non-zero exit is NOT evidence the model issued a bad command. The system prompt tells
# the agent to run the failing test (exit 1 by construction) and to grep around (exit 1 on
# no match) — scoring those as "tool errors" penalises exactly the behaviour we asked for,
# and rewards a model that patches blind. Only count exits where the command never really
# ran: the shell couldn't find/parse it, or it hit the wall clock.
_MALFORMED = re.compile(
    r"command not found"
    r"|: not found"
    r"|unexpected (?:EOF|token)"
    r"|syntax error"
    r"|bash: -c: line"
    r"|Permission denied"
    # A failed `cd` means everything after the `&&` never ran, so the intended work did
    # not happen — that is a real agent error, unlike a grep that ran and matched nothing.
    # Qwen3.8 reaches for `cd /testbed` here: the SWE-rebench *Docker* workdir, which this
    # Docker-free mimic does not have.
    r"|can't cd to"
    r"|cd: .*(?:No such file|not a directory)",
    re.I,
)


def classify(out: str, rc: int) -> str:
    """ok | timeout | malformed | nonzero (the program ran and reported a result)."""
    if rc == 0:
        return "ok"
    if rc == 124:
        return "timeout"
    if rc in (126, 127) or _MALFORMED.search(out):
        return "malformed"
    return "nonzero"

SYSTEM = """You are a software engineer fixing a bug in a Python repository.

Your shell starts in the repository root, which is `{repo}`. There is NO /testbed directory
on this machine — use paths relative to the repository root. A test has been added that
currently FAILS; your job is to
change the SOURCE code so it passes, without breaking existing tests.

Use the `bash` tool to explore, edit, and run tests. Work incrementally:
  1. Find the relevant source file (grep/find).
  2. Read the failing test to understand exactly what is expected.
  3. Make a minimal, correct edit to the source.
  4. Re-run the failing test to confirm.

Edit files with `python - <<'EOF' ... EOF` heredocs or `sed`. Do NOT edit test files.
When the failing test passes, say DONE."""


def sh(cmd: str, cwd: Path, env: dict, timeout: int = 180,
       limit: int | None = 6000) -> tuple[str, int]:
    """Run a shell command; keep only the last ``limit`` chars (None = keep all).

    The cap exists to stop a chatty command from eating the agent's context. The
    GRADER must pass ``limit=None``: it counts `PASSED <id>` lines, and this
    instance's P2P run emits 10,408 chars against a 6,000-char cap. Every one of
    the 34 lines happens to survive here by ~1.7x, but a larger suite would lose
    them and report a false *unresolved* — a silent pessimistic bias that looks
    exactly like a quant that broke.
    """
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(cwd), env=env, timeout=timeout,
                           capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        return (out if limit is None else out[-limit:]), p.returncode
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]", 124


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, default=HERE / "instance.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model-name", default="local")
    ap.add_argument("--label", default=None, help="label for the results row")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reasoning-budget", type=int, default=2048,
                    help="token budget for <think> per turn; -1 unrestricted, 0 disables "
                         "thinking entirely. Enforced server-side by llama.cpp's reasoning "
                         "budget sampler, NOT by truncating the response.")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='JSON forwarded via extra_body, e.g. \'{"enable_thinking":false}\'. '
                         "Use this on vLLM: --reasoning-budget is a llama.cpp-only "
                         "extension and is silently ignored there, so reasoning runs "
                         "unbounded and can consume the whole max_tokens before any tool "
                         "call is emitted.")
    ap.add_argument("--out", type=Path, default=HERE / "swe_mimic_results.csv")
    ap.add_argument("--skip-gate", action="store_true",
                    help="skip the pre-run golden gate (F2P fails / P2P pass at base). "
                         "Only for a deliberately dirty tree — the gate is what "
                         "distinguishes a model that changed nothing from a grader that "
                         "never ran.")
    a = ap.parse_args()

    from agents import (
        Agent,
        ModelSettings,
        OpenAIChatCompletionsModel,
        RunHooks,
        Runner,
        function_tool,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)

    inst = json.loads(a.instance.read_text())
    iid = inst["instance_id"]
    work = HERE / "work" / iid
    repo = work / "repo"
    venv = work / "venv"
    label = a.label or a.model_name

    # reset the repo to a clean base + test_patch, so reps are independent
    subprocess.run(["git", "checkout", "--force", inst["base_commit"]], cwd=repo,
                   capture_output=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=repo, capture_output=True)
    subprocess.run(["git", "apply", str(work / "test_patch.diff")], cwd=repo,
                   capture_output=True)
    # STAGE the test patch. `git diff` (working tree vs index) is read later as "the
    # agent's patch"; with the test patch left unstaged it lands in that diff, so EVERY
    # episode reported patch_produced=1 -- including one where the agent made zero tool
    # calls. Staging it makes the later diff contain exactly what the agent changed.
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)

    env = dict(os.environ)
    env["PATH"] = f"{venv / 'bin'}:{env['PATH']}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    ic = inst["install_config"]
    test_cmd = ic.get("test_cmd") or "pytest"
    f2p = inst["FAIL_TO_PASS"]
    p2p = inst["PASS_TO_PASS"]
    f2p = json.loads(f2p) if isinstance(f2p, str) else f2p
    p2p = json.loads(p2p) if isinstance(p2p, str) else p2p

    def run_ids(ids: list[str]) -> tuple[int, int]:
        if not ids:
            return 0, 0
        passed = 0
        # one pytest invocation, ids quoted
        joined = " ".join(f"'{i}'" for i in ids)
        out, _rc = sh(f"{test_cmd} {joined}", repo, env, timeout=1800, limit=None)
        for i in ids:
            if f"PASSED {i}" in out or f"{i} PASSED" in out:
                passed += 1
        return passed, len(ids)

    # ---- golden gate: the grader must be working BEFORE the model is blamed ------------
    # Without this the harness cannot tell "the model changed nothing" from "pytest never
    # ran". Observed: the instance venv's interpreter was a dangling symlink (its CPython
    # was removed from the machine), so every test invocation failed to start and two
    # models were both recorded as f2p 0/1, p2p 0/34 -- which reads as "both models broke
    # 34 regression tests" and is entirely an artifact. P2P passes at the base commit by
    # construction, so a non-perfect P2P here is always the environment, never the model.
    # Costs ~0.4s on this instance. --skip-gate exists for a deliberately dirty tree.
    if not a.skip_gate:
        g_p2p, g_p2p_n = run_ids(p2p)
        g_f2p, g_f2p_n = run_ids(f2p)
        if g_p2p != g_p2p_n:
            print(f"GOLDEN GATE FAILED: {g_p2p}/{g_p2p_n} PASS_TO_PASS at the base commit "
                  f"(expected all). The test environment is broken, not the model.\n"
                  f"  check: {venv / 'bin' / 'python'} -c 'import pytest'", flush=True)
            return 2
        if g_f2p != 0:
            print(f"GOLDEN GATE FAILED: {g_f2p}/{g_f2p_n} FAIL_TO_PASS already pass at the "
                  f"base commit. The bug is not present, so 'resolved' would be free.",
                  flush=True)
            return 2
        print(f"[gate] ok: F2P 0/{g_f2p_n} pass, P2P {g_p2p}/{g_p2p_n} pass at base",
              flush=True)

    stats = {"tool_calls": 0, "nonzero": 0, "malformed": 0, "timeout": 0,
             "in_tok": 0, "out_tok": 0}
    trajectory: list[dict] = []

    @function_tool
    def bash(command: str) -> str:
        """Run a bash command in the repository root and return combined stdout+stderr."""
        stats["tool_calls"] += 1
        out, rc = sh(command, repo, env)
        kind = classify(out, rc)
        if kind != "ok":
            stats[kind] += 1
        # the full command is recorded so an error rate is auditable rather than asserted
        trajectory.append({"n": stats["tool_calls"], "cmd": command, "rc": rc,
                           "kind": kind, "out_head": out[:400]})
        return f"(exit {rc})\n{out}"

    class Hooks(RunHooks):
        async def on_llm_end(self, context, agent, response):  # noqa: ANN001
            u = getattr(response, "usage", None)
            if u:
                stats["in_tok"] += getattr(u, "input_tokens", 0) or 0
                stats["out_tok"] += getattr(u, "output_tokens", 0) or 0

    client = AsyncOpenAI(base_url=a.base_url, api_key="sk-no-key")
    agent = Agent(
        name="swe-fixer",
        # The real SWE-rebench harness runs the agent inside a container whose checkout IS
        # at /testbed, so a model reaching for it is right there and wrong only here.
        # Naming the actual root keeps this mimic from scoring a Docker-shaped habit as a
        # quantization defect.
        instructions=SYSTEM.format(repo=repo),
        tools=[bash],
        model=OpenAIChatCompletionsModel(model=a.model_name, openai_client=client),
        # max_tokens matches eval.swebench.DEFAULT_MAX_TOKENS. Without a cap a rung that
        # falls into a repetition loop generates until the 32k context is exhausted —
        # ~5.5 min per turn at 99 tok/s, so one episode can run for hours and the loop is
        # scored as "slow" rather than as the degradation it is.
        model_settings=ModelSettings(
            temperature=a.temperature, top_p=a.top_p, max_tokens=8096,
            # thinking_budget_tokens is a llama.cpp server extension (server-common.cpp
            # reads it off the request body), so it rides in extra_body rather than being
            # a first-class ModelSettings field. Sent per-request on purpose: one server
            # can then serve budgeted and unbudgeted episodes for an A/B.
            # thinking_budget_tokens is llama.cpp-only. On vLLM it is silently ignored,
            # so reasoning runs unbounded and the model can burn the whole max_tokens on a
            # single <think> block without ever emitting a tool call (observed: 8096 out
            # tokens, 0 steps). --chat-template-kwargs is the portable lever there.
            extra_body={"thinking_budget_tokens": a.reasoning_budget,
                        **({"chat_template_kwargs": json.loads(a.chat_template_kwargs)}
                           if a.chat_template_kwargs else {})},
        ),
    )

    prompt = (f"Repository: {inst['repo']} (checked out at the failing commit)\n\n"
              f"# Issue\n{inst['problem_statement']}\n\n"
              f"Fix the source so the newly added failing test passes.")

    t0 = time.time()
    exit_status = "completed"
    try:
        await Runner.run(agent, prompt, max_turns=a.max_turns, hooks=Hooks())
    except Exception as e:  # noqa: BLE001
        exit_status = f"error:{type(e).__name__}"
        print(f"[agent ended: {exit_status}: {str(e)[:200]}]")
    wall = time.time() - t0

    # ---- collect the agent's patch ------------------------------------------------------
    # The test patch is staged (see the reset block), so this diff is the agent's work
    # alone. Anything under a test path is still excluded: the agent editing the test to
    # match its own output is not a fix, and the official harness grades source changes.
    diff, _ = sh("git diff --", repo, env)
    patch_produced = bool(diff.strip())

    # ---- grade with the instance's own test_cmd over its recorded ids ------------------
    f2p_pass, f2p_n = run_ids(f2p)
    p2p_pass, p2p_n = run_ids(p2p)
    resolved = (f2p_pass == f2p_n and f2p_n > 0) and (p2p_pass == p2p_n)

    row = {
        "label": label,
        "instance_id": iid,
        "resolved": int(resolved),
        "patch_produced": int(patch_produced),
        "f2p_passed": f2p_pass, "f2p_total": f2p_n,
        "p2p_passed": p2p_pass, "p2p_total": p2p_n,
        "steps": stats["tool_calls"],
        # headline: commands the shell could not run at all (bad syntax / no such binary /
        # timeout). This is the only count that reflects the model issuing a bad command.
        "tool_errors": stats["malformed"] + stats["timeout"],
        "tool_err_rate": round((stats["malformed"] + stats["timeout"])
                               / max(1, stats["tool_calls"]), 4),
        "malformed": stats["malformed"],
        "timeouts": stats["timeout"],
        # a command that ran and reported failure — usually the failing test being
        # reproduced, or a grep with no match. Diagnostic, NOT an error.
        "nonzero_exits": stats["nonzero"],
        "out_tokens": stats["out_tok"],
        "in_tokens": stats["in_tok"],
        "wall_s": round(wall, 1),
        "exit_status": exit_status,
        "hit_max_turns": int(exit_status.startswith("error:MaxTurns")),
        "reasoning_budget": a.reasoning_budget,
    }
    (work / f"patch_{label}.diff").write_text(diff)
    (work / f"result_{label}.json").write_text(json.dumps(row, indent=2))
    (work / f"traj_{label}.json").write_text(json.dumps(trajectory, indent=2))

    import csv
    new = not a.out.exists()
    with a.out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    print("\n" + "=" * 60)
    for k, v in row.items():
        print(f"  {k:16s} {v}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

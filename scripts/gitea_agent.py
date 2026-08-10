#!/usr/bin/env python3
"""Standalone autonomous issue-fixing agent for a self-hosted Gitea.

A single-file port of the OpenAI-Agents-SDK SWE-rebench backend used in
quant-tuner, rewired for a home Gitea deployment:

  * issue source : Gitea REST API (instead of a SWE-rebench JSONL holdout)
  * environment  : a local `git clone` working tree driven by a `bash` tool
                   (instead of a per-instance Docker container)
  * model        : any OpenAI-compatible endpoint (your local `llama-server`,
                   vLLM, Ollama's /v1, OpenAI, etc.)

The agent gets the issue title+body as its task and one `bash` tool that runs in
the checkout. It explores, edits, and (if present) runs tests, then its changes
are collected from the working tree as a `git diff`. Optionally the script
commits that diff to a branch, pushes it, opens a pull request, and/or comments
the patch back on the issue.

Dependencies (only one, plus stdlib):
    pip install openai-agents          # pulls the `openai` client too

Quick start:
    export GITEA_URL=http://localhost:3000
    export GITEA_TOKEN=xxxxxxxx        # Gitea → Settings → Applications → token
    # point at your local model server (default below is llama-server's port)
    python gitea_agent.py --repo myorg/myrepo --issue 42 \
        --base-url http://localhost:8080/v1 --model local \
        --apply            # commit→push branch→open PR (omit for a dry-run diff)

    # or sweep every open issue carrying a label:
    python gitea_agent.py --repo myorg/myrepo --label agent --comment

⚠️  SECURITY: the model's `bash` commands run on THIS host inside the checkout.
    Only point it at repos/issues you trust, or run the whole script inside a
    throwaway container/VM. There is no command sandbox here by design.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Cap a single tool observation so a runaway command can't blow the context.
_MAX_TOOL_OUTPUT_CHARS = 16000


def _default_system_prompt(workdir: str) -> str:
    return f"""\
You are an autonomous software engineer resolving an issue in a git repository \
that is already checked out at {workdir} (your current working directory).

You have one tool: bash(command). Use it to explore the code, reproduce the \
problem, edit files, and run the project's tests if it has any. Make all edits \
directly on the files in the checkout with shell commands (e.g. your language's \
tooling, sed, or writing files via heredocs). Work in small steps and inspect \
each command's output before continuing.

Prefer fixing the actual source/implementation that causes the issue. Reproduce \
the problem first, make a focused change, then re-run to confirm the change \
actually resolves it. Keep the change minimal and in keeping with the \
surrounding code style.

Do NOT run `git commit`, `git checkout`, `git reset`, or `git push` — your final \
patch is collected automatically from the working tree as a git diff. When the \
fix is complete and you have verified it, stop and give a short summary of what \
you changed and why.
"""


def _truncate_output(text: str | None, limit: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    """Clip an over-long tool observation, keeping head and tail."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n...[{len(text) - limit} chars truncated]...\n{tail}"


# --------------------------------------------------------------------------- #
# Local working-tree environment (replaces the SWE-rebench Docker container)
# --------------------------------------------------------------------------- #


@dataclass
class LocalBashEnv:
    """Runs bash commands in a fixed working directory on the host."""

    workdir: Path

    def execute(self, command: str, *, timeout: int) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return {"output": output, "returncode": proc.returncode}
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
            return {"output": f"{partial}\n[command timed out after {timeout}s]", "returncode": 124}
        except Exception as e:  # noqa: BLE001
            return {"output": f"[failed to run command: {e}]", "returncode": 1}


# --------------------------------------------------------------------------- #
# Gitea REST client (stdlib only)
# --------------------------------------------------------------------------- #


class GiteaClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base = base_url.rstrip("/")
        self.api = f"{self.base}/api/v1"
        self.token = token

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        url = path if path.startswith("http") else f"{self.api}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"token {self.token}")
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"Gitea {method} {url} -> {e.code}: {detail}") from e

    def get_repo(self, owner: str, repo: str) -> dict:
        return self._request("GET", f"/repos/{owner}/{repo}")

    def get_issue(self, owner: str, repo: str, index: int) -> dict:
        return self._request("GET", f"/repos/{owner}/{repo}/issues/{index}")

    def list_open_issues(self, owner: str, repo: str, label: str | None, limit: int) -> list[dict]:
        q = {"state": "open", "type": "issues", "limit": str(limit)}
        if label:
            q["labels"] = label
        qs = urllib.parse.urlencode(q)
        items = self._request("GET", f"/repos/{owner}/{repo}/issues?{qs}") or []
        # `type=issues` already excludes PRs, but guard anyway.
        return [i for i in items if "pull_request" not in i or not i.get("pull_request")]

    def comment_issue(self, owner: str, repo: str, index: int, body: str) -> dict:
        return self._request(
            "POST", f"/repos/{owner}/{repo}/issues/{index}/comments", {"body": body}
        )

    def create_pull(
        self, owner: str, repo: str, head: str, base: str, title: str, body: str
    ) -> dict:
        return self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls",
            {"head": head, "base": base, "title": title, "body": body},
        )


# --------------------------------------------------------------------------- #
# The agent loop (OpenAI Agents SDK) — faithful port of the swebench backend
# --------------------------------------------------------------------------- #


@dataclass
class AgentResult:
    summary: str
    patch: str
    exit_status: str
    n_model_calls: int
    tools_used: int
    tool_errors: int
    total_tokens: int
    messages: list[Any] = field(default_factory=list)


def _usage_tuple(usage: Any) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    tot = int(getattr(usage, "total_tokens", 0) or (inp + out))
    return inp, out, tot


def _item_to_dict(item: Any) -> dict:
    for attr in ("model_dump", "dict"):
        fn = getattr(item, attr, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return fn()
    return item if isinstance(item, dict) else {"repr": str(item)}


async def _run_agent_async(
    *,
    task: str,
    env: LocalBashEnv,
    base_url: str,
    api_key: str,
    served_model: str,
    system_prompt: str,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    max_steps: int,
    step_timeout: int,
    instance_timeout: int,
) -> AgentResult:
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

    set_tracing_disabled(True)  # no hosted tracing without an OpenAI key
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    model = OpenAIChatCompletionsModel(model=served_model, openai_client=client)

    counters: dict[str, int] = {"used": 0, "errors": 0}

    @function_tool
    def bash(command: str) -> str:
        """Run a bash command in the repository checkout and return its combined output."""
        out = env.execute(command, timeout=step_timeout)
        counters["used"] += 1
        if int(out.get("returncode", 0) or 0) != 0:
            counters["errors"] += 1
        return _truncate_output(out.get("output", ""))

    # Hooks accumulate calls/tokens/items so they survive a MaxTurnsExceeded /
    # wall-timeout (a weak local model hitting max_turns is the common path).
    class _MetricsHooks(RunHooks):  # type: ignore[misc]
        def __init__(self) -> None:
            self.n_calls = 0
            self.usage: Any = None
            self.items: list[Any] = []

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            self.n_calls += 1
            self.usage = getattr(context, "usage", None)
            out = getattr(response, "output", None)
            if out:
                self.items.extend(out)

    hooks = _MetricsHooks()
    settings = ModelSettings(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    agent = Agent(
        name="gitea-agent",
        instructions=system_prompt,
        model=model,
        tools=[bash],
        model_settings=settings,
    )

    messages: list[Any] = []
    summary = ""
    exit_status = "completed"
    try:
        result = await asyncio.wait_for(
            Runner.run(agent, task, max_turns=max_steps, hooks=hooks),
            timeout=instance_timeout,
        )
        messages = list(result.to_input_list())
        summary = str(getattr(result, "final_output", "") or "")
    except TimeoutError:
        exit_status = "wall_timeout"
    except MaxTurnsExceeded:
        exit_status = "max_turns"
    except Exception as e:  # noqa: BLE001
        exit_status = f"error:{type(e).__name__}"

    if not messages and hooks.items:
        messages = [_item_to_dict(it) for it in hooks.items]

    _, _, total_tokens = _usage_tuple(hooks.usage)

    # Snapshot the working tree as a patch (stages new files so they show up).
    snap = env.execute("git add -A && git diff --cached HEAD", timeout=step_timeout)
    patch = snap.get("output", "") or ""

    return AgentResult(
        summary=summary,
        patch=patch,
        exit_status=exit_status,
        n_model_calls=hooks.n_calls,
        tools_used=counters["used"],
        tool_errors=counters["errors"],
        total_tokens=total_tokens,
        messages=messages,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _clone_url_with_token(clone_url: str, token: str) -> str:
    """Embed the token so non-interactive clone/push works (token as username)."""
    parts = urllib.parse.urlsplit(clone_url)
    netloc = f"{token}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def solve_issue(
    gitea: GiteaClient,
    owner: str,
    repo: str,
    issue: dict,
    args: argparse.Namespace,
    base_dir: Path,
) -> dict:
    index = issue["number"]
    title = issue.get("title", "") or ""
    body = issue.get("body", "") or ""
    task = f"# Issue #{index}: {title}\n\n{body}".strip()
    print(f"\n=== issue #{index}: {title!r} ===", flush=True)

    repo_meta = gitea.get_repo(owner, repo)
    base_branch = args.base_branch or repo_meta.get("default_branch", "main")
    clone_url = repo_meta["clone_url"]

    workdir = base_dir / f"{repo}-issue-{index}"
    if workdir.exists():
        print(f"  reusing existing checkout {workdir}", flush=True)
    else:
        print(f"  cloning {clone_url} @ {base_branch} -> {workdir}", flush=True)
        auth_url = _clone_url_with_token(clone_url, args.gitea_token)
        cp = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", base_branch, auth_url, str(workdir)],
            capture_output=True, text=True,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"git clone failed: {cp.stderr}")

    env = LocalBashEnv(workdir=workdir)
    result = asyncio.run(
        _run_agent_async(
            task=task,
            env=env,
            base_url=args.base_url,
            api_key=args.api_key,
            served_model=args.model,
            system_prompt=args.system_prompt or _default_system_prompt(str(workdir)),
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            step_timeout=args.step_timeout,
            instance_timeout=args.instance_timeout,
        )
    )

    patch_chars = len(result.patch.strip())
    print(
        f"  exit={result.exit_status} patch={'yes' if patch_chars else 'no'} "
        f"calls={result.n_model_calls} tools={result.tools_used} "
        f"tool_err={result.tool_errors} tokens={result.total_tokens}",
        flush=True,
    )

    # Persist artifacts next to the checkout.
    (workdir.parent / f"issue-{index}.patch").write_text(result.patch)
    if args.save_trajectory:
        (workdir.parent / f"issue-{index}.traj.json").write_text(
            json.dumps(
                {"messages": result.messages, "exit_status": result.exit_status,
                 "summary": result.summary},
                indent=2, default=str,
            )
        )

    pr_url = None
    if patch_chars and args.apply:
        pr_url = _commit_push_pr(gitea, owner, repo, index, title, base_branch, result, args, workdir)
    if patch_chars and args.comment:
        note = (
            f"🤖 Agent proposed a patch for this issue "
            f"(exit={result.exit_status}, {result.n_model_calls} model calls).\n\n"
            f"{result.summary}\n\n<details><summary>diff</summary>\n\n"
            f"```diff\n{result.patch[:60000]}\n```\n</details>"
        )
        gitea.comment_issue(owner, repo, index, note)
        print(f"  commented patch on issue #{index}", flush=True)

    return {
        "issue": index,
        "title": title,
        "exit_status": result.exit_status,
        "patch_chars": patch_chars,
        "n_model_calls": result.n_model_calls,
        "tools_used": result.tools_used,
        "tool_errors": result.tool_errors,
        "total_tokens": result.total_tokens,
        "pr_url": pr_url,
    }


def _commit_push_pr(
    gitea: GiteaClient, owner: str, repo: str, index: int, title: str,
    base_branch: str, result: AgentResult, args: argparse.Namespace, workdir: Path,
) -> str | None:
    branch = f"agent/issue-{index}"
    # The agent was told not to touch git; we commit its working-tree edits here.
    for cmd in (["checkout", "-B", branch], ["add", "-A"]):
        cp = _git(cmd, workdir)
        if cp.returncode != 0:
            raise RuntimeError(f"git {' '.join(cmd)} failed: {cp.stderr}")
    _git(["-c", "user.name=gitea-agent", "-c", "user.email=agent@localhost",
          "commit", "-m", f"Fix #{index}: {title}"], workdir)
    push = _git(["push", "-u", "origin", branch, "--force"], workdir)
    if push.returncode != 0:
        raise RuntimeError(f"git push failed: {push.stderr}")
    pr = gitea.create_pull(
        owner, repo, head=branch, base=base_branch,
        title=f"Fix #{index}: {title}",
        body=f"Automated fix for #{index}.\n\n{result.summary}\n\nCloses #{index}",
    )
    url = pr.get("html_url")
    print(f"  opened PR: {url}", flush=True)
    return url


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Gitea
    p.add_argument("--gitea-url", default=os.environ.get("GITEA_URL"),
                   help="Gitea base URL, e.g. http://localhost:3000 (env GITEA_URL)")
    p.add_argument("--gitea-token", default=os.environ.get("GITEA_TOKEN"),
                   help="Gitea API token (env GITEA_TOKEN)")
    p.add_argument("--repo", required=True, help="owner/name of the repository")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--issue", type=int, help="single issue number to solve")
    g.add_argument("--label", help="solve every OPEN issue carrying this label")
    g.add_argument("--all-open", action="store_true", help="solve every OPEN issue")
    p.add_argument("--limit", type=int, default=20, help="max issues when sweeping")
    # Model / endpoint
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"),
                   help="OpenAI-compatible endpoint (default: local llama-server)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    p.add_argument("--model", default="local", help="served model id sent to the endpoint")
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=32768)
    # Loop limits
    p.add_argument("--max-steps", type=int, default=100, help="max agent turns per issue")
    p.add_argument("--step-timeout", type=int, default=120, help="seconds per bash command")
    p.add_argument("--instance-timeout", type=int, default=7200, help="wall seconds per issue")
    # Actions
    p.add_argument("--apply", action="store_true",
                   help="commit the diff to a branch, push, and open a PR")
    p.add_argument("--comment", action="store_true",
                   help="post the diff as a comment on the issue")
    p.add_argument("--base-branch", default=None, help="base branch (default: repo default)")
    p.add_argument("--workdir", default=None, help="where to clone (default: a temp dir)")
    p.add_argument("--system-prompt", default=None, help="override the agent system prompt")
    p.add_argument("--save-trajectory", action="store_true", help="dump the conversation JSON")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not args.gitea_url or not args.gitea_token:
        print("ERROR: set --gitea-url/--gitea-token (or GITEA_URL/GITEA_TOKEN).", file=sys.stderr)
        return 2
    if "/" not in args.repo:
        print("ERROR: --repo must be owner/name.", file=sys.stderr)
        return 2
    owner, repo = args.repo.split("/", 1)

    gitea = GiteaClient(args.gitea_url, args.gitea_token)
    base_dir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="gitea-agent-"))
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"working dir: {base_dir}", flush=True)

    if args.issue is not None:
        issues = [gitea.get_issue(owner, repo, args.issue)]
    else:
        issues = gitea.list_open_issues(owner, repo, args.label, args.limit)
        print(f"found {len(issues)} open issue(s)"
              + (f" with label {args.label!r}" if args.label else ""), flush=True)

    results = []
    for issue in issues:
        try:
            results.append(solve_issue(gitea, owner, repo, issue, args, base_dir))
        except Exception as e:  # noqa: BLE001 — keep going across issues
            print(f"  ERROR on issue #{issue.get('number')}: {e}", flush=True)
            results.append({"issue": issue.get("number"), "error": str(e)})

    (base_dir / "summary.json").write_text(json.dumps(results, indent=2))
    solved = sum(1 for r in results if r.get("patch_chars"))
    print(f"\nDone. {solved}/{len(results)} issue(s) produced a patch. "
          f"Artifacts in {base_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

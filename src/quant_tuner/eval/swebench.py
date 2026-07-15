"""Agentic SWE-rebench benchmark: can a quant actually solve real GitHub issues?

Drives `mini-swe-agent <https://github.com/SWE-agent/mini-swe-agent>`_ against
SWE-rebench instances, each in a clean per-instance Docker container, with the
model served by a local ``llama-server`` over the OpenAI-compatible endpoint
(``eval.server.running_server``). For every instance we:

    1. spin up the instance's Docker image (``DockerEnvironment``);
    2. let the agent read/run/edit until it submits a ``git diff`` patch (or
       hits the step / wall-time limit);
    3. grade the patch by running the gold ``FAIL_TO_PASS`` / ``PASS_TO_PASS``
       tests in the container (``swebench_grade.grade_instance``);
    4. record metrics — token usage, tool (bash) calls, tool errors, whether a
       patch was produced (``patch_rate``) and whether it resolved the issue
       (``pass_rate``) — and save the full conversation trajectory to disk.

Mirrors the shape of :mod:`quant_tuner.eval.toolcall` / :mod:`~.mmlu_pro`:
a float-metrics ``SweSummary`` dataclass + ``run_swebench_eval(holdout,
model_path=… | base_url=…)`` (mutually exclusive) + a thin ``swebench_rep``
adapter for :mod:`quant_tuner.eval.reps`.

Requires the ``swebench`` extra (``mini-swe-agent``) and a running Docker
daemon. On Apple Silicon the SWE-rebench linux/amd64 images run under emulation
(slow but correct) — see ``CLAUDE.md``.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_tuner.eval.agents import get_backend
from quant_tuner.eval.agents.base import AgentRunContext
from quant_tuner.eval.server import running_server
from quant_tuner.eval.swebench_grade import grade_instance
from quant_tuner.eval.toolcall import Sampling

DEFAULT_MAX_STEPS = 100
DEFAULT_INSTANCE_TIMEOUT = 7200  # wall-clock seconds per instance (2 h for 100-step runs)
DEFAULT_STEP_TIMEOUT = 120  # seconds for a single bash command in the container
DEFAULT_MAX_TOKENS = 8096  # per model call (agent reasoning + tool call)
DEFAULT_TEMPERATURE = 0.25  # a little sampling beats greedy for agent loops (avoids tight loops)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass
class SweSummary:
    """Aggregate metrics returned by :func:`run_swebench_eval`."""

    model: str
    n_instances: int
    pass_rate: float  # fraction of instances whose patch resolved the issue
    patch_rate: float  # fraction that produced a non-empty diff
    mean_tokens: float  # mean (prompt+completion) tokens per instance
    total_tokens: float
    mean_steps: float  # mean bash/tool calls per instance
    tool_error_rate: float  # tool errors ÷ tool calls (across all instances)
    mean_wall_sec: float
    n_resolved: int = 0
    n_patched: int = 0
    per_instance: list[dict] = field(default_factory=list)

    def scalar_metrics(self) -> dict[str, float]:
        """The subset that plugs into :mod:`quant_tuner.eval.reps`."""
        return {
            "pass_rate": self.pass_rate,
            "patch_rate": self.patch_rate,
            "mean_tokens": self.mean_tokens,
            "mean_steps": self.mean_steps,
            "tool_error_rate": self.tool_error_rate,
        }


# ---------------------------------------------------------------------------
# Holdout / config helpers
# ---------------------------------------------------------------------------


def load_holdout(path: Path) -> list[dict]:
    """Load a SWE-rebench holdout JSONL (one instance dict per line)."""
    instances: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def _sampling_to_model_kwargs(sampling: Sampling, *, max_tokens: int) -> dict[str, Any]:
    """Render a :class:`Sampling` as litellm ``model_kwargs``.

    Standard OpenAI params go top-level; llama.cpp extensions (``top_k``,
    ``min_p``, ``repeat_penalty``) ride ``extra_body`` exactly as
    :meth:`Sampling.to_request_kwargs` does for the tool-call eval.
    """
    mk: dict[str, Any] = {"temperature": sampling.temperature, "max_tokens": max_tokens}
    if sampling.top_p is not None:
        mk["top_p"] = sampling.top_p
    if sampling.presence_penalty is not None:
        mk["presence_penalty"] = sampling.presence_penalty
    if sampling.seed is not None:
        mk["seed"] = sampling.seed
    extra: dict[str, Any] = {}
    if sampling.top_k is not None:
        extra["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        extra["min_p"] = sampling.min_p
    if sampling.repetition_penalty is not None:
        extra["repeat_penalty"] = sampling.repetition_penalty
    if extra:
        mk["extra_body"] = extra
    return mk


def _build_env_config(step_timeout: int) -> dict:
    """The Docker ``environment`` config consumed by ``get_sb_environment``.

    Reproduces mini-swe-agent's builtin ``benchmarks/swebench.yaml`` environment
    block (``cwd: /testbed``, ``interpreter: ["bash", "-c"]``, ``PAGER``) plus
    our ``timeout`` and a ``PYTHONWARNINGS=ignore`` (so warnings don't appear
    before mini-swe-agent's submission sentinel on line 1). **``cwd: /testbed``
    is load-bearing**: the grader (`swebench_grade.grade_instance`) and the
    mini-swe agent both call ``env.execute`` without an explicit cwd and rely on
    this default — drop it and ``git apply``/``git reset`` run outside the repo
    (in ``/``) and silently fail. The image is derived from the instance by
    ``get_sb_environment``.
    """
    return {
        "environment": {
            "cwd": "/testbed",
            "timeout": step_timeout,
            "interpreter": ["bash", "-c"],
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "PYTHONWARNINGS": "ignore",
            },
        },
    }


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _token_usage(messages: list[dict]) -> dict[str, int]:
    """Sum prompt/completion tokens across all model responses in a trajectory."""
    prompt = completion = total = 0
    for msg in messages:
        usage = (msg.get("extra") or {}).get("response", {})
        usage = usage.get("usage") if isinstance(usage, dict) else None
        if not isinstance(usage, dict):
            continue
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        prompt += p
        completion += c
        total += int(usage.get("total_tokens") or (p + c))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


# ---------------------------------------------------------------------------
# Single-instance run
# ---------------------------------------------------------------------------


def run_instance(
    instance: dict,
    *,
    backend: Any,
    conn: dict[str, Any],
    trajectory_dir: Path,
    progress: bool = False,
) -> dict:
    """Run one instance through ``backend``, grade it, return a metrics record.

    Container creation and grading are backend-agnostic; the agent loop is
    delegated to ``backend.run(ctx)``. ``conn`` carries the model-connection
    fields of :class:`~quant_tuner.eval.agents.base.AgentRunContext`
    (``base_url``, ``served_model``, ``sampling``, limits, ``api_key``,
    ``model_class``).
    """
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    instance_id = instance.get("instance_id", "unknown")
    traj_path = trajectory_dir / f"{instance_id}.traj.json"

    record: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": instance.get("repo", ""),
        "agent": getattr(backend, "name", "?"),
        "exit_status": None,
        "patch_produced": False,
        "patch_chars": 0,
        "tools_used": 0,
        "tool_errors": 0,
        "n_model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "wall_sec": 0.0,
        "resolved": False,
        "grade_error": None,
        "error": None,
    }

    t0 = time.time()
    env = None
    try:
        if progress:
            print(f"  [{instance_id}] starting environment…", flush=True)
        env = get_sb_environment(_build_env_config(conn["step_timeout"]), instance)
        ctx = AgentRunContext(
            instance=instance,
            env=env,
            trajectory_path=traj_path,
            progress=progress,
            **conn,
        )
        result = backend.run(ctx)
        submission = result.submission
        record["exit_status"] = result.exit_status
        record["patch_produced"] = bool(submission.strip())
        record["patch_chars"] = len(submission)
        record["tools_used"] = result.tools_used
        record["tool_errors"] = result.tool_errors
        record["n_model_calls"] = result.n_model_calls
        record["prompt_tokens"] = result.prompt_tokens
        record["completion_tokens"] = result.completion_tokens
        record["total_tokens"] = result.total_tokens
        record["n_empty_patch_nudges"] = result.extra.get("n_empty_patch_nudges", 0)

        if progress:
            print(
                f"  [{instance_id}] exit={record['exit_status']} "
                f"patch={'yes' if record['patch_produced'] else 'no'} "
                f"calls={record['n_model_calls']} — grading…",
                flush=True,
            )
        # Grade by reusing the same container: the grader resets to base_commit,
        # re-applies the submission patch + gold test_patch, then runs the tests.
        grade = grade_instance(instance, submission, env=env)
        record["resolved"] = bool(grade.get("resolved"))
        record["grade_error"] = grade.get("error")
        record["n_fail_to_pass"] = grade.get("n_fail_to_pass", 0)
        record["n_fail_to_pass_passed"] = grade.get("n_fail_to_pass_passed", 0)
        record["n_pass_to_pass"] = grade.get("n_pass_to_pass", 0)
        record["n_pass_to_pass_passed"] = grade.get("n_pass_to_pass_passed", 0)

        # Snapshot wall time before writing (finally updates it too, but the
        # file would be written with 0.0 otherwise since finally runs after).
        record["wall_sec"] = time.time() - t0
        # Persist a human-readable per-instance result alongside the trajectory.
        result_path = trajectory_dir / f"{instance_id}.result.json"
        result_path.write_text(
            json.dumps(
                {**record, "submission": submission, "grade": grade},
                indent=2,
            )
        )
    except Exception as e:  # keep the suite going on a single bad instance
        record["error"] = f"{type(e).__name__}: {e}"
        if progress:
            print(f"  [{instance_id}] ERROR: {record['error']}", flush=True)
    finally:
        record["wall_sec"] = time.time() - t0
        if env is not None:
            cleanup = getattr(env, "cleanup", None)
            if cleanup is not None:
                with contextlib.suppress(Exception):
                    cleanup()
    return record


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _aggregate(model_label: str, records: list[dict]) -> SweSummary:
    n = len(records)
    n_resolved = sum(1 for r in records if r.get("resolved"))
    n_patched = sum(1 for r in records if r.get("patch_produced"))
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in records)
    total_tools = sum(int(r.get("tools_used") or 0) for r in records)
    total_errors = sum(int(r.get("tool_errors") or 0) for r in records)
    total_wall = sum(float(r.get("wall_sec") or 0.0) for r in records)
    denom = n or 1
    return SweSummary(
        model=model_label,
        n_instances=n,
        pass_rate=n_resolved / denom,
        patch_rate=n_patched / denom,
        mean_tokens=total_tokens / denom,
        total_tokens=float(total_tokens),
        mean_steps=total_tools / denom,
        tool_error_rate=(total_errors / total_tools) if total_tools else 0.0,
        mean_wall_sec=total_wall / denom,
        n_resolved=n_resolved,
        n_patched=n_patched,
        per_instance=records,
    )


def run_swebench_eval(
    holdout: Path | Iterable[dict],
    *,
    model_path: Path | None = None,
    base_url: str | None = None,
    sampling: Sampling | None = None,
    model_label: str | None = None,
    served_model: str = "local",
    trajectory_dir: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    instance_timeout: int = DEFAULT_INSTANCE_TIMEOUT,
    step_timeout: int = DEFAULT_STEP_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    agent: str = "mini-swe",
    model_class: str | None = None,
    ctx: int = 32768,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 180.0,
    chat_template_kwargs: str | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int | None = None,
    api_key: str = "sk-no-key",
    progress: bool = False,
    resume: bool = False,
) -> SweSummary:
    """Run the agentic SWE-rebench benchmark over ``holdout``.

    Provide either ``model_path`` (spawn a llama-server) **or** ``base_url``
    (reuse a running one) — mutually exclusive, mirroring the other evals.
    ``agent`` selects the scaffold driving the model (``mini-swe`` |
    ``openai-agents``; see :func:`quant_tuner.eval.agents.get_backend`).
    ``model_class`` is forwarded only to the ``mini-swe`` backend.
    ``trajectory_dir`` receives ``<instance_id>.traj.json`` (full conversation)
    and ``<instance_id>.result.json`` (patch + grade + metrics) for every
    instance.
    """
    if (model_path is None) == (base_url is None):
        raise ValueError("Pass exactly one of model_path= or base_url=")

    backend = get_backend(agent)

    # Quiet mini-swe-agent's startup banner / global-config chatter.
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

    sampling = sampling or Sampling(temperature=DEFAULT_TEMPERATURE)
    instances = list(holdout) if not isinstance(holdout, (str, Path)) else load_holdout(Path(holdout))
    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    label = model_label or (Path(model_path).name if model_path else served_model)

    server_cm = (
        running_server(
            Path(model_path),
            ctx=ctx,
            ngl=ngl,
            log_path=server_log_path,
            startup_timeout=server_startup_timeout,
            chat_template_kwargs=chat_template_kwargs,
            spec_type=spec_type,
            spec_draft_n_max=spec_draft_n_max,
        )
        if model_path is not None
        else nullcontext(base_url)
    )

    records: list[dict] = []
    with server_cm as url:
        assert url is not None  # exactly one of model_path/base_url guaranteed above
        conn: dict[str, Any] = {
            "base_url": url,
            "served_model": served_model,
            "sampling": sampling,
            "max_steps": max_steps,
            "instance_timeout": instance_timeout,
            "step_timeout": step_timeout,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "model_class": model_class,
        }
        for i, instance in enumerate(instances, 1):
            iid = instance.get("instance_id", "?")
            if progress:
                print(f"[{i}/{len(instances)}] {iid}", flush=True)
            # Resume: skip an instance whose result.json already exists and parses
            # (long generation runs shouldn't re-run completed work after a crash).
            if resume:
                done = trajectory_dir / f"{iid}.result.json"
                if done.exists():
                    try:
                        prev = json.loads(done.read_text())
                        records.append({k: v for k, v in prev.items()
                                        if k not in ("submission", "grade")})
                        if progress:
                            print(f"  [{iid}] resume: skip (already graded, "
                                  f"resolved={prev.get('resolved')})", flush=True)
                        continue
                    except Exception:
                        pass  # corrupt/partial result -> re-run it
            records.append(
                run_instance(
                    instance,
                    backend=backend,
                    conn=conn,
                    trajectory_dir=trajectory_dir,
                    progress=progress,
                )
            )

    return _aggregate(label, records)


def swebench_rep(holdout: Path, *, trajectory_dir: Path, **eval_kwargs):
    """Build an ``eval_fn(base_url, sampling, rep) -> dict[str, float]`` for ``reps``.

    Each rep writes trajectories under ``trajectory_dir/rep_<idx>/``. Note agentic
    runs are expensive — ``reps=1`` is the norm.
    """

    def _fn(base_url: str, sampling: Sampling, rep_idx: int) -> dict[str, float]:
        summary = run_swebench_eval(
            holdout,
            base_url=base_url,
            sampling=sampling,
            trajectory_dir=Path(trajectory_dir) / f"rep_{rep_idx}",
            **eval_kwargs,
        )
        return summary.scalar_metrics()

    return _fn

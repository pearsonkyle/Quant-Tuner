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
import copy
import json
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_tuner.eval.server import running_server
from quant_tuner.eval.swebench_grade import grade_instance
from quant_tuner.eval.toolcall import Sampling

DEFAULT_MAX_STEPS = 100
DEFAULT_INSTANCE_TIMEOUT = 7200  # wall-clock seconds per instance (2 h for 100-step runs)
DEFAULT_STEP_TIMEOUT = 120  # seconds for a single bash command in the container
DEFAULT_MAX_TOKENS = 8096  # per model call (agent reasoning + tool call)


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


def _build_base_config(
    *,
    base_url: str,
    served_model: str,
    sampling: Sampling,
    max_steps: int,
    instance_timeout: int,
    step_timeout: int,
    max_tokens: int,
    model_class: str | None,
    api_key: str,
) -> dict:
    """Start from mini-swe-agent's builtin ``swebench.yaml`` and point it at our server."""
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    base = get_config_from_spec(builtin_config_dir / "benchmarks" / "swebench.yaml")

    # litellm talks to a local OpenAI-compatible endpoint via the ``openai/`` prefix
    # + api_base/api_key in model_kwargs. drop_params lets it shed params the
    # server doesn't support; cost_tracking=ignore_errors stops it raising when
    # litellm can't price a local model (cost is always 0 here).
    model_kwargs = _sampling_to_model_kwargs(sampling, max_tokens=max_tokens)
    model_kwargs["api_base"] = base_url
    model_kwargs["api_key"] = api_key

    overrides: dict = {
        "agent": {
            "step_limit": max_steps,
            "wall_time_limit_seconds": instance_timeout,
            "cost_limit": 0.0,  # local model: cost is 0, disable the cost cap
        },
        "model": {
            "model_name": f"openai/{served_model}",
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        },
        "environment": {
            "timeout": step_timeout,
            # Suppress Python warnings so they don't appear before the
            # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT sentinel on line 1.
            # mini-swe-agent checks lines[0].strip() == sentinel exactly —
            # a conda RequestsDependencyWarning prefix causes 65+ missed submissions.
            "env": {"PYTHONWARNINGS": "ignore"},
        },
    }
    if model_class:
        overrides["model"]["model_class"] = model_class
    return recursive_merge(base, overrides)


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


def _make_metrics_agent_class() -> type:
    """Build a DefaultAgent subclass that tallies bash calls and non-zero exits.

    Defined lazily so the module imports without mini-swe-agent installed.
    """
    from minisweagent.agents.default import DefaultAgent

    class _MetricsAgent(DefaultAgent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.tools_used = 0
            self.tool_errors = 0

        def execute_actions(self, message: dict) -> list[dict]:
            actions = (message.get("extra") or {}).get("actions", [])
            outputs = [self.env.execute(action) for action in actions]
            self.tools_used += len(actions)
            self.tool_errors += sum(1 for o in outputs if o.get("returncode", 0) != 0)
            return self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )

    return _MetricsAgent


# ---------------------------------------------------------------------------
# Single-instance run
# ---------------------------------------------------------------------------


def run_instance(
    instance: dict,
    config: dict,
    *,
    trajectory_dir: Path,
    progress: bool = False,
) -> dict:
    """Run the agent on one instance, grade it, and return a metrics record."""
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    instance_id = instance.get("instance_id", "unknown")
    inst_config = copy.deepcopy(config)
    traj_path = trajectory_dir / f"{instance_id}.traj.json"

    record: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": instance.get("repo", ""),
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
    agent = None
    try:
        if progress:
            print(f"  [{instance_id}] starting environment…", flush=True)
        model = get_model(config=inst_config.get("model", {}))
        env = get_sb_environment(inst_config, instance)
        agent_cls = _make_metrics_agent_class()
        agent = agent_cls(model, env, output_path=traj_path, **inst_config.get("agent", {}))

        info = agent.run(instance["problem_statement"])
        submission = info.get("submission", "") or ""
        record["exit_status"] = info.get("exit_status")
        record["patch_produced"] = bool(submission.strip())
        record["patch_chars"] = len(submission)
        record["tools_used"] = agent.tools_used
        record["tool_errors"] = agent.tool_errors
        record["n_model_calls"] = agent.n_calls
        record.update(_token_usage(agent.messages))

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
        if agent is not None:
            with contextlib.suppress(Exception):
                agent.save(traj_path)
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
    model_class: str | None = None,
    ctx: int = 32768,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 180.0,
    chat_template_kwargs: str | None = None,
    api_key: str = "sk-no-key",
    progress: bool = False,
) -> SweSummary:
    """Run the agentic SWE-rebench benchmark over ``holdout``.

    Provide either ``model_path`` (spawn a llama-server) **or** ``base_url``
    (reuse a running one) — mutually exclusive, mirroring the other evals.
    ``trajectory_dir`` receives ``<instance_id>.traj.json`` (full conversation)
    and ``<instance_id>.result.json`` (patch + grade + metrics) for every
    instance.
    """
    if (model_path is None) == (base_url is None):
        raise ValueError("Pass exactly one of model_path= or base_url=")

    # Quiet mini-swe-agent's startup banner / global-config chatter.
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

    sampling = sampling or Sampling(temperature=0.0)
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
        )
        if model_path is not None
        else nullcontext(base_url)
    )

    records: list[dict] = []
    with server_cm as url:
        assert url is not None  # exactly one of model_path/base_url guaranteed above
        config = _build_base_config(
            base_url=url,
            served_model=served_model,
            sampling=sampling,
            max_steps=max_steps,
            instance_timeout=instance_timeout,
            step_timeout=step_timeout,
            max_tokens=max_tokens,
            model_class=model_class,
            api_key=api_key,
        )
        for i, instance in enumerate(instances, 1):
            if progress:
                iid = instance.get("instance_id", "?")
                print(f"[{i}/{len(instances)}] {iid}", flush=True)
            records.append(
                run_instance(
                    instance, config, trajectory_dir=trajectory_dir, progress=progress
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

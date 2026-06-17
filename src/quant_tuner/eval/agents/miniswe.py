"""mini-swe-agent backend — the default scaffold.

A faithful move of the original inline mini-swe-agent logic behind the
:class:`~quant_tuner.eval.agents.base.AgentBackend` contract. All
``minisweagent`` imports are lazy so this module loads (and ``get_backend``
resolves) without the ``swebench`` extra installed.
"""

from __future__ import annotations

import contextlib
from typing import Any

from quant_tuner.eval.agents.base import AgentRunContext, AgentRunResult
from quant_tuner.eval.swebench import _sampling_to_model_kwargs, _token_usage
from quant_tuner.eval.toolcall import Sampling


def _build_model_config(
    *,
    base_url: str,
    served_model: str,
    sampling: Sampling,
    max_steps: int,
    instance_timeout: int,
    max_tokens: int,
    model_class: str | None,
    api_key: str,
) -> dict:
    """Start from mini-swe-agent's builtin ``swebench.yaml`` and point it at our server.

    Returns the merged config; only its ``model`` and ``agent`` sections are
    used here — the Docker ``environment`` is created by the caller.
    """
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    base = get_config_from_spec(builtin_config_dir / "benchmarks" / "swebench.yaml")

    # litellm talks to a local OpenAI-compatible endpoint via the ``openai/`` prefix
    # + api_base/api_key in model_kwargs. cost_tracking=ignore_errors stops it
    # raising when litellm can't price a local model (cost is always 0 here).
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
    }
    if model_class:
        overrides["model"]["model_class"] = model_class
    return recursive_merge(base, overrides)


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


class MiniSweBackend:
    """Drives mini-swe-agent's ``DefaultAgent`` over one instance."""

    name = "mini-swe"

    def run(self, ctx: AgentRunContext) -> AgentRunResult:
        from minisweagent.models import get_model

        config = _build_model_config(
            base_url=ctx.base_url,
            served_model=ctx.served_model,
            sampling=ctx.sampling,
            max_steps=ctx.max_steps,
            instance_timeout=ctx.instance_timeout,
            max_tokens=ctx.max_tokens,
            model_class=ctx.model_class,
            api_key=ctx.api_key,
        )
        model = get_model(config=config.get("model", {}))
        agent_cls = _make_metrics_agent_class()
        agent = agent_cls(
            model, ctx.env, output_path=ctx.trajectory_path, **config.get("agent", {})
        )
        try:
            info = agent.run(ctx.instance["problem_statement"])
        finally:
            # DefaultAgent auto-saves via output_path, but persist a final snapshot
            # so a partial trajectory survives an exception too.
            with contextlib.suppress(Exception):
                agent.save(ctx.trajectory_path)

        submission = info.get("submission", "") or ""
        usage = _token_usage(agent.messages)
        return AgentRunResult(
            submission=submission,
            messages=agent.messages,
            n_model_calls=agent.n_calls,
            tools_used=agent.tools_used,
            tool_errors=agent.tool_errors,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            exit_status=info.get("exit_status"),
        )

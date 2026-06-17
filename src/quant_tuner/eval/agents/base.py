"""The pluggable-agent contract for the agentic SWE-rebench eval.

A *backend* drives one agent scaffold (mini-swe-agent, the OpenAI Agents SDK,
Qwen Code, …) over a single SWE-rebench instance. Everything around it —
per-instance Docker container creation, grading, aggregation, CSV/JSON output,
reps — lives in :mod:`quant_tuner.eval.swebench` and is backend-agnostic. A
backend's one job: given a live Docker ``env`` and a model endpoint, run the
agent loop and return a normalized :class:`AgentRunResult` (a git-diff
``submission`` + a trajectory + tool/token counts).

The backend also owns persisting its own conversation trajectory to
``ctx.trajectory_path`` (each scaffold serializes differently); the caller
writes the per-instance ``*.result.json`` from the returned metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from quant_tuner.eval.toolcall import Sampling


@dataclass
class AgentRunContext:
    """Everything a backend needs to run one instance.

    ``env`` is a live mini-swe-agent ``DockerEnvironment`` (created by the
    caller and cleaned up by the caller); a backend only calls
    ``env.execute({"command": ...}, cwd=..., timeout=...)`` on it.
    """

    instance: dict
    env: Any
    base_url: str
    served_model: str
    sampling: Sampling
    max_steps: int
    instance_timeout: int
    step_timeout: int
    max_tokens: int
    api_key: str
    trajectory_path: Path
    model_class: str | None = None  # mini-swe-agent only; ignored by other backends
    progress: bool = False


@dataclass
class AgentRunResult:
    """Normalized outcome of one agent run, consumed by ``swebench.run_instance``."""

    submission: str  # git diff to grade (may be "")
    messages: list[dict]  # trajectory (already persisted to ctx.trajectory_path by the backend)
    n_model_calls: int = 0
    tools_used: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    exit_status: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class AgentBackend(Protocol):
    """One agent scaffold. Implementations live in sibling modules."""

    name: str

    def run(self, ctx: AgentRunContext) -> AgentRunResult: ...

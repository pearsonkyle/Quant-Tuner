"""Pluggable agent backends for the agentic SWE-rebench eval.

``get_backend(name)`` resolves a scaffold by name. Backend modules import their
heavyweight SDKs lazily (inside ``run``), so resolving a backend never requires
the SDK to be installed — only *running* it does. This mirrors the lazy-import
discipline in :mod:`quant_tuner.eval.swebench`.
"""

from __future__ import annotations

from quant_tuner.eval.agents.base import (
    AgentBackend,
    AgentRunContext,
    AgentRunResult,
)

__all__ = [
    "AgentBackend",
    "AgentRunContext",
    "AgentRunResult",
    "get_backend",
    "available_backends",
]

# name -> "module:attr" of an AgentBackend implementation, imported lazily.
_REGISTRY: dict[str, str] = {
    "mini-swe": "quant_tuner.eval.agents.miniswe:MiniSweBackend",
    "openai-agents": "quant_tuner.eval.agents.openai_agents:OpenAIAgentsBackend",
}


def available_backends() -> list[str]:
    """Names accepted by :func:`get_backend` (and the ``--agent`` CLI flag)."""
    return sorted(_REGISTRY)


def get_backend(name: str) -> AgentBackend:
    """Instantiate the agent backend registered under ``name``."""
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown agent backend {name!r}; choose one of {available_backends()}"
        ) from None
    import importlib

    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, attr)
    return backend_cls()

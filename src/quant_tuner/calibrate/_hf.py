"""Shared helpers for HF-model calibration forward passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Candidate dotted paths (relative to the top-level HF model) for the decoder
# layer stack. DiffusionGemma's causal *encoder* stack is listed rather than
# its diffusion decoder: calibration forwards run in encoder (causal) mode,
# and the two stacks share every backbone weight (tied parameters), so weight
# mutations applied through encoder paths land in the decoder too.
_TRUNK_CANDIDATES = (
    "model",
    "model.encoder.language_model",
    "encoder.language_model",
)


def _get_attr_path(obj: Any, dotted: str) -> Any | None:
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def resolve_text_trunk(model) -> tuple[str, Any]:
    """Locate the text decoder trunk (the module owning ``.layers``).

    Returns ``(dotted_path, module)`` where ``dotted_path`` is relative to the
    top-level model (e.g. ``"model"`` for standard CausalLMs,
    ``"model.encoder.language_model"`` for DiffusionGemma). Raises if no
    candidate matches.
    """
    for path in _TRUNK_CANDIDATES:
        trunk = _get_attr_path(model, path)
        if trunk is not None and hasattr(trunk, "layers"):
            return path, trunk
    raise RuntimeError(
        "no text trunk with .layers found (tried: "
        + ", ".join(f"model.{p}" for p in _TRUNK_CANDIDATES)
        + "); unsupported architecture"
    )


def load_calibration_model(model_dir: Path, *, torch_dtype, device: str):
    """Load an HF checkpoint for calibration, handling non-CausalLM classes.

    ``AutoModelForCausalLM`` covers the standard architectures; checkpoints
    whose class isn't registered there (e.g. ``DiffusionGemmaForBlockDiffusion``)
    are loaded via their concrete ``architectures[0]`` class from
    ``transformers`` instead.
    """
    import transformers
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype, trust_remote_code=True
        )
    except (ValueError, KeyError):
        config = json.loads((Path(model_dir) / "config.json").read_text())
        archs = config.get("architectures") or []
        cls = getattr(transformers, archs[0], None) if archs else None
        if cls is None:
            raise
        model = cls.from_pretrained(
            model_dir, torch_dtype=torch_dtype, trust_remote_code=True
        )
    return model.to(device)


def forward_no_logits(model, input_ids) -> None:
    """Run a forward pass for hook-based stat collection, skipping the LM head.

    Calibration passes only need activations at the hooked Linears — the
    logits are discarded, yet ``model(ids)`` materializes a
    ``[1, ctx, vocab]`` float tensor (several GB at ctx 4096 on a 150k-vocab
    model) and pays the lm_head matmul for every chunk. Calling the decoder
    trunk (``model.model``, or DiffusionGemma's causal encoder stack) directly
    avoids both. Falls back to the full model for architectures without a
    recognizable trunk.
    """
    try:
        _, trunk = resolve_text_trunk(model)
    except RuntimeError:
        model(input_ids)
        return
    trunk(input_ids=input_ids)


def causal_logits(model, input_ids):
    """Causal-LM logits for sanity checks, on architectures where ``model(ids)``
    isn't a causal forward (DiffusionGemma's top-level forward wants canvas
    inputs; its *encoder* + tied ``lm_head`` is the causal path)."""
    trunk_path, trunk = resolve_text_trunk(model)
    if trunk_path == "model":
        return model(input_ids).logits
    hidden = trunk(input_ids=input_ids).last_hidden_state
    head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError(f"no lm_head on model with trunk {trunk_path!r}")
    return head(hidden)

"""Torch device resolution shared by the calibrators."""

from __future__ import annotations


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to the best available backend (cuda > mps > cpu).

    Any explicit device string passes through untouched, so recipes can still
    pin ``device: cpu`` (or a specific ``cuda:N``).
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

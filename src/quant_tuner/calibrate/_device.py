"""Torch device resolution shared by the calibrators."""

from __future__ import annotations

import gc
import sys


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


def release_gpu_memory(tag: str = "") -> None:
    """Return cached-but-free GPU memory to the driver.

    **Call this before handing off to a llama.cpp subprocess.** Torch's caching
    allocator does not release freed blocks back to the driver — it keeps them
    reserved for reuse. That is invisible within one process, but a llama.cpp
    binary is a *separate* process: it sees only what the driver reports free,
    so whatever torch is still holding is memory llama.cpp cannot offload into.

    Measured cost of not doing this (exp-060 AWQ, 2026-08-15): after
    ``awq.apply`` the Python process kept 69.7 GB of a 96 GB card reserved, so
    ``llama-imatrix`` got 26 GB, fell back to CPU for most layers, and ran at
    246 s/chunk against the 109 s/chunk it achieves with the card free — 2.26x
    slower, turning a ~4 h stage into ~9 h.

    Safe to call unconditionally: no-ops without torch or on CPU-only boxes.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        freed = before - torch.cuda.memory_reserved()
        if freed > 0:
            label = f"[{tag}] " if tag else ""
            print(
                f"{label}released {freed / 2**30:.1f} GiB of cached GPU memory "
                f"back to the driver",
                file=sys.stderr,
            )
    mps = getattr(torch, "mps", None)
    if (
        mps is not None
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        mps.empty_cache()

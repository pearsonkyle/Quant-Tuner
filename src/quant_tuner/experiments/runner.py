"""Timing and idempotency helpers for experiment driver scripts."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar


def log(msg: str) -> None:
    """Print a timestamped line, flushed (so it interleaves nicely with subprocess output)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def phase(name: str):
    """Bracket a block with a ``┌─ name`` / ``└─ name done (M.m min)`` log frame."""
    t0 = time.time()
    log(f"┌─ {name}")
    try:
        yield
    finally:
        log(f"└─ {name} done ({(time.time() - t0) / 60:.1f} min)")


T = TypeVar("T")


def step(
    name: str,
    outputs: Path | Iterable[Path],
    build: Callable[[], T],
) -> T | None:
    """Run ``build()`` inside a ``phase(name)`` only if any ``outputs`` are missing.

    ``outputs`` is one or more files the step is expected to produce. If all of
    them already exist on disk, ``build`` is skipped and ``None`` is returned;
    otherwise ``build`` runs inside a timed phase and its return value is
    forwarded. Useful for crash-resume in multi-hour pipelines.
    """
    out_list = [outputs] if isinstance(outputs, Path) else list(outputs)
    if all(p.exists() for p in out_list):
        names = ", ".join(p.name for p in out_list) or name
        log(f"✓ {names} already present")
        return None
    with phase(name):
        return build()

"""GGUF parameter count and bits-per-weight (BPW) computation."""

from __future__ import annotations

import sys
from pathlib import Path

from quant_tuner.paths import LLAMA_CPP_DIR


def _ensure_gguf_py() -> None:
    p = str(LLAMA_CPP_DIR / "gguf-py")
    if p not in sys.path:
        sys.path.insert(0, p)


def n_params(gguf_path: Path | str) -> int:
    """Total parameter count summed across all tensors in a GGUF file."""
    _ensure_gguf_py()
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    total = 0
    for t in reader.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        total += n
    return total


def bpw(gguf_path: Path | str, n: int) -> float:
    """Effective bits per weight: file_size_bits / n_params."""
    size_bytes = Path(gguf_path).stat().st_size
    return size_bytes * 8.0 / n


def size_gib(gguf_path: Path | str) -> float:
    return Path(gguf_path).stat().st_size / (1024**3)

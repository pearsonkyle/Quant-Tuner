"""Gating for lens integration tests that need a built server + a tiny GGUF.

These are skipped unless BOTH are set:
    QT_LENS_IT=1
    QT_TINY_GGUF=/path/to/a/small.gguf   (something llama.cpp can load, e.g. Qwen3-0.6B-Q8_0)

They also require ``native/jlens_server/jlens-server`` to be built
(``bash native/jlens_server/build.sh``); the build itself is exercised by
``test_jlens_server.py::test_build``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _enabled() -> bool:
    return os.environ.get("QT_LENS_IT") == "1"


@pytest.fixture(scope="session")
def tiny_gguf() -> Path:
    if not _enabled():
        pytest.skip("set QT_LENS_IT=1 to run lens integration tests")
    path = os.environ.get("QT_TINY_GGUF")
    if not path or not Path(path).exists():
        pytest.skip("set QT_TINY_GGUF to a small GGUF path")
    return Path(path)


@pytest.fixture(scope="session")
def jlens_server_built() -> Path:
    from quant_tuner.paths import jlens_server_bin

    try:
        return jlens_server_bin()
    except FileNotFoundError:
        pytest.skip("jlens-server not built (run native/jlens_server/build.sh)")

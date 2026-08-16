"""Tests for `calibrate._device.release_gpu_memory`.

Why this exists: torch's caching allocator does not hand freed blocks back to
the driver, and every llama.cpp stage is a SEPARATE PROCESS that can only
offload into what the driver reports free. On exp-060 that cost 2.26x on the
imatrix stage (246 s/chunk vs 109 s/chunk) because `awq.apply` left 69.7 GB of
a 96 GB card reserved. These tests pin the contract; they run on CPU-only boxes.
"""

from __future__ import annotations

import sys
import types

import pytest

from quant_tuner.calibrate._device import release_gpu_memory


def test_release_is_a_noop_without_torch(monkeypatch):
    """Must not explode in an env where torch is not installed."""
    monkeypatch.setitem(sys.modules, "torch", None)
    # `import torch` with a None entry raises ImportError — the path we guard.
    release_gpu_memory()


def test_release_is_a_noop_on_cpu_only(monkeypatch):
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=None),
        mps=None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    release_gpu_memory()  # no attribute access beyond is_available()


def test_release_calls_empty_cache_when_cuda_is_present(monkeypatch):
    calls: list[str] = []
    reserved = iter([80 * 2**30, 2 * 2**30])  # before, after

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            memory_reserved=lambda: next(reserved),
            empty_cache=lambda: calls.append("cuda.empty_cache"),
        ),
        backends=types.SimpleNamespace(mps=None),
        mps=None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    release_gpu_memory("unit-test")

    assert calls == ["cuda.empty_cache"], (
        "empty_cache() is the whole point — without it the freed blocks stay "
        "reserved and the next llama.cpp subprocess cannot use them"
    )


def test_release_reports_the_amount_freed(monkeypatch, capsys):
    reserved = iter([70 * 2**30, 0])
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            memory_reserved=lambda: next(reserved),
            empty_cache=lambda: None,
        ),
        backends=types.SimpleNamespace(mps=None),
        mps=None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    release_gpu_memory("awq.apply")

    err = capsys.readouterr().err
    assert "70.0 GiB" in err and "awq.apply" in err, (
        "the log line is the only way a future run's slowdown gets diagnosed"
    )


@pytest.mark.parametrize("fn", ["apply", "calibrate"])
def test_awq_releases_before_returning(fn):
    """The AWQ stages must release; the pipeline hands straight to llama-imatrix."""
    import inspect

    from quant_tuner.calibrate import awq

    src = inspect.getsource(getattr(awq, fn))
    assert "release_gpu_memory" in src, (
        f"awq.{fn} must release cached GPU memory before returning — the next "
        f"stage is a llama.cpp subprocess that cannot use reserved blocks"
    )


def test_gptq_apply_releases_before_returning():
    import inspect

    from quant_tuner.calibrate import gptq

    assert "release_gpu_memory" in inspect.getsource(gptq.apply)


def test_pipeline_releases_before_every_imatrix_call():
    """Defense in depth: the pipeline releases even if a calibrator regresses."""
    import inspect

    from quant_tuner import pipeline

    src = inspect.getsource(pipeline)
    n_imatrix = src.count("llama_cpp.imatrix(")
    n_release = src.count("release_gpu_memory(")
    assert n_release >= n_imatrix, (
        f"{n_imatrix} llama_cpp.imatrix call site(s) but only {n_release} "
        f"release_gpu_memory() call(s) — each spawn needs a release first"
    )

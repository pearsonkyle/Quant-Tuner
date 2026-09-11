"""Shared pytest fixtures for the unit suite.

`requires_cuda` skips a test when no CUDA device is available. The tests it
gates exercise the CUDA-only SDPA dispatch in `qat/attention.py`
(`use_gqa_in_sdpa`, the `enable_gqa` FlashAttention path) — that function and
kernel path simply do not exist on CPU/MPS boxes, so the tests are meaningless
there and would fail on a transformers/torch version where the SDPA internals
differ. On a CUDA box they run normally.
"""
from __future__ import annotations

import pytest
import torch

_CUDA = torch.cuda.is_available()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_cuda: test exercises the CUDA-only SDPA dispatch "
        "(qat.attention use_gqa_in_sdpa / enable_gqa); skipped when CUDA is unavailable",
    )


@pytest.fixture
def requires_cuda():
    """Skip the calling test when no CUDA device is present."""
    if not _CUDA:
        pytest.skip("CUDA device required (tests the CUDA-only SDPA dispatch)")

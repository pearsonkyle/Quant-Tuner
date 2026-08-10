"""Query-chunked SDPA must be BIT-identical to the stock kernel.

This is the property the long-window training run rests on: if chunking changed the
numerics at all, every result produced above an 8191-token window would be
incomparable with everything produced below it. CPU here — the MPSGraph INT_MAX
limit that motivates chunking is a device constraint, the math is not.
"""

from __future__ import annotations

import pytest
import torch

from quant_tuner.qat.attention import (
    chunked_causal_sdpa,
    disable_chunked_sdpa,
    enable_chunked_sdpa,
    safe_chunk,
)


class FakeAttn(torch.nn.Module):
    is_causal = True
    num_key_value_groups = 4


@pytest.fixture
def patched():
    enable_chunked_sdpa()
    yield
    disable_chunked_sdpa()


def _qkv(seq, heads=8, kv=2, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, heads, seq, dim, generator=g),
            torch.randn(1, kv, seq, dim, generator=g),
            torch.randn(1, kv, seq, dim, generator=g))


@pytest.mark.parametrize("seq", [64, 256, 1000])
@pytest.mark.parametrize("chunk", [16, 64, 4096])
def test_chunked_matches_is_causal_bit_exactly(seq, chunk):
    q, k, v = _qkv(seq)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                           enable_gqa=True)
    got = chunked_causal_sdpa(q, k, v, chunk_hint=chunk, enable_gqa=True)
    assert torch.equal(ref, got)


def test_chunk_size_larger_than_seq_is_a_single_block():
    q, k, v = _qkv(128)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                           enable_gqa=True)
    assert torch.equal(ref, chunked_causal_sdpa(q, k, v, chunk_hint=10_000, enable_gqa=True))


def test_explicit_mask_is_honored_and_kv_is_not_sliced():
    """A caller-supplied mask may be non-causal, so K/V must stay whole under it."""
    q, k, v = _qkv(64, heads=4, kv=4)
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool).bernoulli_(0.7, generator=torch.Generator().manual_seed(3))
    mask[..., 0] = True  # every row must attend to something
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    got = chunked_causal_sdpa(q, k, v, attention_mask=mask, chunk_hint=16)
    assert torch.equal(ref, got)


def test_safe_chunk_keeps_the_score_tensor_under_int_max():
    for heads, kv_len in ((32, 16384), (32, 65536), (8, 8192)):
        c = safe_chunk(heads, kv_len)
        assert heads * c * kv_len < 2**31
    # a huge context still yields a usable block, never 0
    assert safe_chunk(32, 1_000_000) >= 128


def test_patched_forward_matches_stock_forward(patched):
    """Through transformers' own dispatch, with GQA repeat_kv on the module."""
    from transformers.integrations import sdpa_attention as mod

    from quant_tuner.qat.attention import _original_sdpa

    q, k, v = _qkv(2048, heads=8, kv=2)
    ref, _ = _original_sdpa(FakeAttn(), q, k, v, None, scaling=0.125)
    got, _ = mod.sdpa_attention_forward(FakeAttn(), q, k, v, None, scaling=0.125)
    assert torch.equal(ref, got)


def test_short_windows_use_the_stock_kernel_untouched(patched):
    """Below the threshold nothing changes — decode steps keep the fused path."""
    from transformers.integrations import sdpa_attention as mod

    from quant_tuner.qat.attention import _original_sdpa

    q, k, v = _qkv(128, heads=8, kv=2)
    ref, _ = _original_sdpa(FakeAttn(), q, k, v, None, scaling=0.125)
    got, _ = mod.sdpa_attention_forward(FakeAttn(), q, k, v, None, scaling=0.125)
    assert torch.equal(ref, got)


def test_enable_is_idempotent_and_restores(patched):
    from transformers.integrations import sdpa_attention as mod
    patched_fn = mod.sdpa_attention_forward
    enable_chunked_sdpa()
    assert mod.sdpa_attention_forward is patched_fn
    disable_chunked_sdpa()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is mod.sdpa_attention_forward
    enable_chunked_sdpa()  # leave the fixture's teardown something to undo


def test_gradients_flow_through_every_block():
    q, k, v = _qkv(512, heads=4, kv=4)
    q.requires_grad_(True)
    chunked_causal_sdpa(q, k, v, chunk_hint=64).sum().backward()
    # the last query rows attend to the most keys; the first still get a gradient
    assert q.grad is not None
    assert (q.grad.abs().sum(dim=(1, 3))[0] > 0).all()

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
    assert safe_chunk(32, 1_000_000) >= 64


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


# --- cached prefix (kv_len > q_len) ----------------------------------------------------
#
# The prefix-context training path (train.prefix_kv) calls attention with a KV cache, so
# the queries are the LAST q_len positions of a longer sequence. Getting the alignment
# wrong is silent: shapes broadcast, the loss still falls, and the tail simply trains on a
# truncated context. These pin it against an explicitly-constructed mask.

def _cached_qkv(prefix, tail, heads=8, kv=2, dim=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    total = prefix + tail
    return (torch.randn(1, heads, tail, dim, generator=g),
            torch.randn(1, kv, total, dim, generator=g),
            torch.randn(1, kv, total, dim, generator=g))


@pytest.mark.parametrize(("prefix", "tail"), [(100, 28), (1024, 512), (3000, 1000)])
@pytest.mark.parametrize("chunk", [16, 128, 8192])
def test_cached_prefix_matches_an_explicit_offset_mask(prefix, tail, chunk):
    q, k, v = _cached_qkv(prefix, tail)
    # query row r is at absolute position prefix+r and may see keys 0..prefix+r
    mask = torch.ones(tail, prefix + tail, dtype=torch.bool).tril(diagonal=prefix)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                           enable_gqa=True)
    got = chunked_causal_sdpa(q, k, v, chunk_hint=chunk, enable_gqa=True)
    assert torch.equal(ref, got)


def test_cached_prefix_differs_from_the_top_left_is_causal_answer():
    """The bug this guards: torch's is_causal aligns top-LEFT, so with a cache every query
    would see only the prefix head and none of its own recent context."""
    q, k, v = _cached_qkv(512, 128)
    wrong = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                             enable_gqa=True)
    got = chunked_causal_sdpa(q, k, v, chunk_hint=64, enable_gqa=True)
    assert not torch.allclose(wrong, got)


def test_a_full_run_equals_prefix_then_cached_tail():
    """Splitting a sequence into (no-grad prefix, cached tail) must reproduce the tail rows
    of the unsplit causal call exactly — that equivalence is the whole premise."""
    total, tail = 1024, 256
    q, k, v = _qkv(total, heads=8, kv=2, seed=7)
    whole = chunked_causal_sdpa(q, k, v, chunk_hint=128, enable_gqa=True)
    split = chunked_causal_sdpa(q[:, :, total - tail:], k, v, chunk_hint=128,
                                enable_gqa=True)
    assert torch.equal(whole[:, :, total - tail:], split)


def test_kv_shorter_than_q_is_rejected():
    q, k, v = _qkv(64)
    with pytest.raises(ValueError, match="not a causal decoder call"):
        chunked_causal_sdpa(q, k[:, :, :32], v[:, :, :32], enable_gqa=True)


def test_short_cached_window_is_not_sent_to_the_stock_kernel(patched):
    """A 128-token tail is below chunk_above, but with a cache the stock maskless kernel is
    WRONG — the gate has to notice kv_len != q_len."""
    from transformers.integrations import sdpa_attention as mod
    q, k, v = _cached_qkv(2048, 128, heads=8, kv=2)
    got, _ = mod.sdpa_attention_forward(FakeAttn(), q, k, v, None, scaling=0.125)
    mask = torch.ones(128, 2048 + 128, dtype=torch.bool).tril(diagonal=2048)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, mod.repeat_kv(k, 4), mod.repeat_kv(v, 4), attn_mask=mask, scale=0.125)
    assert torch.equal(ref.transpose(1, 2).contiguous(), got)


def test_chunk_is_budgeted_by_bytes_not_just_element_count():
    """A fixed block is an element cap, not a memory cap: the score tensor grows with
    kv_len, so the 2048 block that costs ~2 GiB at kv 8064 costs 8 GiB at kv 32768. That
    is what drove a 32768 run 26 GB into swap."""
    budget = 2 * 1024**3
    for kv in (8064, 16128, 32768, 65536):
        c = safe_chunk(32, kv, itemsize=4, score_bytes=budget)
        assert 32 * c * kv * 4 <= budget, f"kv={kv} chunk={c} exceeds the byte budget"
        assert 32 * c * kv < 2**31
    # at the window the published runs used, the budget must not change the old behavior
    assert safe_chunk(32, 8064, itemsize=4, score_bytes=budget) == 2048


def test_byte_budget_does_not_change_the_numerics():
    q, k, v = _qkv(2048, heads=8, kv=2)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                           enable_gqa=True)
    tiny = chunked_causal_sdpa(q, k, v, score_bytes=1 << 20, enable_gqa=True)
    assert torch.equal(ref, tiny)


def test_cost_per_trained_token_falls_as_the_tail_grows():
    """Attention is ~S^2 per window whatever the prefix/tail split, so cost per TRAINED
    token goes as S^2/T. Minimizing the tail minimizes memory and maximizes cost — the
    probe grid must sweep the tail UPWARD. This pins the arithmetic that decides it."""
    S = 32768
    cost = {T: S * S / T for T in (4096, 8192, 16384, 24576)}
    assert cost[4096] > cost[8192] > cost[16384] > cost[24576]
    # ...and a full-gradient short window is still far cheaper per trained token
    assert cost[24576] > 8064 * 8064 / 8064


# --- score recomputation ----------------------------------------------------------------
#
# Chunking bounds ONE block's score tensor; it does not bound their sum. With grad enabled
# every block saves its own softmax output, so the saved total is the whole
# [heads, q_len, kv_len] matrix regardless of block size — 64 GiB at q 16384 / kv 32768,
# which OOMed the real model at 142 GiB allocated. Checkpointing each block trades one
# extra attention pass for that.

def test_recompute_scores_does_not_change_the_forward():
    q, k, v = _qkv(512, heads=4, kv=4)
    q.requires_grad_(True)
    a = chunked_causal_sdpa(q, k, v, chunk_hint=64, recompute_scores=False)
    b = chunked_causal_sdpa(q, k, v, chunk_hint=64, recompute_scores=True)
    assert torch.equal(a, b)


def test_recompute_scores_does_not_change_the_gradient():
    q, k, v = _qkv(512, heads=4, kv=4)
    qa = q.clone().requires_grad_(True)
    qb = q.clone().requires_grad_(True)
    chunked_causal_sdpa(qa, k, v, chunk_hint=64, recompute_scores=False).sum().backward()
    chunked_causal_sdpa(qb, k, v, chunk_hint=64, recompute_scores=True).sum().backward()
    assert torch.allclose(qa.grad, qb.grad, atol=1e-6)


def test_recompute_is_inert_without_grad():
    """No graph, nothing to save — the checkpoint must not fire (it would cost a second
    forward for nothing on the no_grad prefix pass, which is the expensive one)."""
    q, k, v = _qkv(256, heads=4, kv=4)
    with torch.no_grad():
        ref = chunked_causal_sdpa(q, k, v, chunk_hint=32, recompute_scores=False)
        got = chunked_causal_sdpa(q, k, v, chunk_hint=32, recompute_scores=True)
    assert torch.equal(ref, got)


def test_recompute_scores_works_with_a_cached_prefix():
    prefix, tail = 1024, 256
    q, k, v = _cached_qkv(prefix, tail)
    q.requires_grad_(True)
    mask = torch.ones(tail, prefix + tail, dtype=torch.bool).tril(diagonal=prefix)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                           enable_gqa=True)
    got = chunked_causal_sdpa(q, k, v, chunk_hint=64, enable_gqa=True,
                              recompute_scores=True)
    assert torch.equal(ref, got)


def test_recompute_actually_defers_the_scores_to_backward():
    """If `recompute_scores` silently stops firing — e.g. the requires_grad probe goes
    False — the forward and gradients stay correct and the only symptom is that a long
    window OOMs again. Count the SDPA calls instead: with recompute on, backward must
    re-run them."""
    import torch.nn.functional as F
    calls = {"fwd": 0, "bwd": 0}
    phase = ["fwd"]
    real = F.scaled_dot_product_attention

    def counting(*a, **kw):
        calls[phase[0]] += 1
        return real(*a, **kw)

    q, k, v = _qkv(512, heads=4, kv=4)
    q.requires_grad_(True)
    F.scaled_dot_product_attention = counting
    try:
        out = chunked_causal_sdpa(q, k, v, chunk_hint=64, recompute_scores=True)
        n_blocks = calls["fwd"]
        phase[0] = "bwd"
        out.sum().backward()
    finally:
        F.scaled_dot_product_attention = real
    assert n_blocks == 8, f"expected 8 query blocks, got {n_blocks}"
    assert calls["bwd"] == n_blocks, (
        f"backward re-ran {calls['bwd']} of {n_blocks} blocks — scores are being SAVED, "
        "not recomputed, and a long window will OOM")

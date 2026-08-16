"""Unit tests for the HF-side KLD math.

The reductions in ``kld_hf`` are chunked over the vocab dimension because a
248,320-vocab fp32 logits tensor is 8 GB at ctx 8192. Chunking is where this
kind of code goes quietly wrong — an unrescaled running logsumexp or a top-k
merge that loses candidates across a block boundary both yield *plausible*
numbers. These tests pin chunk-size invariance against the naive formulation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quant_tuner.bench.kld_hf import compare_logits  # noqa: E402


def _naive_kld(ref: torch.Tensor, quant: torch.Tensor) -> torch.Tensor:
    """Unchunked reference implementation: full fp32 softmax, no blocking."""
    logp = torch.log_softmax(ref.float(), dim=-1)
    logq = torch.log_softmax(quant.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def test_identical_logits_give_zero_kld_and_full_agreement():
    torch.manual_seed(0)
    logits = torch.randn(16, 512)
    targets = torch.randint(0, 512, (16,))

    out = compare_logits(logits, logits.clone(), targets, vocab_chunk=64)

    assert torch.allclose(out["kld"], torch.zeros(16), atol=1e-6)
    assert out["top1"].mean().item() == 1.0
    assert out["top5"].mean().item() == 1.0
    # An identical model must also reproduce the reference NLL exactly.
    assert torch.allclose(out["ref_nll"], out["quant_nll"], atol=1e-6)


def test_matches_naive_unchunked_kld():
    torch.manual_seed(1)
    ref = torch.randn(24, 500) * 3.0
    quant = ref + torch.randn(24, 500) * 0.4
    targets = torch.randint(0, 500, (24,))

    out = compare_logits(ref, quant, targets, vocab_chunk=64)

    assert torch.allclose(out["kld"], _naive_kld(ref, quant), atol=1e-5)


@pytest.mark.parametrize("vocab_chunk", [1, 7, 64, 499, 500, 4096])
def test_chunk_size_invariance(vocab_chunk):
    """Every block size must agree with the whole-vocab computation.

    Includes chunks that do not divide the vocab evenly (7, 499) and one larger
    than it (4096) — the boundary cases where a rescaling or top-k merge bug
    shows up.
    """
    torch.manual_seed(2)
    ref = torch.randn(8, 500) * 5.0
    quant = ref + torch.randn(8, 500)
    targets = torch.randint(0, 500, (8,))

    whole = compare_logits(ref, quant, targets, vocab_chunk=500)
    chunked = compare_logits(ref, quant, targets, vocab_chunk=vocab_chunk)

    for key in ("kld", "top1", "top5", "ref_nll", "quant_nll"):
        assert torch.allclose(whole[key], chunked[key], atol=1e-4), key


def test_topk_merge_survives_block_boundaries():
    """The true top-5 placed one-per-block must all still be found."""
    vocab = 50
    ref = torch.full((1, vocab), -10.0)
    quant = torch.full((1, vocab), -10.0)
    # Winners spread across five different blocks of width 10.
    winners = [3, 14, 25, 36, 47]
    for rank, idx in enumerate(winners):
        ref[0, idx] = 10.0 - rank  # strictly descending
        quant[0, idx] = 10.0 - rank

    out = compare_logits(ref, quant, torch.tensor([0]), vocab_chunk=10, top_k=5)

    assert out["top1"].item() == 1.0
    assert out["top5"].item() == 1.0
    assert torch.allclose(out["kld"], torch.zeros(1), atol=1e-6)


def test_extreme_logit_magnitudes_stay_finite():
    """Online rescaling must not overflow when one block holds huge logits."""
    ref = torch.tensor([[-500.0, 0.0, 500.0, 100.0]])
    quant = torch.tensor([[-500.0, 0.0, 499.0, 100.0]])

    out = compare_logits(ref, quant, torch.tensor([2]), vocab_chunk=1)

    assert torch.isfinite(out["kld"]).all()
    assert torch.allclose(out["kld"], _naive_kld(ref, quant), atol=1e-5)


def test_two_pass_cache_roundtrip_is_lossless():
    """The two-pass path caches reference logits as fp16; that must not perturb them.

    fp16 has 10 mantissa bits to bf16's 7 and logits sit far inside its ±65504
    range, so the cast is exact. If a future change caches in a narrower dtype
    (or the model emits fp32 logits), this is what catches the drift.
    """
    torch.manual_seed(4)
    ref = (torch.randn(12, 400) * 8.0).to(torch.bfloat16)
    quant = (ref.float() + torch.randn(12, 400) * 0.5).to(torch.bfloat16)
    targets = torch.randint(0, 400, (12,))

    direct = compare_logits(ref, quant, targets, vocab_chunk=128)
    cached = compare_logits(ref.to(torch.float16), quant, targets, vocab_chunk=128)

    for key in ("kld", "top1", "top5", "ref_nll", "quant_nll"):
        assert torch.equal(direct[key], cached[key]), key


def test_kld_is_never_negative():
    """fp32 accumulation over a large vocab can drift below zero; it is clamped."""
    torch.manual_seed(3)
    ref = torch.randn(32, 1024)
    quant = ref + torch.randn(32, 1024) * 1e-4

    out = compare_logits(ref, quant, torch.zeros(32, dtype=torch.long), vocab_chunk=128)

    assert (out["kld"] >= 0).all()

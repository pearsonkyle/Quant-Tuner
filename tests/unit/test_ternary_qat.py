"""Tests for the per-group TWN STE ternarizer (qat/ternary.py).

The load-bearing invariant is step-0 reproduction: the shipped Ternary-Bonsai
weights are exactly ``s_g * c`` per 128-group with an fp16-native ``s_g``, and
``ternarize_group`` must return them bit-identical so a fine-tune provably
starts from the real model. The fp16 scale quantization (deployed Q2_0
numerics) must not disturb that.
"""

import pytest
import torch

from quant_tuner.qat.ternary import (
    DEFAULT_GROUP_SIZE,
    TernaryLinear,
    ternarize_group,
    ternarize_ste,
)


def _synthetic_ternary(out: int = 8, groups: int = 3, zero_frac: float = 0.37,
                       seed: int = 0) -> torch.Tensor:
    """Build a W that is exactly ``s_g * c`` with fp16-native per-group scales."""
    g = torch.Generator().manual_seed(seed)
    inn = groups * DEFAULT_GROUP_SIZE
    # fp16-native positive scales, one per [out, group]
    scale = (torch.rand(out, groups, 1, generator=g) * 0.05 + 1e-3).to(torch.float16).float()
    codes = torch.randint(-1, 2, (out, groups, DEFAULT_GROUP_SIZE), generator=g).float()
    zero_mask = torch.rand(out, groups, DEFAULT_GROUP_SIZE, generator=g) < zero_frac
    codes[zero_mask] = 0.0
    # guarantee at least one nonzero per group so every scale is exercised
    codes[:, :, 0] = 1.0
    return (scale * codes).reshape(out, inn)


def test_step0_roundtrip_bit_exact():
    W = _synthetic_ternary()
    codes, scale, w_hat = ternarize_group(W)
    assert torch.equal(w_hat, W), "ternarize(shipped-style weights) must be identity"
    assert torch.equal(codes, torch.sign(W))
    # every group's recovered scale is the (single) nonzero magnitude
    groups = W.shape[1] // DEFAULT_GROUP_SIZE
    Wg = W.reshape(W.shape[0], groups, DEFAULT_GROUP_SIZE)
    mags = Wg.abs().amax(dim=2)
    assert torch.equal(scale, mags)


def test_all_zero_group_roundtrips():
    W = torch.zeros(2, DEFAULT_GROUP_SIZE)
    codes, scale, w_hat = ternarize_group(W)
    assert torch.equal(w_hat, W)
    assert torch.equal(codes, W)
    assert torch.equal(scale, torch.zeros(2, 1))


def test_scale_is_fp16_representable():
    # arbitrary fp32 weights: the returned scale must already be on the fp16
    # grid (deployed Q2_0 / F16-GGUF numerics), and finite
    W = torch.randn(4, 2 * DEFAULT_GROUP_SIZE, generator=torch.Generator().manual_seed(1))
    _, scale, w_hat = ternarize_group(W)
    assert torch.equal(scale, scale.half().float())
    assert torch.isfinite(scale).all() and torch.isfinite(w_hat).all()


def test_ste_backward_is_identity():
    W = torch.randn(3, DEFAULT_GROUP_SIZE, requires_grad=True)
    ternarize_ste(W).sum().backward()
    assert W.grad is not None
    assert torch.equal(W.grad, torch.ones_like(W))


def test_group_size_must_divide():
    with pytest.raises(ValueError):
        ternarize_group(torch.randn(2, 100))


def test_ternary_linear_forward_matches_manual():
    lin = torch.nn.Linear(2 * DEFAULT_GROUP_SIZE, 4, bias=True)
    with torch.no_grad():
        lin.weight.copy_(_synthetic_ternary(out=4, groups=2, seed=2))
    wrapped = TernaryLinear(lin, trainable=False)
    x = torch.randn(5, 2 * DEFAULT_GROUP_SIZE)
    # shipped-style weights: wrapped forward == plain forward (step-0 identity)
    assert torch.equal(wrapped(x), torch.nn.functional.linear(x, lin.weight, lin.bias))
    assert not lin.weight.requires_grad


def test_ternary_linear_trainable_flag():
    lin = torch.nn.Linear(DEFAULT_GROUP_SIZE, 2)
    assert TernaryLinear(lin, trainable=True).linear.weight.requires_grad

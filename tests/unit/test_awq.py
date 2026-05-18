"""Unit tests for AWQ pure-tensor helpers (no HF model needed)."""

from __future__ import annotations

import torch

from quant_tuner.calibrate.awq import (
    GroupScale,
    ScaleBundle,
    fake_quant_int4_g128,
    fold_rmsnorm_gain,
    proxy_loss,
    scale_from_alpha,
)


def test_fake_quant_preserves_shape_with_padding():
    """Group size 128 must work even when in_features is not divisible by 128."""
    W = torch.randn(8, 200)  # 200 is not a multiple of 128
    Wq = fake_quant_int4_g128(W, group_size=128)
    assert Wq.shape == W.shape


def test_fake_quant_int4_uses_at_most_15_levels():
    """Symmetric INT4 with range [-7, 7] = 15 unique levels per group."""
    W = torch.randn(4, 128) * 5.0
    Wq = fake_quant_int4_g128(W, group_size=128)
    # Per row × group, every value is one of (k * scale) for k in [-7..7].
    # The number of distinct values per row should be ≤ 15.
    for row in Wq:
        assert len(torch.unique(row)) <= 15


def test_fake_quant_zero_input_is_zero_output():
    Wq = fake_quant_int4_g128(torch.zeros(4, 128))
    assert torch.allclose(Wq, torch.zeros_like(Wq))


def test_scale_from_alpha_zero_is_identity():
    s = torch.tensor([0.1, 1.0, 10.0])
    scale = scale_from_alpha(s, alpha=0.0)
    assert torch.allclose(scale, torch.ones_like(s))


def test_scale_from_alpha_has_unit_geometric_mean():
    """For α > 0, the resulting scale vector must have geomean = 1.0.

    This is the property that keeps the row-wise quantizer scale roughly
    unchanged after the AWQ weight transform.
    """
    s = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
    for a in (0.25, 0.5, 1.0):
        scale = scale_from_alpha(s, alpha=a)
        log_mean = scale.log().mean().item()
        assert abs(log_mean) < 1e-5, f"alpha={a}: log-mean={log_mean}"


def test_proxy_loss_is_zero_when_scale_is_one_and_weights_quant_exact():
    """If W is already quantization-exact (already int4 g128), the proxy loss is 0
    at scale=1 (no scaling, fake_quant is the identity for already-quantized W)."""
    W = torch.zeros(4, 128)
    X = torch.randn(16, 128)
    scale = torch.ones(128)
    loss = proxy_loss(W, X, scale)
    assert loss == 0.0


def test_proxy_loss_is_nonneg():
    W = torch.randn(4, 128) * 0.05
    X = torch.randn(8, 128)
    scale = scale_from_alpha(X.abs().mean(dim=0), alpha=0.5)
    assert proxy_loss(W, X, scale) >= 0.0


def test_fold_rmsnorm_gain_plus_one_is_invariant_in_f32():
    """Qwen3.5: (1 + γ') · norm_out · scale must equal (1 + γ) · norm_out exactly in f32."""
    gamma = torch.tensor([0.1, -0.05, 0.3, 0.0])
    scale = torch.tensor([2.0, 0.5, 1.5, 1.0])
    new_gain = fold_rmsnorm_gain(gamma, 1.0 / scale, plus_one=True)
    # (1 + new_gain) * scale should equal (1 + gamma)
    reconstructed = (1.0 + new_gain) * scale
    torch.testing.assert_close(reconstructed, 1.0 + gamma)


def test_fold_rmsnorm_gain_bare_is_invariant_in_f32():
    """Llama/Mistral/Qwen3: γ' · norm_out · scale must equal γ · norm_out exactly."""
    gamma = torch.tensor([0.1, -0.05, 0.3, 1.0])
    scale = torch.tensor([2.0, 0.5, 1.5, 1.0])
    new_gain = fold_rmsnorm_gain(gamma, 1.0 / scale, plus_one=False)
    torch.testing.assert_close(new_gain * scale, gamma.float())


def test_awq_invariance_end_to_end_in_f32():
    """The whole transform — scale W columns, divide gain — must preserve the f32
    linear-after-norm output exactly."""
    in_f, out_f = 8, 4
    gamma = torch.randn(in_f) * 0.2
    W = torch.randn(out_f, in_f)
    # Some "pre-norm" input; the norm-output for one token is conceptually
    # norm_out · (1 + gamma) — we model that here directly.
    norm_out = torch.randn(in_f)
    y_before = W @ (norm_out * (1.0 + gamma))

    scale = scale_from_alpha(norm_out.abs() + 0.1, alpha=0.5)
    inv = 1.0 / scale
    new_W = W * scale.unsqueeze(0)
    new_gain = fold_rmsnorm_gain(gamma, inv, plus_one=True)
    y_after = new_W @ (norm_out * (1.0 + new_gain))

    torch.testing.assert_close(y_after, y_before, rtol=1e-5, atol=1e-5)


def test_scale_bundle_roundtrip(tmp_path):
    b = ScaleBundle(groups=[
        GroupScale(
            group_id="L0_attn",
            anchor="model.layers.0.self_attn.q_proj",
            members=(
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ),
            prev_norm="model.layers.0.input_layernorm",
            scale=torch.tensor([1.1, 0.9, 1.2, 0.8], dtype=torch.float32),
            alpha=0.5,
        ),
        GroupScale(
            group_id="L0_mlp",
            anchor="model.layers.0.mlp.gate_proj",
            members=("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"),
            prev_norm="model.layers.0.post_attention_layernorm",
            scale=torch.tensor([1.0, 1.0], dtype=torch.float32),
            alpha=0.0,
        ),
    ])
    path = tmp_path / "awq.pt"
    b.save(path)
    loaded = ScaleBundle.load(path)

    assert len(loaded.groups) == 2
    for orig, got in zip(b.groups, loaded.groups, strict=True):
        assert got.group_id == orig.group_id
        assert got.anchor == orig.anchor
        assert got.members == orig.members
        assert got.prev_norm == orig.prev_norm
        assert got.alpha == orig.alpha
        torch.testing.assert_close(got.scale, orig.scale)

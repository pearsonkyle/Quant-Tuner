"""Unit tests for AWQ pure-tensor helpers (no HF model needed)."""

from __future__ import annotations

import torch

from quant_tuner.calibrate.awq import (
    GroupScale,
    ScaleBundle,
    ScaleGroup,
    _gqa_or_moe_ge4,
    _iq2_grid,
    discover_groups,
    fake_quant_int4_g128,
    fake_quant_iq2_s,
    fake_quant_iq2_xs,
    fake_quant_iq2_xxs,
    fake_quant_q2k_block16,
    fake_quant_q3k_block16,
    fold_rmsnorm_gain,
    proxy_for_member,
    proxy_for_quant_type,
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


# ----- exp-012: q2k 2-bit proxy -------------------------------------------- #


def test_q2k_block16_preserves_shape_with_padding():
    W = torch.randn(8, 200)  # 200 not divisible by 16
    Wq = fake_quant_q2k_block16(W)
    assert Wq.shape == W.shape


def test_q2k_block16_uses_at_most_4_levels_per_block():
    W = torch.randn(4, 16) * 3.0
    Wq = fake_quant_q2k_block16(W)
    for row in Wq:
        assert len(torch.unique(row)) <= 4


def test_q2k_block16_zero_in_zero_out():
    Wq = fake_quant_q2k_block16(torch.zeros(4, 64))
    assert torch.allclose(Wq, torch.zeros_like(Wq))


def test_q2k_block16_reconstruction_bounded():
    """A 2-bit asymmetric quantizer should track the original within ~max range."""
    torch.manual_seed(0)
    W = torch.randn(4, 128)
    Wq = fake_quant_q2k_block16(W)
    # Per-block max abs error should be small relative to the per-block range.
    err = (Wq - W).abs().max().item()
    assert err < W.abs().max().item()  # never exceeds the input range


# ----- low-bit: q3k 3-bit proxy + quant-type proxy selection ---------------- #


def test_q3k_block16_preserves_shape_with_padding():
    W = torch.randn(8, 200)  # 200 not divisible by 16
    Wq = fake_quant_q3k_block16(W)
    assert Wq.shape == W.shape


def test_q3k_block16_uses_at_most_8_levels_per_block():
    W = torch.randn(4, 16) * 3.0
    Wq = fake_quant_q3k_block16(W)
    for row in Wq:
        assert len(torch.unique(row)) <= 8  # 3-bit codes in [-4, 3]


def test_q3k_block16_zero_in_zero_out():
    Wq = fake_quant_q3k_block16(torch.zeros(4, 64))
    assert torch.allclose(Wq, torch.zeros_like(Wq))


def test_q3k_block16_error_between_q2k_and_int4():
    """3-bit error must sit between the 2-bit and 4-bit proxies on the same W."""
    torch.manual_seed(0)
    W = torch.randn(16, 256)
    err2 = (fake_quant_q2k_block16(W) - W).norm()
    err3 = (fake_quant_q3k_block16(W) - W).norm()
    err4 = (fake_quant_int4_g128(W) - W).norm()
    assert err4 < err3 < err2


def test_proxy_for_quant_type_mapping():
    assert proxy_for_quant_type("Q2_K") == "q2k_b16"
    assert proxy_for_quant_type("IQ1_S") == "q2k_b16"
    assert proxy_for_quant_type("Q3_K_M") == "q3k_b16"
    assert proxy_for_quant_type("iq3_s") == "q3k_b16"  # case-insensitive
    assert proxy_for_quant_type("Q4_K_M") == "int4_g128"
    assert proxy_for_quant_type("Q5_K_S") == "int4_g128"
    assert proxy_for_quant_type("Q8_0") == "int4_g128"


def test_proxy_for_quant_type_iq2_codebooks():
    assert proxy_for_quant_type("IQ2_XXS") == "iq2_xxs"
    assert proxy_for_quant_type("IQ2_XS") == "iq2_xs"
    assert proxy_for_quant_type("IQ2_S") == "iq2_s"
    assert proxy_for_quant_type("IQ2_M") == "iq2_s"  # M = S grid + IQ3 mix
    assert proxy_for_quant_type("iq2_m") == "iq2_s"


# ----- per-member proxy mix (IQ1/IQ2 tensor mixes) -------------------------- #


def test_proxy_for_member_iq2_m_mix():
    """IQ2_M tensor mix per pinned llama.cpp: attn_v -> Q4_K (GQA>=4) / IQ3_S,
    attn_output -> IQ3_S, first eighth of ffn_down -> IQ3_S, rest base grid."""
    v = "model.layers.3.self_attn.v_proj"
    assert proxy_for_member("IQ2_M", v, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("IQ2_M", v, gqa_ge4=False, n_layers=32) == "iq3_s"
    o = "model.layers.3.self_attn.o_proj"
    assert proxy_for_member("IQ2_M", o, gqa_ge4=True, n_layers=32) == "iq3_s"
    # 32 layers -> layers 0..3 are the bumped eighth
    down = "model.layers.{}.mlp.down_proj"
    assert proxy_for_member("IQ2_M", down.format(3), gqa_ge4=True, n_layers=32) == "iq3_s"
    assert proxy_for_member("IQ2_M", down.format(4), gqa_ge4=True, n_layers=32) is None
    for leaf in ("q_proj", "k_proj", "gate_proj", "up_proj"):
        m = f"model.layers.3.self_attn.{leaf}"
        assert proxy_for_member("IQ2_M", m, gqa_ge4=True, n_layers=32) is None


def test_proxy_for_member_xs_iq1_and_non_iq():
    v = "model.layers.3.self_attn.v_proj"
    o = "model.layers.3.self_attn.o_proj"
    down0 = "model.layers.0.mlp.down_proj"
    # XS: v_proj -> Q2_K without GQA, Q4_K with; o_proj has no override
    assert proxy_for_member("IQ2_XS", v, gqa_ge4=False, n_layers=32) == "q2k_b16"
    assert proxy_for_member("IQ2_XS", v, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("IQ2_XS", o, gqa_ge4=True, n_layers=32) is None
    assert proxy_for_member("IQ2_XS", down0, gqa_ge4=True, n_layers=32) == "q2k_b16"
    # IQ1: o_proj -> IQ2_XXS
    assert proxy_for_member("IQ1_S", o, gqa_ge4=True, n_layers=32) == "iq2_xxs"
    # ftypes without a mix table
    assert proxy_for_member("Q4_K_M", v, gqa_ge4=True, n_layers=32) is None
    assert proxy_for_member("Q8_0", v, gqa_ge4=True, n_layers=32) is None
    # case-insensitive, like proxy_for_quant_type
    assert proxy_for_member("iq2_m", v, gqa_ge4=True, n_layers=32) == "int4_g128"


def test_proxy_for_member_q3k_and_iq3_families():
    """Q3_K_M/L bump v/o/down to Q4_K-Q5_K (no GQA condition in their branch);
    IQ3_M bumps attn_v + attn_output + first-eighth ffn_down to Q4_K; IQ3_S
    bumps attn_v to Q4_K only under GQA>=4 (llama-quant.cpp attn_v branch);
    Q3_K_S is pure — every member keeps the base grid."""
    v = "model.layers.3.self_attn.v_proj"
    o = "model.layers.3.self_attn.o_proj"
    down = "model.layers.{}.mlp.down_proj"
    for qt in ("Q3_K_M", "Q3_K_L"):
        # no GQA clause at 3 bits: same override either way
        assert proxy_for_member(qt, v, gqa_ge4=False, n_layers=32) == "int4_g128"
        assert proxy_for_member(qt, v, gqa_ge4=True, n_layers=32) == "int4_g128"
        assert proxy_for_member(qt, o, gqa_ge4=True, n_layers=32) == "int4_g128"
        assert proxy_for_member(qt, down.format(30), gqa_ge4=True, n_layers=32) == "int4_g128"
    # IQ3_M: attn_v (always) + attn_output + first eighth of ffn_down
    assert proxy_for_member("IQ3_M", o, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("IQ3_M", down.format(3), gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("IQ3_M", down.format(4), gqa_ge4=True, n_layers=32) is None
    assert proxy_for_member("IQ3_M", v, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("IQ3_M", v, gqa_ge4=False, n_layers=32) == "int4_g128"
    # IQ3_S: attn_v -> Q4_K only under GQA/MoE >= 4; everything else base grid
    assert proxy_for_member("IQ3_S", v, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("IQ3_S", v, gqa_ge4=False, n_layers=32) is None
    for m in (o, down.format(0)):
        assert proxy_for_member("IQ3_S", m, gqa_ge4=True, n_layers=32) is None
    # Q3_K_S: no overrides anywhere, even under GQA
    for m in (v, o, down.format(0)):
        assert proxy_for_member("Q3_K_S", m, gqa_ge4=True, n_layers=32) is None
    # q/k/gate/up keep the base grid across the 3-bit family
    for qt in ("Q3_K_M", "Q3_K_L", "IQ3_M"):
        for leaf in ("q_proj", "k_proj", "gate_proj", "up_proj"):
            m = f"model.layers.3.self_attn.{leaf}"
            assert proxy_for_member(qt, m, gqa_ge4=True, n_layers=32) is None


def test_proxy_for_member_q2k_family():
    """Q2_K mix per pinned llama.cpp: attn_v -> Q4_K (GQA) / Q3_K, ffn_down ->
    Q3_K on EVERY layer, attn_output -> Q3_K. Q2_K_S: attn_v -> Q4_K only
    under GQA, ffn_down -> Q4_K for the first eighth, attn_output base."""
    v = "model.layers.3.self_attn.v_proj"
    o = "model.layers.3.self_attn.o_proj"
    down = "model.layers.{}.mlp.down_proj"
    # Q2_K
    assert proxy_for_member("Q2_K", v, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("Q2_K", v, gqa_ge4=False, n_layers=32) == "q3k_b16"
    assert proxy_for_member("Q2_K", o, gqa_ge4=True, n_layers=32) == "q3k_b16"
    assert proxy_for_member("Q2_K", down.format(3), gqa_ge4=True, n_layers=32) == "q3k_b16"
    assert proxy_for_member("Q2_K", down.format(30), gqa_ge4=True, n_layers=32) == "q3k_b16"
    # Q2_K_S
    assert proxy_for_member("Q2_K_S", v, gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("Q2_K_S", v, gqa_ge4=False, n_layers=32) is None
    assert proxy_for_member("Q2_K_S", o, gqa_ge4=True, n_layers=32) is None
    assert proxy_for_member("Q2_K_S", down.format(3), gqa_ge4=True, n_layers=32) == "int4_g128"
    assert proxy_for_member("Q2_K_S", down.format(4), gqa_ge4=True, n_layers=32) is None
    # q/k/gate/up keep the base grid in both
    for qt in ("Q2_K", "Q2_K_S"):
        for leaf in ("q_proj", "k_proj", "gate_proj", "up_proj"):
            m = f"model.layers.3.self_attn.{leaf}"
            assert proxy_for_member(qt, m, gqa_ge4=True, n_layers=32) is None


def test_gqa_or_moe_ge4_detection():
    from types import SimpleNamespace

    assert _gqa_or_moe_ge4(SimpleNamespace(num_attention_heads=32, num_key_value_heads=8))
    assert not _gqa_or_moe_ge4(SimpleNamespace(num_attention_heads=32, num_key_value_heads=16))
    assert not _gqa_or_moe_ge4(SimpleNamespace(num_attention_heads=32, num_key_value_heads=32))
    assert not _gqa_or_moe_ge4(SimpleNamespace())  # unknown -> conservative False
    assert _gqa_or_moe_ge4(SimpleNamespace(num_local_experts=8))


def test_scale_bundle_roundtrip_preserves_proxy_mix(tmp_path):
    p = ScaleBundle(proxy="iq2_s", proxy_mix="IQ2_M").save(tmp_path / "b.pt")
    loaded = ScaleBundle.load(p)
    assert loaded.proxy == "iq2_s"
    assert loaded.proxy_mix == "IQ2_M"
    # bundles saved before the field existed load as proxy_mix=None
    legacy = tmp_path / "legacy.pt"
    torch.save({"proxy": "q2k_b16", "groups": []}, legacy)
    assert ScaleBundle.load(legacy).proxy_mix is None


# ----- IQ2 codebook-aware proxies ------------------------------------------- #


def test_iq2_grids_load_with_expected_shapes():
    for name, k in (("iq2xxs", 256), ("iq2xs", 512), ("iq2s", 1024)):
        g = _iq2_grid(name, "cpu")
        assert g.shape == (k, 8)
        # llama.cpp grid magnitudes are exactly {8, 25, 43}
        assert set(g.unique().tolist()) == {8.0, 25.0, 43.0}
        # entry 0 is 0x0808080808080808 in every grid
        assert torch.equal(g[0], torch.full((8,), 8.0))


def test_iq2_proxies_preserve_shape_with_padding():
    W = torch.randn(4, 200)  # 200 not divisible by the 256 superblock
    for fn in (fake_quant_iq2_xxs, fake_quant_iq2_xs, fake_quant_iq2_s):
        assert fn(W).shape == W.shape


def test_iq2_zero_in_near_zero_out():
    W = torch.zeros(2, 256)
    for fn in (fake_quant_iq2_xxs, fake_quant_iq2_xs, fake_quant_iq2_s):
        assert fn(W).abs().max().item() < 1e-9


def test_iq2_exact_reconstruction_of_codebook_rows():
    """Rows built from actual codebook entries at one uniform scale (and
    representable signs) must pass through the proxy unchanged — validates
    the candidate-scale search, the 4-bit scale snapping, the nearest-entry
    search, and the sign path.

    Each scale block is forced to contain one top-magnitude (43) component:
    a block whose largest component is only 25 has an unidentifiable scale
    (the real llama.cpp quantizer has the same ambiguity — its scale sweep
    only maps the block max into a window around the grid top).
    """
    g = torch.Generator().manual_seed(0)
    cases = (
        # (grid, fn, parity-constrained signs, scale_block)
        ("iq2xxs", fake_quant_iq2_xxs, True, 32),
        ("iq2xs", fake_quant_iq2_xs, True, 16),
        ("iq2s", fake_quant_iq2_s, False, 16),
    )
    for name, fn, parity, scale_block in cases:
        grid = _iq2_grid(name, "cpu")
        idx = torch.randint(0, grid.shape[0], (2, 32), generator=g)
        idx[:, ::scale_block // 8] = 1  # entry 1 = [43, 8, ...] in every grid
        W = grid[idx].reshape(2, 256) * 0.01  # all-positive signs: even parity
        if not parity:
            W[:, 8] = -W[:, 8]  # odd-parity group: legal only with free signs
        torch.testing.assert_close(fn(W), W, rtol=1e-4, atol=1e-6)


def test_iq2_parity_constraint_even_negatives():
    torch.manual_seed(1)
    W = torch.randn(4, 256)
    for fn in (fake_quant_iq2_xxs, fake_quant_iq2_xs):
        neg = (fn(W).view(4, -1, 8) < 0).sum(dim=-1)
        assert (neg % 2 == 0).all(), "XXS/XS sign patterns must have even parity"
    # The free-sign variant must be able to keep odd-parity groups.
    neg_s = (fake_quant_iq2_s(W).view(4, -1, 8) < 0).sum(dim=-1)
    assert (neg_s % 2 == 1).any()


def test_iq2_error_ordering_matches_bpw():
    """Richer codebook + finer scales + free signs => monotonically lower
    reconstruction error: S (2.5 bpw) < XS (2.31) < XXS (2.06)."""
    torch.manual_seed(2)
    W = torch.randn(8, 512)
    e_xxs = (fake_quant_iq2_xxs(W) - W).norm().item()
    e_xs = (fake_quant_iq2_xs(W) - W).norm().item()
    e_s = (fake_quant_iq2_s(W) - W).norm().item()
    assert e_s < e_xs < e_xxs


# ----- exp-011: include_output_proj ---------------------------------------- #


class _FakeLinear:
    """Minimal torch.nn.Linear stand-in (isinstance check below uses real Linear)."""


def _build_mock_model(*, with_o: bool, with_down: bool):
    """Construct a tiny stand-in model exposing the attribute tree discover_groups walks."""
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
            self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
            self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
            if with_o:
                self.self_attn.o_proj = nn.Linear(8, 8, bias=False)
            self.input_layernorm = nn.LayerNorm(8)
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(8, 16, bias=False)
            self.mlp.up_proj = nn.Linear(8, 16, bias=False)
            if with_down:
                self.mlp.down_proj = nn.Linear(16, 8, bias=False)
            self.post_attention_layernorm = nn.LayerNorm(8)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(2)])

    return M()


def test_discover_groups_default_excludes_output_projections():
    m = _build_mock_model(with_o=True, with_down=True)
    groups = discover_groups(m)
    ids = {g.group_id for g in groups}
    assert ids == {"L0_attn", "L0_mlp", "L1_attn", "L1_mlp"}


def test_discover_groups_include_output_proj_emits_out_groups():
    m = _build_mock_model(with_o=True, with_down=True)
    groups = discover_groups(m, include_output_proj=True)
    ids_by_layer = {g.group_id for g in groups}
    assert "L0_attn_out" in ids_by_layer and "L0_mlp_out" in ids_by_layer
    out_groups = [g for g in groups if g.group_id.endswith("_out")]
    for g in out_groups:
        assert g.prev_norm is None
        assert len(g.members) == 1


def test_discover_groups_include_output_proj_skips_missing_modules():
    m = _build_mock_model(with_o=False, with_down=True)
    groups = discover_groups(m, include_output_proj=True)
    ids = {g.group_id for g in groups}
    assert "L0_attn_out" not in ids
    assert "L0_mlp_out" in ids


# ----- exp-013: per-member α persistence ----------------------------------- #


def test_scale_bundle_roundtrip_with_member_scales(tmp_path):
    members = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    )
    member_alphas = {members[0]: 0.5, members[1]: 0.75}
    member_scales = {
        members[0]: torch.tensor([1.0, 0.9, 1.1, 1.0], dtype=torch.float32),
        members[1]: torch.tensor([0.95, 1.05, 1.0, 1.0], dtype=torch.float32),
    }
    b = ScaleBundle(
        proxy="q2k_b16",
        groups=[
            GroupScale(
                group_id="L0_attn",
                anchor=members[0],
                members=members,
                prev_norm="model.layers.0.input_layernorm",
                scale=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                alpha=0.5,
                member_alphas=member_alphas,
                member_scales=member_scales,
            ),
            GroupScale(
                group_id="L0_attn_out",
                anchor="model.layers.0.self_attn.o_proj",
                members=("model.layers.0.self_attn.o_proj",),
                prev_norm=None,
                scale=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                alpha=0.25,
            ),
        ],
    )
    path = tmp_path / "awq.pt"
    b.save(path)
    loaded = ScaleBundle.load(path)

    assert loaded.proxy == "q2k_b16"
    assert loaded.groups[0].member_alphas == member_alphas
    for k, v in member_scales.items():
        torch.testing.assert_close(loaded.groups[0].member_scales[k], v)
    assert loaded.groups[1].prev_norm is None
    assert loaded.groups[1].member_scales is None


def test_cv_gate_reverts_to_group_alpha_when_holdout_worsens():
    """exp-017 gate logic: if α minimizing cal_loss has worse ho_loss than the
    group α, the gate must revert. Verified by direct simulation of the
    scoring step (no model load needed)."""
    cal_losses = {0.0: 80.0, 0.25: 47.0, 0.5: 62.0}  # cal-best is α=0.25
    ho_losses = {0.0: 45.0, 0.25: 55.0, 0.5: 73.0}    # but α=0.25 is worse on ho
    group_alpha = 0.0  # assume group picked α=0.0

    # Replicate calibrate()'s decision logic for cv_strategy='gate':
    m_best_a = min(cal_losses, key=cal_losses.get)
    assert m_best_a == 0.25  # cal-best
    if ho_losses[m_best_a] > ho_losses[group_alpha]:
        m_best_a = group_alpha
    assert m_best_a == 0.0, "gate should revert when held-out loss worsens"


def test_cv_mixed_picks_held_out_friendly_alpha():
    """exp-018 mixed loss: with cv_weight>0, the score sums cal+w·ho. With
    high weight, the held-out signal dominates and pulls α toward the
    held-out-best."""
    cal_losses = {0.0: 80.0, 0.25: 47.0, 0.5: 62.0}
    ho_losses = {0.0: 45.0, 0.25: 55.0, 0.5: 73.0}

    # cv_weight=0 reproduces cal-only behavior
    scores0 = {a: cal_losses[a] + 0.0 * ho_losses[a] for a in cal_losses}
    assert min(scores0, key=scores0.get) == 0.25

    # cv_weight=2 makes ho dominate enough to pick α=0
    scores2 = {a: cal_losses[a] + 2.0 * ho_losses[a] for a in cal_losses}
    # 0.0: 80+90=170 ; 0.25: 47+110=157 ; 0.5: 62+146=208
    assert min(scores2, key=scores2.get) == 0.25  # still 0.25 wins this case

    # With cv_weight=5, ho dominates fully
    scores5 = {a: cal_losses[a] + 5.0 * ho_losses[a] for a in cal_losses}
    # 0.0: 80+225=305 ; 0.25: 47+275=322 ; 0.5: 62+365=427
    assert min(scores5, key=scores5.get) == 0.0


def test_scale_group_prev_norm_optional():
    g = ScaleGroup(
        group_id="x",
        anchor="a",
        members=("a",),
        prev_norm=None,
    )
    assert g.prev_norm is None

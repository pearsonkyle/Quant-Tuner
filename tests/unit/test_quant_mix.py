"""Tests for the shared llama-quantize tensor-mix table (calibrate/_quant_mix).

Mirrors ``llama_tensor_get_type`` in the pinned llama.cpp
(``vendor/llama.cpp/src/llama-quant.cpp`` @ f3e18281). The AWQ-proxy view of
the same table is tested in test_awq.py; these tests cover the raw ftype tags
plus the 4-bit branches only GPTQ consumes.
"""

from types import SimpleNamespace

from quant_tuner.calibrate._quant_mix import (
    gqa_or_moe_ge4,
    target_type_for_member,
    use_more_bits,
)

V = "model.layers.{}.self_attn.v_proj"
OPROJ = "model.layers.{}.self_attn.o_proj"
DOWN = "model.layers.{}.mlp.down_proj"
BASE_MEMBERS = ("q_proj", "k_proj", "gate_proj", "up_proj")


def test_iq2_family_targets():
    # IQ2_M (S/M branch): v -> Q4_K (GQA) / IQ3_S, o -> IQ3_S, first-eighth down -> IQ3_S
    assert target_type_for_member("IQ2_M", V.format(3), gqa_ge4=True, n_layers=32) == "Q4_K"
    assert target_type_for_member("IQ2_M", V.format(3), gqa_ge4=False, n_layers=32) == "IQ3_S"
    assert target_type_for_member("IQ2_M", OPROJ.format(3), gqa_ge4=True, n_layers=32) == "IQ3_S"
    assert target_type_for_member("IQ2_M", DOWN.format(3), gqa_ge4=True, n_layers=32) == "IQ3_S"
    assert target_type_for_member("IQ2_M", DOWN.format(4), gqa_ge4=True, n_layers=32) is None
    # IQ2_XS (non-S/M): v -> Q2_K without GQA, o -> base, first-eighth down -> Q2_K
    assert target_type_for_member("IQ2_XS", V.format(3), gqa_ge4=False, n_layers=32) == "Q2_K"
    assert target_type_for_member("IQ2_XS", OPROJ.format(3), gqa_ge4=True, n_layers=32) is None
    assert target_type_for_member("IQ2_XS", DOWN.format(0), gqa_ge4=True, n_layers=32) == "Q2_K"
    # IQ1: o -> IQ2_XXS
    assert target_type_for_member("IQ1_S", OPROJ.format(3), gqa_ge4=True, n_layers=32) == "IQ2_XXS"
    for leaf in BASE_MEMBERS:
        m = f"model.layers.3.self_attn.{leaf}"
        assert target_type_for_member("IQ2_M", m, gqa_ge4=True, n_layers=32) is None


def test_q2k_family_targets():
    assert target_type_for_member("Q2_K", V.format(3), gqa_ge4=True, n_layers=32) == "Q4_K"
    assert target_type_for_member("Q2_K", V.format(3), gqa_ge4=False, n_layers=32) == "Q3_K"
    assert target_type_for_member("Q2_K", OPROJ.format(3), gqa_ge4=True, n_layers=32) == "Q3_K"
    # ffn_down bumped on EVERY layer for Q2_K
    assert target_type_for_member("Q2_K", DOWN.format(30), gqa_ge4=True, n_layers=32) == "Q3_K"
    # Q2_K_S: v only under GQA, down only first eighth, o base
    assert target_type_for_member("Q2_K_S", V.format(3), gqa_ge4=False, n_layers=32) is None
    assert target_type_for_member("Q2_K_S", OPROJ.format(3), gqa_ge4=True, n_layers=32) is None
    assert target_type_for_member("Q2_K_S", DOWN.format(3), gqa_ge4=True, n_layers=32) == "Q4_K"
    assert target_type_for_member("Q2_K_S", DOWN.format(4), gqa_ge4=True, n_layers=32) is None


def test_q3k_and_iq3_family_targets():
    for qt in ("Q3_K_M", "Q3_K_L"):
        for m in (V.format(3), OPROJ.format(3), DOWN.format(30)):
            assert target_type_for_member(qt, m, gqa_ge4=False, n_layers=32) == "Q4_K"
    # IQ3_M: attn_v always Q4_K (llama-quant.cpp attn_v branch), o + first-eighth down
    assert target_type_for_member("IQ3_M", V.format(3), gqa_ge4=False, n_layers=32) == "Q4_K"
    assert target_type_for_member("IQ3_M", OPROJ.format(3), gqa_ge4=False, n_layers=32) == "Q4_K"
    assert target_type_for_member("IQ3_M", DOWN.format(3), gqa_ge4=False, n_layers=32) == "Q4_K"
    assert target_type_for_member("IQ3_M", DOWN.format(4), gqa_ge4=False, n_layers=32) is None
    # IQ3_S: attn_v -> Q4_K only under GQA/MoE >= 4; pure otherwise
    assert target_type_for_member("IQ3_S", V.format(3), gqa_ge4=True, n_layers=32) == "Q4_K"
    assert target_type_for_member("IQ3_S", V.format(3), gqa_ge4=False, n_layers=32) is None
    assert target_type_for_member("IQ3_S", OPROJ.format(3), gqa_ge4=True, n_layers=32) is None
    assert target_type_for_member("IQ3_S", DOWN.format(0), gqa_ge4=True, n_layers=32) is None
    # Q3_K_S: pure
    for m in (V.format(3), OPROJ.format(3), DOWN.format(0)):
        assert target_type_for_member("Q3_K_S", m, gqa_ge4=True, n_layers=32) is None


def test_q4_k_m_use_more_bits_schedule():
    """Q4_K_M: attn_v + ffn_down -> Q6_K on the use_more_bits layer schedule
    (first eighth, last eighth, every third in between)."""
    n = 32
    for i in range(n):
        expect = "Q6_K" if use_more_bits(i, n) else None
        assert target_type_for_member("Q4_K_M", V.format(i), gqa_ge4=False, n_layers=n) == expect
        assert (
            target_type_for_member("Q4_K_M", DOWN.format(i), gqa_ge4=False, n_layers=n) == expect
        )
    # boundary sanity for the schedule itself: 32 layers -> 0..3 and 28..31
    # bumped, plus (i-4) % 3 == 2 in between
    assert [i for i in range(8) if use_more_bits(i, 8)] == [0, 3, 6, 7]
    assert use_more_bits(0, 32) and use_more_bits(3, 32)
    assert not use_more_bits(4, 32)
    assert use_more_bits(6, 32)  # (6-4) % 3 == 2
    assert use_more_bits(28, 32) and use_more_bits(31, 32)
    # other members never bumped
    assert target_type_for_member("Q4_K_M", OPROJ.format(0), gqa_ge4=True, n_layers=32) is None
    for leaf in BASE_MEMBERS:
        m = f"model.layers.0.self_attn.{leaf}"
        assert target_type_for_member("Q4_K_M", m, gqa_ge4=True, n_layers=32) is None


def test_iq4_xs_targets():
    """IQ4_XS: attn_v -> Q5_K under GQA/MoE >= 4 only. The ffn_down
    first-eighth bump requires no-imatrix, which never happens in this
    pipeline, so it is deliberately not modeled."""
    assert target_type_for_member("IQ4_XS", V.format(3), gqa_ge4=True, n_layers=32) == "Q5_K"
    assert target_type_for_member("IQ4_XS", V.format(3), gqa_ge4=False, n_layers=32) is None
    assert target_type_for_member("IQ4_XS", DOWN.format(0), gqa_ge4=True, n_layers=32) is None
    assert target_type_for_member("IQ4_XS", OPROJ.format(3), gqa_ge4=True, n_layers=32) is None


def test_no_mix_without_layer_metadata():
    # n_layers unknown -> layer-indexed bumps conservatively off, static ones stay
    assert target_type_for_member("Q4_K_M", V.format(0), gqa_ge4=True, n_layers=None) is None
    assert target_type_for_member("IQ2_M", DOWN.format(0), gqa_ge4=True, n_layers=None) is None
    assert target_type_for_member("Q2_K", DOWN.format(0), gqa_ge4=True, n_layers=None) == "Q3_K"
    # pure/unknown ftypes: never any override
    for qt in ("Q8_0", "Q6_K", "F16"):
        assert target_type_for_member(qt, V.format(0), gqa_ge4=True, n_layers=32) is None


def test_gqa_or_moe_ge4_detection():
    assert gqa_or_moe_ge4(SimpleNamespace(num_attention_heads=32, num_key_value_heads=8))
    assert not gqa_or_moe_ge4(SimpleNamespace(num_attention_heads=32, num_key_value_heads=16))
    assert not gqa_or_moe_ge4(SimpleNamespace())  # unknown -> conservative False
    assert gqa_or_moe_ge4(SimpleNamespace(num_local_experts=8))
    assert gqa_or_moe_ge4(SimpleNamespace(num_experts=4))

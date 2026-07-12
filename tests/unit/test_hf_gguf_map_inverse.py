"""Inverse HF<->GGUF name mapping tests (fixes measure_quant_noise --empirical)."""

from __future__ import annotations

from quant_tuner.models.hf_gguf_map import (
    _HF_TO_GGUF_RULES,
    gguf_to_hf_name,
    gguf_to_hf_names,
    map_hf_to_gguf,
)


def test_every_rule_roundtrips():
    for pat, _tmpl in _HF_TO_GGUF_RULES:
        hf = pat.pattern.lstrip("^").rstrip("$").replace(r"\.", ".").replace(r"(\d+)", "7")
        gguf = map_hf_to_gguf(hf)
        assert gguf is not None, f"forward map miss for {hf}"
        assert hf in gguf_to_hf_names(gguf), f"inverse map miss: {hf} -> {gguf} -> {gguf_to_hf_names(gguf)}"


def test_attn_output_has_two_candidates():
    names = gguf_to_hf_names("blk.3.attn_output.weight")
    assert "model.layers.3.self_attn.o_proj" in names
    assert "model.layers.3.self_attn.out_proj" in names


def test_weight_suffix_tolerated():
    assert gguf_to_hf_names("blk.0.ffn_up") == gguf_to_hf_names("blk.0.ffn_up.weight")


def test_output_maps_to_lm_head():
    assert gguf_to_hf_names("output.weight") == ["lm_head"]


def test_gguf_to_hf_name_first_candidate():
    assert gguf_to_hf_name("blk.5.ffn_down.weight") == "model.layers.5.mlp.down_proj"


def test_unknown_returns_empty():
    assert gguf_to_hf_names("blk.0.nonexistent.weight") == []
    assert gguf_to_hf_name("totally.unknown") is None

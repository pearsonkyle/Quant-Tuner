"""Unit tests for imatrix calibrator helpers (no model load required)."""

import numpy as np

from quant_tuner.calibrate.imatrix import _col_l2_sq, _l1_normalize
from quant_tuner.models.hf_gguf_map import is_ssm, map_hf_to_gguf


def test_col_l2_sq_sums_over_rows():
    # GGUF stores weights as [n_out, n_in]. We want ||W[:, c]||^2 per input channel c.
    w = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -3.0]], dtype=np.float32)  # 2x3
    out = _col_l2_sq(w)
    # cols: 1^2+4^2=17, 2^2+0^2=4, 3^2+3^2=18
    np.testing.assert_allclose(out, [17.0, 4.0, 18.0])


def test_l1_normalize_preserves_zero_vector():
    v = np.zeros(8, dtype=np.float32)
    np.testing.assert_array_equal(_l1_normalize(v), v)


def test_l1_normalize_makes_mean_magnitude_one():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    n = _l1_normalize(v)
    # Mean magnitude should now be 1.0
    assert abs(float(np.abs(n).mean()) - 1.0) < 1e-6


def test_map_hf_to_gguf_attention():
    assert map_hf_to_gguf("model.layers.3.self_attn.q_proj") == "blk.3.attn_q.weight"
    assert map_hf_to_gguf("model.layers.10.self_attn.o_proj") == "blk.10.attn_output.weight"
    assert map_hf_to_gguf("model.layers.0.self_attn.out_proj") == "blk.0.attn_output.weight"


def test_map_hf_to_gguf_mlp_and_head():
    assert map_hf_to_gguf("model.layers.7.mlp.gate_proj") == "blk.7.ffn_gate.weight"
    assert map_hf_to_gguf("model.layers.7.mlp.up_proj") == "blk.7.ffn_up.weight"
    assert map_hf_to_gguf("model.layers.7.mlp.down_proj") == "blk.7.ffn_down.weight"
    assert map_hf_to_gguf("lm_head") == "output.weight"


def test_map_hf_to_gguf_unmapped_returns_none():
    assert map_hf_to_gguf("model.embed_tokens") is None
    assert map_hf_to_gguf("model.layers.0.self_attn.rotary_emb.inv_freq") is None


def test_is_ssm():
    assert is_ssm("blk.3.ssm_dt.weight")
    assert not is_ssm("blk.3.attn_q.weight")
    assert not is_ssm("blk.3.ffn_down.weight")

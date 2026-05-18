"""Unit tests for imatrix calibrator helpers (no model load required)."""

import numpy as np

from quant_tuner.calibrate import imatrix as imx
from quant_tuner.calibrate.imatrix import (
    ForwardStats,
    _col_l2_sq,
    _l1_normalize,
    build_analytic,
    build_outlier_l4,
)
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


def test_forward_stats_roundtrip(tmp_path):
    stats = ForwardStats(
        e_a4={
            "blk.0.attn_q.weight": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "blk.0.ffn_down.weight": np.array([1.0, 2.0], dtype=np.float32),
        },
        max_abs={
            "blk.0.attn_q.weight": np.array([1.5, 2.5, 0.5], dtype=np.float32),
            "blk.0.ffn_down.weight": np.array([4.0, 5.0], dtype=np.float32),
        },
    )
    out = tmp_path / "stats.npz"
    stats.save(out)
    loaded = ForwardStats.load(out)
    assert set(loaded.e_a4) == set(stats.e_a4)
    for k in stats.e_a4:
        np.testing.assert_allclose(loaded.e_a4[k], stats.e_a4[k])
        np.testing.assert_allclose(loaded.max_abs[k], stats.max_abs[k])


def test_build_analytic_respects_ssm_passthrough(monkeypatch):
    # Two linear-projection tensors and one SSM tensor.
    base = {
        "blk.0.attn_q.weight": np.array([1.0, 4.0, 9.0], dtype=np.float32),    # E[a^2]
        "blk.0.ffn_down.weight": np.array([2.0, 2.0], dtype=np.float32),
        "blk.0.ssm_dt.weight": np.array([7.0, 7.0, 7.0], dtype=np.float32),
    }
    # W[:, c] columns chosen so ||W[:, c]||^2 is easy to verify.
    weights = {
        "blk.0.attn_q.weight": np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
        ),  # col sums-of-squares: [1, 1, 2]
        "blk.0.ffn_down.weight": np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        # col sums-of-squares: [9 + 16, 0] = [25, 0]
        "blk.0.ssm_dt.weight": np.zeros((2, 3), dtype=np.float32),
    }
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: base)
    monkeypatch.setattr(imx, "_load_weights", lambda _p: weights)

    out = build_analytic("dummy_f16", "dummy_base")

    # attn_q: [1*1, 1*4, 2*9] = [1, 4, 18]
    np.testing.assert_allclose(out["blk.0.attn_q.weight"], [1.0, 4.0, 18.0])
    # ffn_down: [25*2, 0*2] = [50, 0]
    np.testing.assert_allclose(out["blk.0.ffn_down.weight"], [50.0, 0.0])
    # SSM tensor MUST pass through raw E[a^2], not be reranked.
    np.testing.assert_allclose(out["blk.0.ssm_dt.weight"], [7.0, 7.0, 7.0])


def test_build_outlier_l4_falls_back_when_stats_missing(monkeypatch):
    """If forward stats lack a tensor, that tensor falls back to E[a^2]."""
    base = {
        "blk.0.attn_q.weight": np.array([0.0, 16.0, 81.0], dtype=np.float32),
        "blk.0.attn_k.weight": np.array([0.5, 0.5, 0.5], dtype=np.float32),  # no fwd stats
        "blk.0.ssm_dt.weight": np.array([7.0, 7.0], dtype=np.float32),
    }
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: base)

    stats = ForwardStats(
        e_a4={"blk.0.attn_q.weight": np.array([1.0, 16.0, 81.0], dtype=np.float32)},
        max_abs={"blk.0.attn_q.weight": np.array([2.0, 4.0, 9.0], dtype=np.float32)},
    )
    out = build_outlier_l4("dummy_base", stats)

    np.testing.assert_allclose(out["blk.0.attn_q.weight"], [1.0, 4.0, 9.0])
    np.testing.assert_allclose(out["blk.0.attn_k.weight"], [0.5, 0.5, 0.5])  # E[a^2] fallback
    np.testing.assert_allclose(out["blk.0.ssm_dt.weight"], [7.0, 7.0])  # SSM passthrough

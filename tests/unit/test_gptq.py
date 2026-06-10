"""Unit tests for GPTQ pure-tensor pieces (no HF model needed)."""

from __future__ import annotations

import pytest
import torch

from quant_tuner.calibrate.gptq import (
    Hessian,
    _parse_perplexity,
    gptq_round_tensor,
    grid_for_quant_type,
)


def _random_psd(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A.T @ A + 0.5 * torch.eye(n)


def test_gptq_round_shape_mismatch_raises():
    W = torch.randn(4, 8)
    H = torch.eye(7)
    with pytest.raises(ValueError, match="H shape"):
        gptq_round_tensor(W, H)


def test_gptq_round_int4_quantization_levels():
    """With INT4 (qmax=7), each per-row-per-group value lands on one of 15 levels."""
    W = torch.randn(4, 64) * 0.5
    Q, _ = gptq_round_tensor(
        W, _random_psd(64), n_bits=4, group_size=32, actorder=False
    )
    assert Q.shape == W.shape
    # Number of distinct values per row should be modest — bounded by groups × (2*qmax+1)
    for row in Q:
        # 64 / 32 = 2 groups, each ≤ 15 levels → ≤ 30 unique values per row.
        assert len(torch.unique(row)) <= 30


def test_gptq_round_dead_columns_zeroed():
    """A column with zero H-diagonal must come out exactly zero."""
    n = 32
    W = torch.randn(4, n) * 0.5
    H = _random_psd(n, seed=1)
    H[5, :] = 0.0
    H[:, 5] = 0.0  # column 5 is "dead"
    Q, stats = gptq_round_tensor(W, H, group_size=8, actorder=False)
    assert stats.dead_cols >= 1
    assert torch.allclose(Q[:, 5], torch.zeros(4))


def test_gptq_round_high_bits_low_error():
    """At 8 bits, the rounded output should track W extremely closely."""
    n = 32
    W = torch.randn(4, n) * 0.3
    Q, stats = gptq_round_tensor(
        W, _random_psd(n, seed=2), n_bits=8, group_size=8, actorder=False
    )
    rel = ((Q - W).norm() / W.norm()).item()
    assert rel < 0.05, f"INT8 GPTQ relative error too high: {rel}"
    # The stats-reported frob_rel must agree with the directly-computed one
    # to within float noise (i.e. the in-function metric is honest).
    assert abs(stats.frob_rel - rel) < 1e-5


def test_gptq_round_int2_has_high_error():
    """At 2 bits, even GPTQ can't avoid substantial error — verifies frob_rel scales."""
    n = 32
    W = torch.randn(4, n) * 0.3
    _, stats_int8 = gptq_round_tensor(
        W, _random_psd(n, seed=6), n_bits=8, group_size=8, actorder=False
    )
    _, stats_int2 = gptq_round_tensor(
        W, _random_psd(n, seed=6), n_bits=2, group_size=8, actorder=False
    )
    assert stats_int2.frob_rel > 5 * stats_int8.frob_rel


def test_gptq_round_actorder_zero_diag_handled():
    """Act-order permutation must not break when some H diag entries are zero."""
    n = 16
    W = torch.randn(2, n) * 0.5
    H = _random_psd(n, seed=3)
    H[0, :] = 0.0
    H[:, 0] = 0.0  # dead column 0 has lowest priority under act-order
    Q, stats = gptq_round_tensor(W, H, group_size=4, actorder=True)
    assert stats.dead_cols >= 1
    assert torch.allclose(Q[:, 0], torch.zeros(2))


def test_gptq_round_actorder_and_no_actorder_agree_on_dead_cols():
    """The set of dead columns must be permutation-invariant."""
    n = 16
    W = torch.randn(2, n) * 0.5
    H = _random_psd(n, seed=4)
    for i in (3, 7, 11):
        H[i, :] = 0.0
        H[:, i] = 0.0
    _, stats_a = gptq_round_tensor(W, H, group_size=4, actorder=True)
    _, stats_b = gptq_round_tensor(W, H, group_size=4, actorder=False)
    assert stats_a.dead_cols == stats_b.dead_cols == 3


# ----- low-bit: asymmetric (min+scale) grid --------------------------------- #


def test_gptq_asym_2bit_exact_on_grid():
    """Weights already on an asymmetric 2-bit grid reconstruct exactly."""
    # Grid: min=-0.4, scale=0.2 -> levels {-0.4, -0.2, 0.0, 0.2}.
    levels = torch.tensor([-0.4, -0.2, 0.0, 0.2])
    g = torch.Generator().manual_seed(0)
    W = levels[torch.randint(0, 4, (4, 16), generator=g)]
    W[:, 0] = -0.4  # pin both grid extremes so wmin/wmax recover the grid
    W[:, 1] = 0.2
    Q, stats = gptq_round_tensor(
        W, torch.eye(16), n_bits=2, group_size=16, sym=False, actorder=False
    )
    torch.testing.assert_close(Q, W)
    assert stats.frob_rel < 1e-6


def test_gptq_asym_2bit_beats_sym_2bit():
    """At 2 bits, a symmetric grid has 3 usable levels pinned to the group max;
    the asymmetric grid uses all 4 and must reconstruct gaussian W better."""
    torch.manual_seed(1)
    W = torch.randn(8, 64) * 0.3
    H = _random_psd(64, seed=1)
    _, stats_sym = gptq_round_tensor(
        W, H, n_bits=2, group_size=16, sym=True, actorder=False
    )
    _, stats_asym = gptq_round_tensor(
        W, H, n_bits=2, group_size=16, sym=False, actorder=False
    )
    assert stats_asym.frob_rel < stats_sym.frob_rel


def test_gptq_asym_uses_at_most_2pow_n_levels():
    torch.manual_seed(2)
    W = torch.randn(4, 32)
    Q, _ = gptq_round_tensor(
        W, torch.eye(32), n_bits=2, group_size=16, sym=False, actorder=False
    )
    for row in Q:
        for g0 in range(0, 32, 16):
            assert len(torch.unique(row[g0:g0 + 16])) <= 4


def test_gptq_sym_auto_selected_below_4_bits():
    """sym=None must behave as asymmetric at 2-3 bits and symmetric at 4+."""
    torch.manual_seed(3)
    W = torch.randn(4, 32) * 0.5
    H = _random_psd(32, seed=3)
    for n_bits, expected_sym in ((2, False), (3, False), (4, True), (8, True)):
        Q_auto, _ = gptq_round_tensor(
            W, H, n_bits=n_bits, group_size=16, actorder=False
        )
        Q_explicit, _ = gptq_round_tensor(
            W, H, n_bits=n_bits, group_size=16, sym=expected_sym, actorder=False
        )
        torch.testing.assert_close(Q_auto, Q_explicit)


def test_gptq_asym_dead_columns_still_zeroed():
    """Dead-channel zeroing must survive the asymmetric grid (0 representable)."""
    n = 32
    W = torch.randn(4, n) * 0.5
    H = _random_psd(n, seed=7)
    H[5, :] = 0.0
    H[:, 5] = 0.0
    Q, stats = gptq_round_tensor(
        W, H, n_bits=2, group_size=16, sym=False, actorder=False
    )
    assert stats.dead_cols >= 1
    assert torch.allclose(Q[:, 5], torch.zeros(4))


def test_grid_for_quant_type_mapping():
    assert grid_for_quant_type("Q2_K") == {"n_bits": 2, "group_size": 16, "sym": False}
    assert grid_for_quant_type("IQ2_M") == {"n_bits": 2, "group_size": 16, "sym": False}
    assert grid_for_quant_type("IQ1_S") == {"n_bits": 2, "group_size": 16, "sym": False}
    assert grid_for_quant_type("iq3_s") == {"n_bits": 3, "group_size": 16, "sym": False}
    assert grid_for_quant_type("Q3_K_M") == {"n_bits": 3, "group_size": 16, "sym": False}
    assert grid_for_quant_type("Q4_K_M") == {"n_bits": 4, "group_size": 32, "sym": True}
    assert grid_for_quant_type("Q5_K_S") == {"n_bits": 4, "group_size": 32, "sym": True}


def test_hessian_save_load_roundtrip(tmp_path):
    n = 16
    h = Hessian(name="model.layers.0.self_attn.q_proj", H=_random_psd(n, seed=5), n_seen=4096)
    path = tmp_path / "q.pt"
    h.save(path)
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()  # tmp must be renamed away
    loaded = Hessian.load(path)
    assert loaded.name == h.name
    assert loaded.n_seen == h.n_seen
    torch.testing.assert_close(loaded.H, h.H)


def test_hessian_save_is_atomic_via_tmp_rename(tmp_path):
    """A pre-existing file at the target path is replaced; tmp is gone afterward."""
    target = tmp_path / "q.pt"
    target.write_bytes(b"stale")
    Hessian(name="x", H=torch.eye(4), n_seen=1).save(target)
    loaded = Hessian.load(target)
    assert loaded.name == "x" and loaded.n_seen == 1
    assert not target.with_suffix(".pt.tmp").exists()


def test_parse_perplexity_picks_final_line():
    text = (
        "perplexity: tokenizing the input ..\n"
        "perplexity: [1]9.5, [2]9.4, ...\n"
        "perplexity: 9.20 seconds per pass\n"
        "Final estimate: PPL = 9.5421 +/- 0.05123\n"
    )
    assert _parse_perplexity(text) == 9.5421


def test_parse_perplexity_returns_none_when_missing():
    assert _parse_perplexity("no perplexity here") is None

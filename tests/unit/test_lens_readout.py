"""Readout math tests on synthetic weights (no model load required)."""

from __future__ import annotations

import numpy as np

from quant_tuner.lens.lens import JacobianLensGGUF
from quant_tuner.lens.model_reader import ReadoutWeights
from quant_tuner.lens.readout import LensReadout


def _weights(d=8, V=20, seed=0, softcap=None):
    rng = np.random.default_rng(seed)
    return ReadoutWeights(
        arch="test", n_layers=3, d_model=d, n_vocab=V,
        w_unembed=rng.standard_normal((V, d)).astype(np.float32),
        norm_weight=np.ones(d, dtype=np.float32), norm_bias=None,
        norm_type="rms", eps=1e-5, logit_softcap=softcap,
    )


def test_identity_lens_equals_logit_lens():
    w = _weights()
    lens = JacobianLensGGUF.identity(d_model=w.d_model, layers=[0, 1])
    ro = LensReadout(w, lens)
    h = np.random.default_rng(1).standard_normal((4, w.d_model)).astype(np.float32)
    np.testing.assert_allclose(ro.lens_logits(h, 0, use_lens=True), w.unembed(h), atol=1e-4)


def test_use_lens_false_is_logit_lens_even_with_fitted_J():
    w = _weights()
    rng = np.random.default_rng(2)
    lens = JacobianLensGGUF({0: rng.standard_normal((w.d_model, w.d_model)).astype(np.float32)},
                            d_model=w.d_model, target_layer=2, fit_method="regression")
    ro = LensReadout(w, lens)
    h = rng.standard_normal((3, w.d_model)).astype(np.float32)
    np.testing.assert_allclose(ro.lens_logits(h, 0, use_lens=False), w.unembed(h), atol=1e-4)


def test_softcap_bounds_logits():
    w = _weights(softcap=5.0)
    lens = JacobianLensGGUF.identity(d_model=w.d_model, layers=[0])
    ro = LensReadout(w, lens)
    h = np.random.default_rng(3).standard_normal((4, w.d_model)).astype(np.float32) * 100
    logits = ro.lens_logits(h, 0)
    assert np.all(np.abs(logits) <= 5.0 + 1e-4)


def test_lens_vector_raises_target_logit():
    w = _weights()
    rng = np.random.default_rng(4)
    lens = JacobianLensGGUF({0: rng.standard_normal((w.d_model, w.d_model)).astype(np.float32)},
                            d_model=w.d_model, target_layer=2, fit_method="regression")
    ro = LensReadout(w, lens)
    t = 3
    v = ro.lens_vector(0, t, unit=True)
    h0 = rng.standard_normal(w.d_model).astype(np.float32)
    base = (lens.transport(h0[None, :], 0) @ w.w_unembed.T)[0, t]
    bumped = (lens.transport((h0 + 0.2 * v)[None, :], 0) @ w.w_unembed.T)[0, t]
    assert bumped > base


def test_topk_descending():
    logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
    ids, vals = LensReadout.topk(logits, 3)
    assert list(ids[0]) == [1, 4, 2]
    np.testing.assert_allclose(vals[0], [5.0, 4.0, 3.0])


def test_ranks_of_matches_bruteforce():
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((6, 40)).astype(np.float32)
    ids = np.array([0, 10, 39], dtype=np.int64)
    ranks = LensReadout.ranks_of(logits, ids)
    for r in range(logits.shape[0]):
        for j, tid in enumerate(ids):
            expected = int((logits[r] > logits[r, tid]).sum())
            assert ranks[r, j] == expected


def test_ablate_factors_remove_projection():
    w = _weights()
    rng = np.random.default_rng(6)
    lens = JacobianLensGGUF({0: rng.standard_normal((w.d_model, w.d_model)).astype(np.float32)},
                            d_model=w.d_model, target_layer=2, fit_method="regression")
    ro = LensReadout(w, lens)
    A, B = ro.ablate_factors(0, [3])
    # applying h += A(Bh) should zero the component along the token's direction
    v = ro.lens_vector(0, 3, unit=True)
    h = rng.standard_normal(w.d_model).astype(np.float32)
    h_edited = h + A @ (B @ h)
    assert abs(float(v @ h_edited)) < abs(float(v @ h)) + 1e-6
    assert abs(float(v @ h_edited)) < 1e-4

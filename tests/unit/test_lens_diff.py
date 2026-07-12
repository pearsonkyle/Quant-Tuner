"""Diff-engine tests on synthetic capture runs (no model load required)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tuner.lens.capture import (
    ACTIVATIONS_NAME,
    FINAL_LOGPROBS_NAME,
    MANIFEST_NAME,
    Run,
    RunManifest,
    _log_softmax,
)
from quant_tuner.lens.diff import diff_runs, validate_pair
from quant_tuner.lens.lens import JacobianLensGGUF
from quant_tuner.lens.model_reader import ReadoutWeights
from quant_tuner.lens.readout import LensReadout

D, V, LAYERS, TOKENS, POS = 8, 20, [0, 1, 2], [1, 2, 3, 4], [0, 1, 2, 3]


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    return ReadoutWeights(
        arch="t", n_layers=3, d_model=D, n_vocab=V,
        w_unembed=rng.standard_normal((V, D)).astype(np.float32),
        norm_weight=np.ones(D, dtype=np.float32), norm_bias=None,
        norm_type="rms", eps=1e-5, logit_softcap=None,
    )


def _make_run(dirp, acts, model="m.gguf", logits_pos=None, logits=None):
    dirp.mkdir(parents=True, exist_ok=True)
    m = RunManifest(
        run_id=dirp.name, model_path=model, model_fingerprint="x",
        tokens=TOKENS, n_prompt=4, n_gen=0, capture_layers=LAYERS, positions=POS,
        logits_positions=logits_pos or [], interventions=[], n_predict=0,
        sampling={}, llama_commit="c", h_rms_observed={str(l): 3.0 for l in LAYERS},
    )
    np.savez(dirp / ACTIVATIONS_NAME, **{f"layer_{l}": acts[l].astype(np.float16) for l in LAYERS})
    if logits is not None:
        np.savez(dirp / FINAL_LOGPROBS_NAME, positions=np.asarray(logits_pos), logprobs=logits)
    (dirp / MANIFEST_NAME).write_text(m.to_json())
    return Run(dirp)


def test_identical_runs_fully_agree(tmp_path):
    rng = np.random.default_rng(1)
    acts = {l: rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    w = _weights()
    ro = LensReadout(w, JacobianLensGGUF.identity(d_model=D, layers=[0, 1]))
    a = _make_run(tmp_path / "a", acts)
    b = _make_run(tmp_path / "b", acts)
    d = diff_runs(a, b, readout_run=ro, readout_ref=ro)
    assert d.agree.all()
    np.testing.assert_allclose(d.layer_rel_l2, 0, atol=1e-3)
    np.testing.assert_allclose(d.final_kld, 0, atol=1e-4)
    assert d.summary()["lens_agree_mean"] == 1.0


def test_exact_kld_from_stored_logits(tmp_path):
    rng = np.random.default_rng(2)
    actsA = {l: rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    actsB = {l: actsA[l] + 0.5 * rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    w = _weights()
    ro = LensReadout(w, JacobianLensGGUF.identity(d_model=D, layers=[0, 1]))
    lpA = _log_softmax(ro.lens_logits(actsA[2], 2))
    lpB = _log_softmax(ro.lens_logits(actsB[2], 2))
    ref = _make_run(tmp_path / "ref", actsA, logits_pos=[3], logits=lpA[3:4])
    cand = _make_run(tmp_path / "cand", actsB, logits_pos=[3], logits=lpB[3:4])
    d = diff_runs(cand, ref, readout_run=ro, readout_ref=ro)
    expected = float((np.exp(lpA[3]) * (lpA[3] - lpB[3])).sum())
    assert d.final_kld_exact[-1]
    assert abs(d.final_kld[-1] - expected) < 1e-4


def test_rank_shift_known(tmp_path):
    # craft candidate where the reference top-1 falls to a known rank
    w = _weights(seed=3)
    ro = LensReadout(w, JacobianLensGGUF.identity(d_model=D, layers=[0, 1]))
    rng = np.random.default_rng(4)
    acts = {l: rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    ref = _make_run(tmp_path / "r", acts)
    cand = _make_run(tmp_path / "c", acts)
    d = diff_runs(cand, ref, readout_run=ro, readout_ref=ro)
    # identical acts => ref top-1 is candidate top-1 => rank 0 everywhere
    assert (d.ref_top1_rank_in_run == 0).all()


def test_validate_pair_token_mismatch(tmp_path):
    rng = np.random.default_rng(5)
    acts = {l: rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    a = _make_run(tmp_path / "a", acts)
    b = _make_run(tmp_path / "b", acts)
    b.manifest.tokens = [9, 9, 9, 9]
    with pytest.raises(ValueError, match="different token"):
        validate_pair(a, b)


def test_pinned_token_trajectories(tmp_path):
    rng = np.random.default_rng(6)
    acts = {l: rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    w = _weights()
    ro = LensReadout(w, JacobianLensGGUF.identity(d_model=D, layers=[0, 1]))
    a = _make_run(tmp_path / "a", acts)
    b = _make_run(tmp_path / "b", acts)
    d = diff_runs(a, b, readout_run=ro, readout_ref=ro, pinned_tokens=[5, 7])
    assert set(d.pinned) == {5, 7}
    assert d.pinned[5]["rank_run"].shape == (len(LAYERS), len(POS))
    # identical runs => identical rank trajectories
    np.testing.assert_array_equal(d.pinned[5]["rank_run"], d.pinned[5]["rank_ref"])


def test_diff_save_load(tmp_path):
    rng = np.random.default_rng(7)
    acts = {l: rng.standard_normal((len(POS), D)).astype(np.float32) for l in LAYERS}
    w = _weights()
    ro = LensReadout(w, JacobianLensGGUF.identity(d_model=D, layers=[0, 1]))
    a = _make_run(tmp_path / "a", acts)
    b = _make_run(tmp_path / "b", {l: acts[l] + 0.1 for l in LAYERS})
    d = diff_runs(a, b, readout_run=ro, readout_ref=ro, pinned_tokens=[3])
    out = d.save(tmp_path / "diff")
    from quant_tuner.lens.diff import RunDiff

    d2 = RunDiff.load(out)
    np.testing.assert_array_equal(d.agree, d2.agree)
    np.testing.assert_allclose(d.final_kld, d2.final_kld, atol=1e-6)
    assert set(d2.pinned) == {3}

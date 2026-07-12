"""Lens GGUF + JLNS roundtrip tests (no model load required)."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from quant_tuner.lens.client import NativeClient
from quant_tuner.lens.lens import JacobianLensGGUF


def test_lens_gguf_roundtrip(tmp_path):
    d = 8
    rng = np.random.default_rng(0)
    jac = {0: rng.standard_normal((d, d)).astype(np.float32),
           2: rng.standard_normal((d, d)).astype(np.float32)}
    bias = {0: rng.standard_normal(d).astype(np.float32)}
    lens = JacobianLensGGUF(
        jac, d_model=d, n_prompts=7, target_layer=3, fit_method="regression",
        base_model="tiny.gguf", biases=bias, h_rms={0: 3.0, 2: 5.0},
        extra_meta={"corpus_sha256": "abc123", "model_sha256": "def456"},
    )
    p = tmp_path / "lens.gguf"
    lens.save(str(p))
    loaded = JacobianLensGGUF.load(str(p))

    assert loaded.d_model == d
    assert loaded.source_layers == [0, 2]
    assert loaded.fit_method == "regression"
    assert loaded.n_prompts == 7
    assert loaded.target_layer == 3
    np.testing.assert_allclose(loaded.jacobians[0], jac[0], atol=1e-3)
    np.testing.assert_allclose(loaded.biases[0], bias[0], atol=1e-4)
    assert loaded.h_rms == {0: 3.0, 2: 5.0}
    assert loaded.extra_meta == {"corpus_sha256": "abc123", "model_sha256": "def456"}


def test_identity_lens_is_logit_lens(tmp_path):
    lens = JacobianLensGGUF.identity(d_model=6, layers=[0, 1, 2])
    assert lens.fit_method == "identity"
    for layer in (0, 1, 2):
        np.testing.assert_allclose(lens.jacobians[layer], np.eye(6), atol=0)


def test_merge_weights_by_n_prompts():
    d = 4
    a = JacobianLensGGUF({0: np.ones((d, d), np.float32)}, d_model=d, n_prompts=1)
    b = JacobianLensGGUF({0: np.full((d, d), 3.0, np.float32)}, d_model=d, n_prompts=3)
    merged = JacobianLensGGUF.merge([a, b])
    # weighted mean: (1*1 + 3*3) / 4 = 2.5
    np.testing.assert_allclose(merged.jacobians[0], np.full((d, d), 2.5), atol=1e-6)
    assert merged.n_prompts == 4


def _make_jlns(activations: dict[int, np.ndarray], tokens: list[int]) -> bytes:
    """Hand-build a JLNS binary payload for parse testing."""
    payload = b""
    acts_meta = []
    for layer, arr in activations.items():
        arr16 = arr.astype(np.float16)
        offset = len(payload)
        payload += arr16.tobytes()
        acts_meta.append({
            "layer": layer, "dtype": "f16", "shape": list(arr.shape),
            "offset": offset, "nbytes": arr16.nbytes,
        })
    header = {
        "tokens": tokens, "n_prompt": len(tokens), "n_gen": 0,
        "generated": [], "activations": acts_meta, "logits": [], "timings": {},
    }
    hdr = json.dumps(header).encode()
    return b"JLNS" + struct.pack("<II", 1, len(hdr)) + hdr + payload


def test_jlns_parse_roundtrip():
    d = 5
    acts = {0: np.arange(3 * d, dtype=np.float32).reshape(3, d),
            1: np.ones((3, d), dtype=np.float32)}
    buf = _make_jlns(acts, [10, 11, 12])
    result = NativeClient._parse_forward(buf)
    assert result.tokens == [10, 11, 12]
    assert set(result.activations) == {0, 1}
    np.testing.assert_allclose(result.activations[0], acts[0], atol=1e-2)
    np.testing.assert_allclose(result.activations[1], acts[1], atol=1e-3)


def test_jlns_bad_magic_raises():
    with pytest.raises(RuntimeError, match="magic"):
        NativeClient._parse_forward(b"XXXX____")


def test_intervention_encoding_lowrank_shapes():
    d, k = 6, 2
    a = np.zeros((d, k), np.float32)
    b = np.zeros((k, d), np.float32)
    enc = NativeClient._encode_iv({"layer": 3, "mode": "lowrank", "a": a, "b": b})
    assert enc["mode"] == "lowrank"
    assert enc["k"] == k
    with pytest.raises(ValueError, match="lowrank"):
        NativeClient._encode_iv({"layer": 0, "mode": "lowrank",
                                 "a": np.zeros((d, k)), "b": np.zeros((k + 1, d))})

"""Integration: build jlens-server, verify capture + numpy-vs-server logit parity.

The parity check (numpy readout vs the server's own logits, corr >= 0.9999) is
the drift canary: if a llama.cpp submodule bump changes the residual (l_out)
semantics or the readout weights, this fails loudly.

Gated on QT_LENS_IT=1 + QT_TINY_GGUF (see conftest).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from quant_tuner.paths import JLENS_SERVER_DIR


@pytest.mark.integration
def test_build():
    """native/jlens_server/build.sh succeeds against the vendored llama.cpp."""
    if os.environ.get("QT_LENS_IT") != "1":
        pytest.skip("set QT_LENS_IT=1")
    rc = subprocess.call(["bash", str(JLENS_SERVER_DIR / "build.sh")])
    assert rc == 0


@pytest.mark.integration
def test_props_l_out_ok(tiny_gguf, jlens_server_built):
    from quant_tuner.lens.backend import running_jlens_server

    with running_jlens_server(tiny_gguf, ctx=1024, ngl=0) as client:
        props = client.props()
        assert props["l_out_ok"] is True
        assert props["n_layer"] > 0
        assert "llama_commit" in props


@pytest.mark.integration
def test_numpy_readout_matches_server_logits(tiny_gguf, jlens_server_built):
    """numpy final-layer readout ≈ server logits (corr >= 0.9999)."""
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    with running_jlens_server(tiny_gguf, ctx=1024, ngl=0) as client:
        tokens = client.tokenize("The capital of France is", parse_special=False)
        n_layer = client.props()["n_layer"]
        fr = client.forward(tokens, capture_layers=[n_layer - 1],
                            logits_positions=[-1], dtype="f32")
        weights = ReadoutWeights.from_gguf(str(tiny_gguf))
        lens = JacobianLensGGUF.identity(d_model=weights.d_model, layers=[0])
        ro = LensReadout(weights, lens)
        h_final = fr.activations[n_layer - 1][-1][None, :]
        numpy_logits = ro.lens_logits(h_final, n_layer - 1, use_lens=False)[0]
        server_logits = fr.logits[max(fr.logits)]
        corr = float(np.corrcoef(numpy_logits, server_logits)[0, 1])
        assert corr >= 0.9999, f"logit correlation too low: {corr}"


@pytest.mark.integration
def test_capture_and_diff_identity(tiny_gguf, jlens_server_built, tmp_path):
    """Capture the same prompt twice; the diff must fully agree / zero KLD."""
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.capture import capture_run, load_run
    from quant_tuner.lens.diff import diff_runs
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    runs = tmp_path / "runs"
    with running_jlens_server(tiny_gguf, ctx=1024, ngl=0) as client:
        tokens = client.tokenize("Hello world", parse_special=False)
        a = capture_run(client, tokens, runs_dir=runs, logits_positions=[-1], tags={"r": "a"})
        b = capture_run(client, tokens, runs_dir=runs, logits_positions=[-1],
                        tags={"r": "b"}, overwrite=True)
    weights = ReadoutWeights.from_gguf(str(tiny_gguf))
    ro = LensReadout(weights, JacobianLensGGUF.identity(d_model=weights.d_model, layers=[0]))
    d = diff_runs(load_run(a), load_run(b), readout_run=ro, readout_ref=ro)
    assert d.agree.all()
    assert float(np.abs(d.final_kld).max()) < 1e-4

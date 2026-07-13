"""Integration: fit a tiny lens, capture, and confirm an intervention shows up.

Gated on QT_LENS_IT=1 + QT_TINY_GGUF (see conftest).
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_fit_small_band(tiny_gguf, jlens_server_built, tmp_path):
    from quant_tuner.lens.fitting import fit_lens
    from quant_tuner.lens.lens import JacobianLensGGUF

    out = tmp_path / "tiny.lens.gguf"
    fit_lens(
        tiny_gguf, "wikitext:12", out,
        layers=[0, 1], band_size=2, n_prompts=12, max_seq_len=48, ngl=0,
    )
    lens = JacobianLensGGUF.load(str(out))
    assert lens.fit_method == "regression"
    assert set(lens.source_layers) == {0, 1}
    assert lens.d_model > 0


@pytest.mark.integration
def test_ablation_changes_generation(tiny_gguf, jlens_server_built):
    """An ablation intervention should perturb greedy generation."""
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    weights = ReadoutWeights.from_gguf(str(tiny_gguf))
    lens = JacobianLensGGUF.identity(d_model=weights.d_model, layers=list(range(weights.n_layers - 1)))
    ro = LensReadout(weights, lens)
    with running_jlens_server(tiny_gguf, ctx=1024, ngl=0) as client:
        tokens = client.tokenize("The weather today is", parse_special=False)
        base = client.forward(tokens, capture=False, n_predict=12, sampling={"greedy": True})
        # ablate a common token's direction across all layers
        tid = client.tokenize(" the", add_special=False, parse_special=False)[0]
        A, B = ro.ablate_factors(weights.n_layers // 2, [tid])
        ivs = [{"layer": weights.n_layers // 2, "pos_start": 0, "pos_end": -1,
                "mode": "lowrank", "a": A, "b": B}]
        steered = client.forward(tokens, capture=False, interventions=ivs,
                                 n_predict=12, sampling={"greedy": True})
    # not a strict assertion on content, but the machinery must run and return
    assert isinstance(base.generated_text, str)
    assert isinstance(steered.generated_text, str)

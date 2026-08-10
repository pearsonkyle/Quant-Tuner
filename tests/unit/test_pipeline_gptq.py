"""Pipeline wiring tests for the GPTQ calibration branch (no model/llama.cpp needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_tuner import pipeline
from quant_tuner.config import CalibrationConfig, QuantizeConfig, RunConfig


def _make_cfg(tmp_path: Path, params: dict, quant_type: str = "Q4_K_M") -> RunConfig:
    return RunConfig(
        name="t",
        model="org/repo",
        workspace=tmp_path / "ws",
        calibration=CalibrationConfig(method="gptq", variant="default", params=params),
        quantize=QuantizeConfig(type=quant_type),
    )


def _stub_gptq_steps(monkeypatch, calls: dict) -> None:
    def fake_calibrate(model_dir, corpus, hessians_dir, **kw):
        calls["calibrate"] = kw
        hessians_dir.mkdir(parents=True, exist_ok=True)
        return None  # pipeline lambda touches the _done sentinel

    def fake_apply(model_dir, hessians_dir, out_dir, **kw):
        calls["apply"] = kw
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").touch()
        return out_dir

    def fake_verify(f16, eval_ds, *, reference_ppl=None, max_ratio=1.5,
                    ctx=8192, log=None):
        calls.setdefault("verify", []).append(
            {"model": Path(f16).name, "reference_ppl": reference_ppl,
             "max_ratio": max_ratio}
        )
        return 10.0

    def fake_convert(src, dst, log=None):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.touch()
        return dst

    def fake_imatrix(model, corpus, out, ctx=512, log=None, **kw):
        calls.setdefault("imatrix", []).append({"model": Path(model).name})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    def fake_variant(**kw):
        calls["imatrix_variant"] = kw
        Path(kw["out_path"]).touch()
        return kw["out_path"]

    monkeypatch.setattr(pipeline.gptq, "calibrate", fake_calibrate)
    monkeypatch.setattr(pipeline.gptq, "apply", fake_apply)
    monkeypatch.setattr(pipeline.gptq, "verify_perplexity", fake_verify)
    monkeypatch.setattr(pipeline.convert, "hf_to_f16_gguf", fake_convert)
    monkeypatch.setattr(pipeline.llama_cpp, "imatrix", fake_imatrix)
    monkeypatch.setattr(pipeline.imatrix, "calibrate", fake_variant)


def _run(cfg, monkeypatch) -> tuple[dict, dict]:
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    train = ws.corpus_dir / "train.txt"
    train.write_text("x")
    eval_ = ws.corpus_dir / "eval.txt"
    eval_.write_text("y")

    calls: dict = {}
    _stub_gptq_steps(monkeypatch, calls)
    artifacts = pipeline.calibrate(cfg, ws, f16, train, eval_)
    return calls, artifacts


def test_gptq_params_routed_to_apply_not_calibrate(tmp_path, monkeypatch):
    """Rounding params (n_bits etc.) must reach gptq.apply, not gptq.calibrate
    (which would raise TypeError)."""
    cfg = _make_cfg(tmp_path, {
        "tokens": 1024, "ctx": 512,
        "n_bits": 4, "group_size": 32, "dampen": 0.02, "actorder": False,
        "sanity_max_rel": 0.4,
    })
    calls, _ = _run(cfg, monkeypatch)

    assert calls["calibrate"] == {"tokens": 1024, "ctx": 512}
    assert calls["apply"] == {
        "n_bits": 4, "group_size": 32, "dampen": 0.02, "actorder": False,
        "sanity_max_rel": 0.4, "sym": True,  # grid default for Q4_K_M
    }


def test_gptq_mac_knobs_routed_to_calibrate(tmp_path, monkeypatch):
    """hessian_device / layers_per_pass are calibrate-stage params and must
    not leak into gptq.apply."""
    cfg = _make_cfg(tmp_path, {
        "tokens": 1024, "hessian_device": "mps", "layers_per_pass": 4,
    })
    calls, _ = _run(cfg, monkeypatch)

    assert calls["calibrate"]["hessian_device"] == "mps"
    assert calls["calibrate"]["layers_per_pass"] == 4
    assert "hessian_device" not in calls["apply"]
    assert "layers_per_pass" not in calls["apply"]


def test_gptq_grid_auto_derived_from_quant_type(tmp_path, monkeypatch):
    """With no explicit grid params, Q2_K must get the 2-bit asym g16 grid and
    the relaxed 2-bit guardrails."""
    cfg = _make_cfg(tmp_path, {"tokens": 1024}, quant_type="Q2_K")
    calls, _ = _run(cfg, monkeypatch)

    assert calls["apply"]["n_bits"] == 2
    assert calls["apply"]["group_size"] == 16
    assert calls["apply"]["sym"] is False
    assert calls["apply"]["sanity_max_rel"] == 1.0
    # verify_perplexity: first call measures the reference, second checks the
    # rounded model against it with the bits-relaxed ratio.
    assert calls["verify"][0]["reference_ppl"] is None
    assert calls["verify"][1]["reference_ppl"] == 10.0
    assert calls["verify"][1]["max_ratio"] == 4.0


def test_gptq_explicit_params_override_auto_grid(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path,
        {"n_bits": 3, "sym": True, "ppl_max_ratio": 9.0},
        quant_type="Q2_K",
    )
    calls, _ = _run(cfg, monkeypatch)

    assert calls["apply"]["n_bits"] == 3
    assert calls["apply"]["sym"] is True
    assert calls["apply"]["group_size"] == 16          # still from the grid
    assert calls["apply"]["sanity_max_rel"] == 0.75    # relaxed for n_bits=3
    assert calls["verify"][1]["max_ratio"] == 9.0


def test_gptq_grid_mix_defaults_to_quant_type(tmp_path, monkeypatch):
    """With no pinned grid, apply gets grid_mix=<quantize.type> so mixed
    ftypes round each tensor on its real target's grid."""
    for qt in ("Q2_K", "IQ2_M", "IQ3_M", "IQ4_XS", "Q4_K_M"):
        cfg = _make_cfg(tmp_path / qt.lower(), {"tokens": 1024}, quant_type=qt)
        calls, _ = _run(cfg, monkeypatch)
        assert calls["apply"]["grid_mix"] == qt


def test_gptq_pinned_grid_suppresses_grid_mix(tmp_path, monkeypatch):
    """Pinning any base-grid param opts out of the mix (mirror of AWQ's
    pinned-proxy semantics); an explicit grid_mix stacks with it."""
    cfg = _make_cfg(tmp_path / "a", {"n_bits": 2}, quant_type="Q2_K")
    calls, _ = _run(cfg, monkeypatch)
    assert "grid_mix" not in calls["apply"]

    cfg = _make_cfg(
        tmp_path / "b", {"n_bits": 2, "grid_mix": "Q2_K"}, quant_type="Q2_K")
    calls, _ = _run(cfg, monkeypatch)
    assert calls["apply"]["grid_mix"] == "Q2_K"


def test_gptq_imatrix_collected_on_rounded_f16(tmp_path, monkeypatch):
    """The imatrix must be collected on the GPTQ-rounded F16 (after the PPL
    guardrail), not the original — llama-quantize sees the rounded weights."""
    cfg = _make_cfg(tmp_path, {"tokens": 1024}, quant_type="Q2_K")
    calls, artifacts = _run(cfg, monkeypatch)

    assert calls["imatrix"] == [{"model": "model-f16-gptq.gguf"}]
    assert artifacts["imatrix"].name == "imatrix-gptq.gguf"
    assert artifacts["f16"].name == "model-f16-gptq.gguf"
    # guardrail ran before the imatrix collection existed in `calls` order
    assert len(calls["verify"]) == 2


def test_gptq_imatrix_variant_stacks_on_rounded_model(tmp_path, monkeypatch):
    """params.imatrix_variant re-weights the rounded-model imatrix, mirroring
    the AWQ stacking; the variant must not leak into gptq.calibrate/apply."""
    cfg = _make_cfg(
        tmp_path, {"tokens": 1024, "imatrix_variant": "hybrid_custom"},
        quant_type="Q2_K",
    )
    calls, artifacts = _run(cfg, monkeypatch)

    assert "imatrix_variant" not in calls["calibrate"]
    assert "imatrix_variant" not in calls["apply"]
    kw = calls["imatrix_variant"]
    assert kw["variant"] == "hybrid_custom"
    assert Path(kw["f16_gguf"]).name == "model-f16-gptq.gguf"
    assert Path(kw["base_imatrix"]).name == "imatrix-gptq.gguf"
    assert Path(kw["model_dir"]).name == "model_gptq"
    assert artifacts["imatrix"].name == "imatrix-gptq-hybrid_custom.gguf"


def test_quantize_filename_includes_imatrix_variant(tmp_path, monkeypatch):
    """Changing params.imatrix_variant must change the GGUF name, or a re-run
    in the same workspace would bench a stale file under the new label."""
    import quant_tuner.quantize.gguf as gguf_mod

    monkeypatch.setattr(
        gguf_mod, "quantize",
        lambda src, out, qtype, **kw: (out.touch(), out)[1],
    )
    cfg = RunConfig(
        name="t", model="m", workspace=tmp_path / "ws",
        calibration=CalibrationConfig(
            method="gptq", params={"imatrix_variant": "hybrid_custom"}),
        quantize=QuantizeConfig(type="Q2_K"),
    )
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    imx = ws.calibration_dir / "imatrix-gptq-hybrid_custom.gguf"
    imx.parent.mkdir(parents=True, exist_ok=True)
    imx.touch()
    out = pipeline.quantize_model(cfg, ws, f16, {"imatrix": imx, "f16": f16})
    assert out.name == "Q2_K-gptq-hybrid_custom.gguf"


def test_quantize_iq2_without_imatrix_rejected(tmp_path):
    cfg = RunConfig(
        name="t", model="m", workspace=tmp_path / "ws",
        calibration=CalibrationConfig(method="none"),
        quantize=QuantizeConfig(type="IQ2_M"),
    )
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    with pytest.raises(ValueError, match="importance matrix"):
        pipeline.quantize_model(cfg, ws, f16, {})


def test_quantize_q4_without_imatrix_allowed(tmp_path, monkeypatch):
    import quant_tuner.quantize.gguf as gguf_mod

    monkeypatch.setattr(
        gguf_mod, "quantize",
        lambda src, out, qtype, **kw: (out.touch(), out)[1],
    )
    cfg = RunConfig(
        name="t", model="m", workspace=tmp_path / "ws",
        calibration=CalibrationConfig(method="none"),
        quantize=QuantizeConfig(type="Q4_K_M"),
    )
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    out = pipeline.quantize_model(cfg, ws, f16, {})
    assert out.name == "Q4_K_M.gguf"

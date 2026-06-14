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
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    monkeypatch.setattr(pipeline.gptq, "calibrate", fake_calibrate)
    monkeypatch.setattr(pipeline.gptq, "apply", fake_apply)
    monkeypatch.setattr(pipeline.gptq, "verify_perplexity", fake_verify)
    monkeypatch.setattr(pipeline.convert, "hf_to_f16_gguf", fake_convert)
    monkeypatch.setattr(pipeline.llama_cpp, "imatrix", fake_imatrix)


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
        lambda src, out, qtype, imatrix=None, log=None: (out.touch(), out)[1],
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

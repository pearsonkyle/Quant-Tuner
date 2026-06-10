"""Pipeline wiring tests for the AWQ calibration branch (no model/llama.cpp needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_tuner import pipeline
from quant_tuner.calibrate._device import resolve_device
from quant_tuner.config import CalibrationConfig, RunConfig


def _make_cfg(tmp_path: Path, params: dict) -> RunConfig:
    return RunConfig(
        name="t",
        model="org/repo",
        workspace=tmp_path / "ws",
        calibration=CalibrationConfig(method="awq", variant="a050", params=params),
    )


def _stub_awq_steps(monkeypatch, calls: dict) -> None:
    def fake_calibrate(model_dir, corpus, out_path, **kw):
        calls["calibrate"] = kw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        return out_path

    def fake_apply(model_dir, bundle, out_dir, **kw):
        calls["apply"] = kw
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").touch()
        return out_dir

    def fake_convert(src, dst, log=None):
        calls.setdefault("convert", []).append((Path(src), Path(dst)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.touch()
        return dst

    def fake_imatrix(model, corpus, out, ctx=512, log=None, **kw):
        calls["imatrix_model"] = Path(model)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    def fake_variant_calibrate(**kw):
        calls["imatrix_variant"] = kw
        kw["out_path"].parent.mkdir(parents=True, exist_ok=True)
        kw["out_path"].touch()
        return kw["out_path"]

    monkeypatch.setattr(pipeline.awq, "calibrate", fake_calibrate)
    monkeypatch.setattr(pipeline.awq, "apply", fake_apply)
    monkeypatch.setattr(pipeline.convert, "hf_to_f16_gguf", fake_convert)
    monkeypatch.setattr(pipeline.llama_cpp, "imatrix", fake_imatrix)
    monkeypatch.setattr(pipeline.imatrix, "calibrate", fake_variant_calibrate)


def test_awq_params_routed_to_apply_not_calibrate(tmp_path, monkeypatch):
    """Fold-stage params (rmsnorm_plus_one etc.) must reach awq.apply and must
    NOT be forwarded to awq.calibrate (which would raise TypeError)."""
    cfg = _make_cfg(tmp_path, {
        "rmsnorm_plus_one": False,
        "sanity_max_rel": 0.05,
        "proxy_tokens": 128,
        "device": "cpu",
    })
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    train = ws.corpus_dir / "train.txt"
    train.write_text("x")

    calls: dict = {}
    _stub_awq_steps(monkeypatch, calls)
    artifacts = pipeline.calibrate(cfg, ws, f16, train, ws.corpus_dir / "eval.txt")

    assert calls["calibrate"] == {
        "force_alpha": 0.5,  # injected by variant=a050
        "proxy_tokens": 128,
        "device": "cpu",
    }
    assert calls["apply"] == {
        "rmsnorm_plus_one": False,
        "sanity_max_rel": 0.05,
        "device": "cpu",
    }
    assert artifacts["f16"] == ws.gguf_dir / "model-f16-awq.gguf"


def test_awq_imatrix_collected_on_folded_f16(tmp_path, monkeypatch):
    """The quantize-stage imatrix must be computed on the AWQ-folded F16, not
    the unfolded reference (whose per-channel E[a^2] is stale post-fold)."""
    cfg = _make_cfg(tmp_path, {})
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    train = ws.corpus_dir / "train.txt"
    train.write_text("x")

    calls: dict = {}
    _stub_awq_steps(monkeypatch, calls)
    artifacts = pipeline.calibrate(cfg, ws, f16, train, ws.corpus_dir / "eval.txt")

    assert calls["imatrix_model"] == ws.gguf_dir / "model-f16-awq.gguf"
    assert artifacts["imatrix"] == ws.calibration_dir / "imatrix-awq.gguf"
    assert "imatrix_variant" not in calls  # default: no re-weighting pass


def test_awq_imatrix_variant_stacks_on_folded_model(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, {"imatrix_variant": "hybrid_custom"})
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    train = ws.corpus_dir / "train.txt"
    train.write_text("x")

    calls: dict = {}
    _stub_awq_steps(monkeypatch, calls)
    artifacts = pipeline.calibrate(cfg, ws, f16, train, ws.corpus_dir / "eval.txt")

    # imatrix_variant must not leak into awq.calibrate kwargs.
    assert "imatrix_variant" not in calls["calibrate"]
    v = calls["imatrix_variant"]
    assert v["variant"] == "hybrid_custom"
    assert v["f16_gguf"] == ws.gguf_dir / "model-f16-awq.gguf"
    assert v["base_imatrix"] == ws.calibration_dir / "imatrix-awq.gguf"
    assert artifacts["imatrix"] == ws.calibration_dir / "imatrix-awq-hybrid_custom.gguf"


def test_quantize_filename_includes_variant(tmp_path, monkeypatch):
    """Two variants in one workspace must produce two GGUFs — otherwise the
    second run benches the first run's file under the new label."""
    import quant_tuner.quantize.gguf as gguf_mod

    quantized: list[Path] = []
    monkeypatch.setattr(
        gguf_mod, "quantize",
        lambda src, out, qtype, imatrix=None, log=None: (out.touch(), out)[1],
    )

    ws = pipeline.prepare_workspace(_make_cfg(tmp_path, {}))
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    for variant in ("analytic", "hybrid_custom"):
        cfg = RunConfig(
            name="t", model="m", workspace=tmp_path / "ws",
            calibration=CalibrationConfig(method="imatrix", variant=variant),
        )
        im = ws.calibration_dir / f"imatrix-{variant}.gguf"
        im.touch()
        out = pipeline.quantize_model(cfg, ws, f16, {"imatrix": im, "f16": f16})
        quantized.append(out)

    assert quantized[0] != quantized[1]
    assert quantized[0].name == "Q4_K_M-imatrix-analytic.gguf"
    assert quantized[1].name == "Q4_K_M-imatrix-hybrid_custom.gguf"


def test_resolve_device_passthrough_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_awq_calibrate_validates_cv_before_model_load(tmp_path):
    """cv_strategy without holdout_text must fail fast — before any model load."""
    from quant_tuner.calibrate import awq

    with pytest.raises(ValueError, match="holdout_text"):
        awq.calibrate(
            tmp_path, tmp_path / "corpus.txt", tmp_path / "out.pt",
            cv_strategy="mixed",
        )


def test_awq_calibrate_rejects_unknown_proxy_before_model_load(tmp_path):
    from quant_tuner.calibrate import awq

    with pytest.raises(ValueError, match="unknown proxy"):
        awq.calibrate(
            tmp_path, tmp_path / "corpus.txt", tmp_path / "out.pt",
            proxy="nope",
        )

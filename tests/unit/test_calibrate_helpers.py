"""Tests for the shared calibration helpers (_hf.forward_no_logits) and the
pipeline wiring added for tokenization-consistent eval + configurable
imatrix ctx."""

from __future__ import annotations

from pathlib import Path

from quant_tuner import pipeline
from quant_tuner.calibrate._hf import forward_no_logits
from quant_tuner.config import BenchConfig, CalibrationConfig, DataConfig, RunConfig


class _Recorder:
    def __init__(self, layers: bool):
        if layers:
            self.layers = ["L0"]
        self.called_with = None

    def __call__(self, input_ids=None):
        self.called_with = input_ids


class _CausalLM:
    """Stand-in with the standard ``.model.layers`` decoder trunk."""

    def __init__(self):
        self.model = _Recorder(layers=True)
        self.called_with = None

    def __call__(self, ids):
        self.called_with = ids


def test_forward_no_logits_calls_trunk_not_lm_head():
    m = _CausalLM()
    forward_no_logits(m, "IDS")
    assert m.model.called_with == "IDS"
    assert m.called_with is None  # full forward (lm_head) never invoked


def test_forward_no_logits_falls_back_without_standard_trunk():
    class Bare:
        def __init__(self):
            self.called_with = None

        def __call__(self, ids):
            self.called_with = ids

    b = Bare()
    forward_no_logits(b, "IDS")
    assert b.called_with == "IDS"

    class OddTrunk(Bare):  # has .model but no .layers -> must fall back too
        def __init__(self):
            super().__init__()
            self.model = object()

    o = OddTrunk()
    forward_no_logits(o, "IDS")
    assert o.called_with == "IDS"


def test_prepare_corpora_uses_prebuilt_eval_corpus(tmp_path: Path):
    """bench.eval_corpus short-circuits log-derived eval (no logs needed when
    the train corpus is also pre-built) — the path for tokenization-clean
    PPL/KLD eval text from scripts/build_corpora.py."""
    train_src = tmp_path / "cal.txt"
    train_src.write_text("calibration text")
    eval_src = tmp_path / "external_eval.txt"
    eval_src.write_text("raw eval text, no chat markers")

    cfg = RunConfig(
        name="t", model="m", workspace=tmp_path / "ws",
        data=DataConfig(corpus=train_src),  # no logs
        bench=BenchConfig(eval_corpus=eval_src),
    )
    ws = pipeline.prepare_workspace(cfg)
    train, eval_ = pipeline.prepare_corpora(cfg, ws)
    assert train.read_text() == "calibration text"
    assert eval_.read_text() == "raw eval text, no chat markers"


def test_imatrix_ctx_routed_to_llama_imatrix(tmp_path: Path, monkeypatch):
    """calibration.params.imatrix_ctx reaches llama-imatrix and is NOT
    forwarded to the variant builder."""
    captured: dict = {}

    def fake_imatrix(model, corpus, out, ctx=512, log=None, **kw):
        captured["ctx"] = ctx
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    def fake_variant(**kw):
        captured["variant_kwargs"] = kw
        kw["out_path"].touch()
        return kw["out_path"]

    monkeypatch.setattr(pipeline.llama_cpp, "imatrix", fake_imatrix)
    monkeypatch.setattr(pipeline.imatrix, "calibrate", fake_variant)

    cfg = RunConfig(
        name="t", model="m", workspace=tmp_path / "ws",
        calibration=CalibrationConfig(
            method="imatrix", variant="hybrid_custom",
            params={"imatrix_ctx": 2048},
        ),
    )
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    train = ws.corpus_dir / "train.txt"
    train.write_text("x")

    pipeline.calibrate(cfg, ws, f16, train, ws.corpus_dir / "eval.txt")
    assert captured["ctx"] == 2048
    assert "imatrix_ctx" not in captured["variant_kwargs"]


def test_imatrix_ctx_defaults_to_4096(tmp_path: Path, monkeypatch):
    """Without an explicit param, llama-imatrix runs at DEFAULT_IMATRIX_CTX
    (4096) so the packer's <=3500-token windows fit one context chunk — the
    old 512 default sliced every window across ~7 chunks."""
    captured: dict = {}

    def fake_imatrix(model, corpus, out, ctx=512, log=None, **kw):
        captured["ctx"] = ctx
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    monkeypatch.setattr(pipeline.llama_cpp, "imatrix", fake_imatrix)

    cfg = RunConfig(
        name="t", model="m", workspace=tmp_path / "ws",
        calibration=CalibrationConfig(method="imatrix", variant="default"),
    )
    ws = pipeline.prepare_workspace(cfg)
    f16 = ws.gguf_dir / "model-f16.gguf"
    f16.touch()
    train = ws.corpus_dir / "train.txt"
    train.write_text("x")

    pipeline.calibrate(cfg, ws, f16, train, ws.corpus_dir / "eval.txt")
    assert captured["ctx"] == pipeline.DEFAULT_IMATRIX_CTX == 4096

"""Unit tests for vllm_export (no llmcompressor / no model files needed)."""

from pathlib import Path

import pytest

from quant_tuner.vllm_export import PTQConfig, build_calibration_samples
from quant_tuner.vllm_export.w4a16 import DEFAULT_IGNORE


class FakeTokenizer:
    """Tokenizer stub: one int per whitespace-separated word."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [hash(w) % 1000 for w in text.split()]


def _cfg(tmp_path: Path, **kw) -> PTQConfig:
    corpus = tmp_path / "corpus.cal.txt"
    if not corpus.exists():
        corpus.write_text("word " * 5000)
    defaults = dict(
        model_id="fake/model",
        out_dir=tmp_path / "out",
        corpus_files=[corpus],
        ctx=256,
        budget_tokens=1024,
    )
    defaults.update(kw)
    return PTQConfig(**defaults)


def test_validate_accepts_good_config(tmp_path):
    _cfg(tmp_path).validate()


def test_validate_rejects_missing_corpus(tmp_path):
    cfg = _cfg(tmp_path, corpus_files=[tmp_path / "nope.txt"])
    with pytest.raises(ValueError, match="not found"):
        cfg.validate()


def test_validate_rejects_empty_corpus_list(tmp_path):
    cfg = _cfg(tmp_path, corpus_files=[])
    with pytest.raises(ValueError, match="at least one"):
        cfg.validate()


def test_validate_rejects_budget_below_ctx(tmp_path):
    cfg = _cfg(tmp_path, ctx=512, budget_tokens=256)
    with pytest.raises(ValueError, match="budget_tokens"):
        cfg.validate()


def test_validate_rejects_unknown_scheme(tmp_path):
    cfg = _cfg(tmp_path, scheme="W2A2")
    with pytest.raises(ValueError, match="unsupported scheme"):
        cfg.validate()


def test_samples_are_ctx_sized_and_within_budget(tmp_path):
    cfg = _cfg(tmp_path, ctx=256, budget_tokens=1024)
    samples = build_calibration_samples(cfg, FakeTokenizer())
    assert samples, "expected at least one sample"
    assert all(len(s) <= 256 for s in samples)
    # budget ~1024 at ctx 256 -> ~4 chunks (never wildly more)
    assert len(samples) <= 6


def test_samples_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    a = build_calibration_samples(cfg, FakeTokenizer())
    b = build_calibration_samples(cfg, FakeTokenizer())
    assert a == b


def test_budget_split_across_files(tmp_path):
    big = tmp_path / "big.txt"
    small = tmp_path / "small.txt"
    big.write_text("word " * 8000)
    small.write_text("word " * 1000)
    cfg = _cfg(tmp_path, corpus_files=[big, small], ctx=256, budget_tokens=2048)
    samples = build_calibration_samples(cfg, FakeTokenizer())
    # every file contributes at least one chunk
    assert len(samples) >= 2


def test_tiny_corpus_errors(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text(" ")
    cfg = _cfg(tmp_path, corpus_files=[empty])
    with pytest.raises(ValueError, match="zero tokens"):
        build_calibration_samples(cfg, FakeTokenizer())


def test_default_ignore_keeps_head_and_towers_full_precision():
    # lm_head must never be quantized (rare-token fidelity — the osoi5 lesson);
    # multimodal towers/embeddings mirror Google's official QAT W4A16 layout.
    assert "lm_head" in DEFAULT_IGNORE
    joined = " ".join(DEFAULT_IGNORE)
    for frag in ("vision_tower", "audio_tower", "embed_tokens", "per_layer"):
        assert frag in joined


def test_package_imports_without_llmcompressor():
    # run_ptq's heavy dep is lazy; importing the package must always work.
    import quant_tuner.vllm_export as ve

    assert callable(ve.run_ptq)

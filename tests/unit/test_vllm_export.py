"""Unit tests for vllm_export (no llmcompressor / no model files needed)."""

import json
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


# --- ignore auditing -------------------------------------------------------
#
# DEFAULT_IGNORE is gemma-shaped. Applied to a checkpoint whose tower is named
# something else it matches nothing and the tower is quantized to int4 against a
# text-only corpus, with no error raised. These pin the auditing that catches it.


def _write_index(tmp_path: Path, tensor_names: list[str]) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "shard-0.safetensors" for n in tensor_names}})
    )
    return root


QWEN35_SAMPLE = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.q_proj.bias",
    "model.visual.blocks.0.attn.proj.weight",
    "model.visual.blocks.0.attn.proj.bias",
    "mtp.layers.0.mlp.down_proj.weight",
    "lm_head.weight",
]


def test_checkpoint_module_names_strips_param_leaf_and_dedups():
    import tempfile

    from quant_tuner.vllm_export import checkpoint_module_names

    with tempfile.TemporaryDirectory() as td:
        root = _write_index(Path(td), QWEN35_SAMPLE)
        names = checkpoint_module_names(root)

    # weight+bias of one module collapse to a single module name.
    assert "model.visual.blocks.0.attn.proj" in names
    assert names.count("model.visual.blocks.0.attn.proj") == 1
    assert "lm_head" in names


def test_gemma_default_patterns_are_dead_on_a_qwen_style_tower():
    """The exact silent failure DEFAULT_IGNORE has on Qwen3.5."""
    from quant_tuner.vllm_export.w4a16 import _match_counts

    modules = sorted({n.rsplit(".", 1)[0] for n in QWEN35_SAMPLE})
    counts = _match_counts(modules, DEFAULT_IGNORE)

    assert counts["re:.*vision_tower.*"] == 0  # <- matches nothing here
    assert counts["lm_head"] == 1
    # ...while the pattern that actually fits the checkpoint does match.
    assert _match_counts(modules, ("re:.*visual.*",))["re:.*visual.*"] == 1


def test_match_counts_distinguishes_regex_from_exact():
    from quant_tuner.vllm_export.w4a16 import _match_counts

    modules = ["lm_head", "model.lm_head_proj", "mtp.fc"]
    counts = _match_counts(modules, ("lm_head", "re:mtp.*"))

    assert counts["lm_head"] == 1  # exact: does not catch lm_head_proj
    assert counts["re:mtp.*"] == 1


def test_dropped_tensors_reports_checkpoint_only_names(monkeypatch):
    """Tensors with no live module vanish from the export; no ignore can help."""
    import tempfile

    from quant_tuner.vllm_export import w4a16

    monkeypatch.setattr(
        w4a16,
        "model_module_names",
        lambda model_id, model_class=None: [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.visual.blocks.0.attn.proj",
            "lm_head",
        ],
    )
    with tempfile.TemporaryDirectory() as td:
        root = _write_index(Path(td), QWEN35_SAMPLE)
        dropped = w4a16.dropped_tensors(root)

    # transformers ignores ^mtp.* on load for every qwen3_5 class.
    assert dropped == ["mtp.layers.0.mlp.down_proj"]


def test_resolve_model_class_rejects_unknown_name():
    from quant_tuner.vllm_export import resolve_model_class

    with pytest.raises(ValueError, match="no class named"):
        resolve_model_class("NotARealModelClass")


def test_resolve_model_class_defaults_to_auto_causal_lm():
    from transformers import AutoModelForCausalLM

    from quant_tuner.vllm_export import resolve_model_class

    assert resolve_model_class(None) is AutoModelForCausalLM

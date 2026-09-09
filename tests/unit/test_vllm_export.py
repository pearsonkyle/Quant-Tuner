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


# --- weight grid -----------------------------------------------------------
#
# `scheme="W4A16"` is a preset (int4 / group-128 / symmetric / minmax). Anything
# else has to be handed to llmcompressor as explicit config_groups, and the two
# are mutually exclusive at the modifier. These pin which path a config takes.


def test_preset_scheme_needs_no_config_groups(tmp_path):
    from quant_tuner.vllm_export import build_config_groups

    cfg = _cfg(tmp_path)
    assert cfg.custom_weight_grid() is False
    assert build_config_groups(cfg) is None


@pytest.mark.parametrize(
    "override",
    [
        {"group_size": 32},
        {"symmetric": False},
        {"observer": "imatrix-mse"},
        {"actorder": "static"},
    ],
)
def test_any_grid_override_switches_to_config_groups(tmp_path, override):
    from quant_tuner.vllm_export import build_config_groups

    cfg = _cfg(tmp_path, **override)
    assert cfg.custom_weight_grid() is True
    assert build_config_groups(cfg) is not None


def test_config_groups_match_the_published_card_recipe(tmp_path):
    """int4, asymmetric, group 32, imatrix-mse, static act-order."""
    from quant_tuner.vllm_export import build_config_groups

    cfg = _cfg(
        tmp_path,
        group_size=32,
        symmetric=False,
        observer="imatrix-mse",
        actorder="static",
    )
    cfg.validate()
    weights = build_config_groups(cfg)["group_0"]["weights"]

    assert weights == {
        "num_bits": 4,
        "type": "int",
        "symmetric": False,
        "strategy": "group",
        "group_size": 32,
        "observer": "imatrix-mse",
        "actorder": "static",
    }


def test_per_channel_grid_drops_group_size(tmp_path):
    # strategy "channel" with a group_size is contradictory — compressed-tensors
    # rejects it, so the key must be absent, not present-and-negative.
    from quant_tuner.vllm_export import build_config_groups

    cfg = _cfg(tmp_path, group_size=-1)
    weights = build_config_groups(cfg)["group_0"]["weights"]
    assert weights["strategy"] == "channel"
    assert "group_size" not in weights


def test_w8a16_grid_uses_8_bits(tmp_path):
    from quant_tuner.vllm_export import build_config_groups

    cfg = _cfg(tmp_path, scheme="W8A16", group_size=32)
    assert build_config_groups(cfg)["group_0"]["weights"]["num_bits"] == 8


def test_validate_rejects_group_size_vllm_cannot_serve(tmp_path):
    # Exports cleanly, then fails at `vllm serve` — after the calibration is
    # already spent. Catch it at config time.
    cfg = _cfg(tmp_path, group_size=48)
    with pytest.raises(ValueError, match="not servable by vLLM"):
        cfg.validate()


def test_validate_rejects_unknown_observer_and_actorder(tmp_path):
    with pytest.raises(ValueError, match="unknown observer"):
        _cfg(tmp_path, observer="magic").validate()
    with pytest.raises(ValueError, match="unknown actorder"):
        _cfg(tmp_path, actorder="sideways").validate()


def test_observer_spellings_are_interchangeable(tmp_path):
    # The registry name is hyphenated; recipes in the wild write it either way.
    _cfg(tmp_path, observer="imatrix_mse").validate()
    _cfg(tmp_path, observer="imatrix-mse").validate()


def test_activation_quantized_schemes_refuse_a_custom_grid(tmp_path):
    # A hand-built group for W8A8 would have to specify input_activations too;
    # silently dropping the override would be worse than refusing.
    cfg = _cfg(tmp_path, scheme="W8A8", group_size=32)
    with pytest.raises(ValueError, match="preset settings only"):
        cfg.validate()


# --- fp8 KV cache ----------------------------------------------------------


def test_kv_cache_scheme_is_static_per_tensor_fp8(tmp_path):
    """`dynamic: False` is the whole point — it is what makes the oneshot pass
    *calibrate* the scales rather than defer them to runtime."""
    from quant_tuner.vllm_export import build_kv_cache_scheme

    cfg = _cfg(tmp_path, kv_cache_dtype="fp8_e4m3")
    scheme = build_kv_cache_scheme(cfg)

    assert scheme["num_bits"] == 8
    assert scheme["type"] == "float"
    assert scheme["strategy"] == "tensor"
    assert scheme["dynamic"] is False


def test_no_kv_cache_scheme_by_default(tmp_path):
    from quant_tuner.vllm_export import build_kv_cache_scheme

    assert build_kv_cache_scheme(_cfg(tmp_path)) is None


def test_kv_cache_scheme_is_a_copy_not_the_module_constant(tmp_path):
    from quant_tuner.vllm_export import KV_CACHE_SCHEMES, build_kv_cache_scheme

    scheme = build_kv_cache_scheme(_cfg(tmp_path, kv_cache_dtype="fp8_e4m3"))
    scheme["num_bits"] = 999
    assert KV_CACHE_SCHEMES["fp8_e4m3"]["num_bits"] == 8


def test_validate_rejects_unknown_kv_cache_dtype(tmp_path):
    with pytest.raises(ValueError, match="unknown kv_cache_dtype"):
        _cfg(tmp_path, kv_cache_dtype="fp4").validate()


# --- export verification ---------------------------------------------------
#
# fp8 KV fails *quietly*: a checkpoint whose scales never got written still
# loads and still serves. These pin the guardrail that catches it.


def _write_export(tmp_path: Path, quantization_config, tensors: list[str]) -> Path:
    out = tmp_path / "export"
    out.mkdir(exist_ok=True)
    config = {"architectures": ["Fake"]}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (out / "config.json").write_text(json.dumps(config))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "shard-0.safetensors" for n in tensors}})
    )
    return out


_KV_QCFG = {
    "format": "pack-quantized",
    "config_groups": {},
    "kv_cache_scheme": {"num_bits": 8, "type": "float", "strategy": "tensor"},
}


def test_verify_export_rejects_a_checkpoint_with_no_quantization_config(tmp_path):
    from quant_tuner.vllm_export import verify_export

    out = _write_export(tmp_path, None, ["model.layers.0.self_attn.q_proj.weight"])
    with pytest.raises(RuntimeError, match="no quantization_config"):
        verify_export(out, _cfg(tmp_path))


def test_verify_export_rejects_requested_kv_with_no_kv_cache_scheme(tmp_path):
    from quant_tuner.vllm_export import verify_export

    out = _write_export(
        tmp_path,
        {"format": "pack-quantized", "config_groups": {}},
        ["model.layers.0.self_attn.q_proj.weight"],
    )
    cfg = _cfg(tmp_path, kv_cache_dtype="fp8_e4m3")
    with pytest.raises(RuntimeError, match="no kv_cache_scheme"):
        verify_export(out, cfg)


def test_verify_export_rejects_kv_scheme_that_produced_no_scales(tmp_path):
    """The nastiest variant: the config *claims* fp8 KV, the tensors are absent."""
    from quant_tuner.vllm_export import verify_export

    out = _write_export(tmp_path, _KV_QCFG, ["model.layers.0.self_attn.q_proj.weight"])
    cfg = _cfg(tmp_path, kv_cache_dtype="fp8_e4m3")
    with pytest.raises(RuntimeError, match="no k_scale/v_scale tensors"):
        verify_export(out, cfg)


def test_verify_export_accepts_a_calibrated_kv_checkpoint(tmp_path):
    from quant_tuner.vllm_export import verify_export

    out = _write_export(
        tmp_path,
        _KV_QCFG,
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.v_scale",
        ],
    )
    observed = verify_export(out, _cfg(tmp_path, kv_cache_dtype="fp8_e4m3"))
    assert observed["kv_scale_tensors"] == {"k_scale": 1, "v_scale": 1}


def test_count_kv_scales_ignores_linear_attention_layers(tmp_path):
    """A hybrid model yields fewer scales than it has layers, correctly:
    compressed-tensors' KV targets match `self_attn`/`attention`, and Qwen3.8's
    48 DeltaNet layers are `linear_attn`. Count against real attention layers,
    never against layer count."""
    from quant_tuner.vllm_export import count_kv_scales

    out = _write_export(
        tmp_path,
        _KV_QCFG,
        [
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.3.self_attn.k_scale",
            "model.layers.3.self_attn.v_scale",
            "model.layers.7.self_attn.k_scale",
            "model.layers.7.self_attn.v_scale",
        ],
    )
    assert count_kv_scales(out) == {"k_scale": 2, "v_scale": 2}


# --- llmcompressor contract ------------------------------------------------
#
# Everything above tests the dicts we build. These test that llmcompressor
# actually *accepts* them and resolves them to what we meant — the one thing a
# pure-dict test cannot catch, and the thing that breaks on a version bump.
# Skipped without the `vllm-ptq` extra, so the base env stays green.


def _gptq_modifier(cfg):
    from llmcompressor.modifiers.quantization import GPTQModifier

    from quant_tuner.vllm_export import build_config_groups, build_kv_cache_scheme

    kwargs = {"ignore": list(cfg.ignore)}
    groups = build_config_groups(cfg)
    if groups is not None:
        kwargs["config_groups"] = groups
    else:
        kwargs["targets"] = "Linear"
        kwargs["scheme"] = cfg.scheme
    kv = build_kv_cache_scheme(cfg)
    if kv is not None:
        kwargs["kv_cache_scheme"] = kv
    return GPTQModifier(**kwargs)


def test_llmcompressor_resolves_the_card_recipe(tmp_path):
    pytest.importorskip("llmcompressor")

    cfg = _cfg(
        tmp_path,
        group_size=32,
        symmetric=False,
        observer="imatrix-mse",
        actorder="static",
        kv_cache_dtype="fp8_e4m3",
    )
    weights = _gptq_modifier(cfg).resolved_config.config_groups["group_0"].weights

    assert (weights.num_bits, weights.type.value if hasattr(weights.type, "value")
            else weights.type) == (4, "int")
    assert weights.symmetric is False
    assert weights.group_size == 32
    assert str(weights.strategy) .endswith("group")
    assert weights.observer == "imatrix-mse"
    assert str(weights.actorder).endswith("static")


def test_llmcompressor_kv_scheme_resolves_to_fp8_e4m3(tmp_path):
    """The dtype is implied by num_bits+type, never named — assert the resolved
    zero-point dtype so a scheme silently resolving to fp8_e5m2 (or to nothing)
    cannot pass."""
    import torch

    pytest.importorskip("llmcompressor")

    cfg = _cfg(tmp_path, kv_cache_dtype="fp8_e4m3")
    kv = _gptq_modifier(cfg).kv_cache_scheme

    assert kv is not None
    assert kv.num_bits == 8
    assert kv.dynamic is False
    assert kv.zp_dtype == torch.float8_e4m3fn


def test_kv_cache_scheme_makes_attention_a_calibration_target(tmp_path):
    """Without attention in resolved_targets no KV observer ever attaches and
    the export carries no scales — the failure verify_export exists to catch."""
    pytest.importorskip("llmcompressor")

    plain = _gptq_modifier(_cfg(tmp_path)).resolved_targets
    with_kv = _gptq_modifier(_cfg(tmp_path, kv_cache_dtype="fp8_e4m3")).resolved_targets

    added = with_kv - plain
    assert added, "kv_cache_scheme added no targets"
    assert all("attn" in t or "attention" in t for t in added), added


def test_known_observers_still_exist_in_the_registry():
    """Drift canary: our tuple is a copy of llmcompressor's registry."""
    pytest.importorskip("llmcompressor")
    from llmcompressor.observers import Observer

    from quant_tuner.vllm_export import KNOWN_OBSERVERS, normalize_observer

    registered = {normalize_observer(n) for n in Observer.registered_names()}
    assert {normalize_observer(o) for o in KNOWN_OBSERVERS} <= registered


def test_known_actorder_matches_compressed_tensors_enum():
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization import ActivationOrdering

    from quant_tuner.vllm_export import KNOWN_ACTORDER

    assert set(KNOWN_ACTORDER) == {e.value for e in ActivationOrdering}


def test_kv_scale_suffixes_track_compressed_tensors_targets():
    """If compressed-tensors renames its KV targets, count_kv_scales silently
    counts zero and verify_export starts failing on good checkpoints."""
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization.utils import KV_CACHE_TARGETS

    joined = " ".join(KV_CACHE_TARGETS)
    assert "self_attn" in joined or "attention" in joined

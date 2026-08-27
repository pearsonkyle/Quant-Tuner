"""vLLM-native PTQ exports (compressed-tensors), sibling to the GGUF pipeline.

Everything here produces a *safetensors checkpoint vLLM serves directly*
(``quantization_config`` in ``config.json``), not a GGUF. The heavy dependency
(``llmcompressor``) is imported lazily inside :func:`quant_tuner.vllm_export.w4a16.run_ptq`
so importing this package — and resolving configs in tests — never needs it
(same convention as ``eval.agents``).
"""

from quant_tuner.vllm_export.w4a16 import (
    DEFAULT_IGNORE,
    KNOWN_ACTORDER,
    KNOWN_OBSERVERS,
    KV_CACHE_SCHEMES,
    KV_SCALE_SUFFIXES,
    SUPPORTED_SCHEMES,
    VLLM_GROUP_SIZES,
    WEIGHT_ONLY_SCHEMES,
    PTQConfig,
    audit_ignore,
    build_calibration_samples,
    build_config_groups,
    build_kv_cache_scheme,
    checkpoint_module_names,
    checkpoint_tensor_names,
    count_kv_scales,
    dropped_tensors,
    model_module_names,
    normalize_observer,
    resolve_model_class,
    run_ptq,
    verify_export,
)

__all__ = [
    "DEFAULT_IGNORE",
    "KNOWN_ACTORDER",
    "KNOWN_OBSERVERS",
    "KV_CACHE_SCHEMES",
    "KV_SCALE_SUFFIXES",
    "PTQConfig",
    "SUPPORTED_SCHEMES",
    "VLLM_GROUP_SIZES",
    "WEIGHT_ONLY_SCHEMES",
    "audit_ignore",
    "build_calibration_samples",
    "build_config_groups",
    "build_kv_cache_scheme",
    "checkpoint_module_names",
    "checkpoint_tensor_names",
    "count_kv_scales",
    "dropped_tensors",
    "model_module_names",
    "normalize_observer",
    "resolve_model_class",
    "run_ptq",
    "verify_export",
]

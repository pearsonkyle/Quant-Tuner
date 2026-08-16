"""vLLM-native PTQ exports (compressed-tensors), sibling to the GGUF pipeline.

Everything here produces a *safetensors checkpoint vLLM serves directly*
(``quantization_config`` in ``config.json``), not a GGUF. The heavy dependency
(``llmcompressor``) is imported lazily inside :func:`quant_tuner.vllm_export.w4a16.run_ptq`
so importing this package — and resolving configs in tests — never needs it
(same convention as ``eval.agents``).
"""

from quant_tuner.vllm_export.w4a16 import (
    DEFAULT_IGNORE,
    PTQConfig,
    audit_ignore,
    build_calibration_samples,
    checkpoint_module_names,
    dropped_tensors,
    model_module_names,
    resolve_model_class,
    run_ptq,
)

__all__ = [
    "DEFAULT_IGNORE",
    "PTQConfig",
    "audit_ignore",
    "build_calibration_samples",
    "checkpoint_module_names",
    "dropped_tensors",
    "model_module_names",
    "resolve_model_class",
    "run_ptq",
]

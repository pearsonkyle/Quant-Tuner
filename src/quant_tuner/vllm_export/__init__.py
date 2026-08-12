"""vLLM-native PTQ exports (compressed-tensors), sibling to the GGUF pipeline.

Everything here produces a *safetensors checkpoint vLLM serves directly*
(``quantization_config`` in ``config.json``), not a GGUF. The heavy dependency
(``llmcompressor``) is imported lazily inside :func:`quant_tuner.vllm_export.w4a16.run_ptq`
so importing this package — and resolving configs in tests — never needs it
(same convention as ``eval.agents``).
"""

from quant_tuner.vllm_export.w4a16 import PTQConfig, build_calibration_samples, run_ptq

__all__ = ["PTQConfig", "build_calibration_samples", "run_ptq"]

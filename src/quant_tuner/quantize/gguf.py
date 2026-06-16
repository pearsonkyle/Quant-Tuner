"""GGUF quantization via llama-quantize.

The quant `type` is passed through to llama-quantize unchanged — to switch from
Q4_K_M to IQ4_XS, Q5_K_M, IQ3_S, etc., change one string.
"""

from __future__ import annotations

from pathlib import Path

from quant_tuner.models import llama_cpp


def quantize(
    f16_gguf: Path,
    out_gguf: Path,
    quant_type: str,
    imatrix: Path | None = None,
    log: Path | None = None,
    tensor_types: dict[str, str] | None = None,
) -> Path:
    """Quantize an F16 GGUF to `quant_type`, optionally guided by an imatrix.

    ``tensor_types`` pins matching tensors to a different ggml type (by name
    substring) — e.g. ``{"nextn": "q8_0"}`` keeps the MTP draft head
    near-lossless while the trunk goes to a 2-bit type.
    """
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    return llama_cpp.quantize(
        f16_gguf, out_gguf, quant_type,
        imatrix=imatrix, log=log, tensor_types=tensor_types,
    )

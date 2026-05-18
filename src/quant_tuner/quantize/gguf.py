"""GGUF quantization via llama-quantize.

The quant `type` is passed through to llama-quantize unchanged — to switch from
Q4_K_M to IQ4_XS, Q5_K_M, IQ3_S, etc., change one string. The registry below is
purely informational (for help text); unknown types still work.
"""

from __future__ import annotations

from pathlib import Path

from quant_tuner.models import llama_cpp

# Common llama.cpp quant tags. Not exhaustive; any tag llama-quantize accepts works.
KNOWN_QUANT_TYPES: tuple[str, ...] = (
    "F16", "BF16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M", "Q5_K_S",
    "Q4_K_M", "Q4_K_S",
    "Q3_K_M", "Q3_K_S",
    "IQ4_XS", "IQ4_NL",
    "IQ3_S", "IQ3_XXS",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
)


def quantize(
    f16_gguf: Path,
    out_gguf: Path,
    quant_type: str,
    imatrix: Path | None = None,
    log: Path | None = None,
) -> Path:
    """Quantize an F16 GGUF to `quant_type`, optionally guided by an imatrix."""
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    return llama_cpp.quantize(f16_gguf, out_gguf, quant_type, imatrix=imatrix, log=log)

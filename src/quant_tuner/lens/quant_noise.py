"""Per-tensor quantization noise: how far did each weight tensor move?

Importable home for the weight-noise steps that ``scripts/measure_quant_noise.py``
grew inline, so the calibration study (:mod:`quant_tuner.lens.study`) can share
them. The method: dequantize a quant back to F16 (``llama-quantize
--allow-requantize``), then measure per-tensor relative RMSE against the true
F16. This isolates the pure weight-space damage a quant type inflicts on each
tensor — the raw material for correlating "where does calibration buy fidelity"
against imatrix importance concentration.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from quant_tuner.paths import ensure_gguf_py, llama_bin

logger = logging.getLogger(__name__)


def dequantize_gguf(quant_gguf: Path, tmp_dir: Path) -> Path:
    """Requantize a quant back to F16 via ``llama-quantize --allow-requantize``."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{Path(quant_gguf).stem}_dequant_f16.gguf"
    if out.exists():
        logger.info("dequant F16 already exists: %s", out.name)
        return out
    cmd = [str(llama_bin("llama-quantize")), "--allow-requantize", str(quant_gguf), str(out), "F16"]
    logger.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"llama-quantize failed:\n{result.stderr}")
    return out


def load_gguf_weights(gguf_path: Path) -> dict[str, np.ndarray]:
    """Load all F16/F32/BF16 tensors from a GGUF as name -> float32 array."""
    ensure_gguf_py()
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(str(gguf_path))
    weights: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        tt = t.tensor_type
        if tt == GGMLQuantizationType.F16:
            arr = t.data.view(np.float16).reshape(t.shape).astype(np.float32)
        elif tt == GGMLQuantizationType.F32:
            arr = t.data.view(np.float32).reshape(t.shape)
        elif tt == GGMLQuantizationType.BF16:
            raw = t.data.view(np.uint16).reshape(t.shape)
            arr = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            continue
        weights[t.name] = arr
    return weights


def measure_weight_noise(
    f16_weights: dict[str, np.ndarray],
    dq_weights: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-tensor relative RMSE ``||W_f16 - W_dq|| / ||W_f16||``."""
    shared = set(f16_weights) & set(dq_weights)
    noise: dict[str, float] = {}
    for name in shared:
        w_ref = f16_weights[name]
        w_dq = dq_weights[name]
        if w_ref.shape != w_dq.shape:
            continue
        diff_rms = float(np.sqrt(np.mean((w_ref - w_dq) ** 2)))
        ref_rms = float(np.sqrt(np.mean(w_ref**2))) + 1e-8
        noise[name] = diff_rms / ref_rms
    return noise


def per_tensor_rmse(f16_gguf: Path, quant_gguf: Path, tmp_dir: Path) -> dict[str, float]:
    """End-to-end: dequant the quant, diff against F16, return per-tensor RMSE."""
    dq = dequantize_gguf(quant_gguf, tmp_dir)
    return measure_weight_noise(load_gguf_weights(f16_gguf), load_gguf_weights(dq))

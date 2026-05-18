"""HuggingFace → F16 GGUF via the vendored llama.cpp converter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from quant_tuner.paths import LLAMA_CPP_DIR


def hf_to_f16_gguf(model_dir: Path, out_gguf: Path, log: Path | None = None) -> Path:
    """Convert a HuggingFace model directory to an F16 GGUF.

    Uses llama.cpp/convert_hf_to_gguf.py. Requires the HF model files to be present
    locally (use `models.extract` first if starting from a remote repo).
    """
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing converter: {converter}")
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(converter),
        str(model_dir),
        "--outfile", str(out_gguf),
        "--outtype", "f16",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out)
    if proc.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf failed:\n{out[-2000:]}")
    return out_gguf

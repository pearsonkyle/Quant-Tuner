"""Workspace layout. A workspace is one calibration job's output directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_CPP_DIR = REPO_ROOT / "vendor" / "llama.cpp"


def llama_bin(name: str) -> Path:
    """Locate a built llama.cpp binary, honoring $LLAMA_CPP_DIR override."""
    root = Path(os.environ.get("LLAMA_CPP_DIR", LLAMA_CPP_DIR))
    candidate = root / "build" / "bin" / name
    if not candidate.exists():
        raise FileNotFoundError(
            f"{candidate} not found. Build llama.cpp first:\n"
            f"  cmake -S {root} -B {root}/build -DGGML_METAL=ON\n"
            f"  cmake --build {root}/build -j"
        )
    return candidate


@dataclass(frozen=True)
class Workspace:
    """All artifacts for one calibration job live under root/."""

    root: Path

    @property
    def model_extracted(self) -> Path:
        return self.root / "model_extracted"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration"

    @property
    def gguf_dir(self) -> Path:
        return self.root / "gguf"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def results_csv(self) -> Path:
        return self.root / "results.csv"

    def ensure(self) -> None:
        for d in (
            self.model_extracted,
            self.corpus_dir,
            self.calibration_dir,
            self.gguf_dir,
            self.eval_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

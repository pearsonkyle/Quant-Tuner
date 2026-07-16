"""CLI shim for QAT export — all logic lives in ``quant_tuner.qat.export``.

    LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python \
        scripts/exp057_qat_export.py --latents out/exp-058/trained_iter5/trained_latents.pt \
        --tag iter5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.export import export_qat  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--tag", default="qat")
    args = ap.parse_args()
    export_qat(args.latents, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())

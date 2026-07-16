"""CLI shim for the QAT trainer — all logic lives in ``quant_tuner.qat.train``.

    PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \
        scripts/exp058_qat_train_v2.py --corpus out/exp-058/distill_corpus_4096.pt \
        --layers 0-35 --optim adafactor --epochs 1 --grad-accum 4 --lr 5e-5 \
        --out out/exp-058/trained_iter5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

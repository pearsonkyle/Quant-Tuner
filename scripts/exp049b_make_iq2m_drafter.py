"""exp-049b: build an IQ2_M MTP drafter via a UNIFORM (synthetic) imatrix.

IQ2_M requires an imatrix, but the `gemma4-assistant` drafter cannot run
standalone in llama-imatrix ("Gemma4Assistant requires ctx_other to be set" —
its attention has only attn_q/attn_output and cross-attends to the trunk's KV),
so no *calibrated* imatrix is collectable. This writes a uniform imatrix
(importance = 1.0 for every input channel of all 2-D matmul weights) purely to
satisfy llama-quantize's gate, then quantizes. The result has NO activation
weighting — it is a floor on IQ2_M quality, not the best achievable.

Usage:
    PYTHONPATH=vendor/llama.cpp/gguf-py .venv/bin/python \
        scripts/exp049b_make_iq2m_drafter.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vendor/llama.cpp/gguf-py"))

from gguf import GGUFReader  # noqa: E402

from quant_tuner.calibrate.imatrix import write_imatrix  # noqa: E402

PKG = REPO / "uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF"
EXP = REPO / "out/exp-049"
DRAFTER = PKG / "mtp-gemma-4-31B-it.gguf"
# Any existing imatrix GGUF works as a KV-schema source (chunk_count/size/datasets).
SCHEMA = REPO / "out/exp-044/vanilla/imatrix-cal.gguf"
IMATRIX = EXP / "mtp.uniform.imatrix.gguf"
OUT = EXP / "mtp-gemma-4-31B-it-IQ2_M.gguf"
QUANTIZE = REPO / "vendor/llama.cpp/build/bin/llama-quantize"


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)

    # Uniform importance = ones(ne[0]) for every 2-D matmul weight. gguf-py
    # ReaderTensor.shape is in ggml order, so shape[0] == ne[0] == n_in (the
    # channel axis the imatrix indexes). Covers blk.* attn_q/attn_output/ffn_*
    # AND nextn.{pre,post}_projection + token_embd — llama-quantize demands an
    # entry for every tensor it puts at a very-low-bit type.
    reader = GGUFReader(str(DRAFTER))
    importance = {}
    for t in reader.tensors:
        shp = tuple(int(x) for x in t.shape)
        if len(shp) == 2 and shp[0] > 1 and shp[1] > 1:
            importance[t.name] = np.ones(shp[0], dtype=np.float32)
    print(f"uniform imatrix: {len(importance)} matmul tensors")

    write_imatrix(IMATRIX, importance, SCHEMA,
                  dataset_label="<uniform-synthetic-no-activation-weighting>")
    print(f"wrote {IMATRIX}")

    cmd = [str(QUANTIZE), "--allow-requantize", "--imatrix", str(IMATRIX),
           str(DRAFTER), str(OUT), "IQ2_M"]
    print("+", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        return r.returncode
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

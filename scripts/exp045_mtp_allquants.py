"""exp-045: re-quantize ALL tmax-27b quants from the MTP-grafted F16, bundling the
grafted Qwopus nextn head (blk.64) at Q8_0 into every GGUF.

Consumes exp045_mtp_iq4xs.py's F16 (out/exp-045/model-f16-mtp.gguf, trunk + grafted
blk.64). Bundling blk.64@Q8 does NOT change any trunk tensor, so the trunk's KLD / PPL
/ top_p are identical to the non-MTP quants already in results.csv — no re-bench needed.
Draft acceptance was characterized separately (mtp_accept_sweep_iq4xs.json).

Outputs out/exp-045/<label>_mtp/tmax-27b-<QUANT>-mtp.gguf for the 6 release quants.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp045_mtp_allquants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import gguf

EXP45 = REPO / "out" / "exp-045"
F16_MTP = EXP45 / "model-f16-mtp.gguf"
HYBRID_IMATRIX = EXP45 / "imatrix-hybrid_custom.gguf"
NEXTN_PIN = {"blk.64.": "q8_0"}

# (label, quant ftype, method)
ROWS = [
    ("q2k_plain_mtp", "Q2_K",   "plain"),
    ("iq2xs_mtp",     "IQ2_XS", "imatrix"),
    ("iq2m_mtp",      "IQ2_M",  "imatrix"),
    ("q2ks_mtp",      "Q2_K_S", "imatrix"),
    ("iq3m_mtp",      "IQ3_M",  "imatrix"),
    ("iq4xs_mtp",     "IQ4_XS", "imatrix"),
    ("q5km_mtp",      "Q5_K_M", "imatrix"),
]


def main() -> int:
    if not F16_MTP.exists():
        raise FileNotFoundError(f"run exp045_mtp_iq4xs.py first (need grafted F16): {F16_MTP}")
    if not HYBRID_IMATRIX.exists():
        raise FileNotFoundError(f"missing imatrix: {HYBRID_IMATRIX}")

    for label, quant, method in ROWS:
        sub = EXP45 / label
        (sub / "logs").mkdir(parents=True, exist_ok=True)
        qpath = sub / f"tmax-27b-{quant}-mtp.gguf"
        imat = None if method == "plain" else HYBRID_IMATRIX
        with phase(f"[exp-045][{label}] quantize {quant} (nextn@Q8)"):
            step("quantize", qpath,
                 lambda q=qpath, qt=quant, im=imat, ld=sub / "logs": gguf.quantize(
                     F16_MTP, q, qt, imatrix=im, tensor_types=NEXTN_PIN,
                     log=ld / "quantize.log"))
            log(f"  {qpath.name}: {qpath.stat().st_size/1024**3:.2f} GiB")

    log("")
    log("=== all 6 MTP-bundled quants built ===")
    log("  Next: exp045_prepare_release.py (stages the -mtp GGUFs), then clean up the")
    log("  grafted F16 + model_extracted_mtp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

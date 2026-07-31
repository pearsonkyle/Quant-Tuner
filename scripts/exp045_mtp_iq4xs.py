"""exp-045: convert the MTP-grafted tmax-27b to F16 and quantize IQ4_XS (nextn@Q8).

Consumes exp045_graft_mtp.py's output (tmax trunk + grafted Qwopus nextn head). Emits
a F16 GGUF whose blk.64 is the draft head, then an IQ4_XS quant that pins the WHOLE
nextn layer (blk.64.*) to Q8_0 (near-lossless draft) while the trunk uses the existing
hybrid imatrix. After this, bench acceptance with bench_mtp_speed.py --n-max {1,2,3,4}.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp045_graft_mtp.py
    PYTHONPATH=src .venv/bin/python scripts/exp045_mtp_iq4xs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import convert, gguf

EXP45 = REPO / "out" / "exp-045"
GRAFT_HF = EXP45 / "model_extracted_mtp"
F16_MTP = EXP45 / "model-f16-mtp.gguf"
HYBRID_IMATRIX = EXP45 / "imatrix-hybrid_custom.gguf"
OUT_DIR = EXP45 / "iq4xs_mtp"
QUANT = OUT_DIR / "tmax-27b-IQ4_XS-mtp.gguf"
LOGS = EXP45 / "logs"
NEXTN_PIN = {"blk.64.": "q8_0"}


def _inspect_blk64() -> None:
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys,re;from gguf import GGUFReader;r=GGUFReader(sys.argv[1]);"
         "n=[t.name for t in r.tensors];"
         "b=[x for x in n if x.startswith('blk.64.')];"
         "print('  blk.64 tensors:',len(b),b[:4])",
         str(F16_MTP)],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO / "vendor" / "llama.cpp" / "gguf-py")})
    print(out.stdout or out.stderr[-400:])


def main() -> int:
    if not (GRAFT_HF / "config.json").exists():
        raise FileNotFoundError(f"run exp045_graft_mtp.py first: {GRAFT_HF}")
    if not HYBRID_IMATRIX.exists():
        raise FileNotFoundError(f"missing imatrix (run exp045_quants_tmax.py): {HYBRID_IMATRIX}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)

    with phase("[exp-045] convert grafted HF -> F16 GGUF (blk.64 = nextn)"):
        step("convert", F16_MTP,
             lambda: convert.hf_to_f16_gguf(GRAFT_HF, F16_MTP, log=LOGS / "convert-mtp.log"))
    _inspect_blk64()

    with phase("[exp-045] quantize IQ4_XS (nextn@Q8)"):
        step("quantize", QUANT,
             lambda: gguf.quantize(F16_MTP, QUANT, "IQ4_XS", imatrix=HYBRID_IMATRIX,
                                   tensor_types=NEXTN_PIN, log=OUT_DIR / "logs" / "quantize.log"))

    log("")
    log("=== MTP-grafted IQ4_XS built ===")
    log(f"  {QUANT}")
    log("  Next: bench_mtp_speed.py --model <this> --holdout-jsonl logtrain.jsonl")
    log("  --n-max {1,2,3,4} to measure draft acceptance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

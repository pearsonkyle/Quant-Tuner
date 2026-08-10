"""exp-041 setup: fetch Jackrong/Qwopus3.6-27B-Coder, strip vision, KEEP MTP,
convert to F16 GGUF, and verify the nextn/MTP tensors survived conversion.

Qwopus3.6-27B-Coder is a full finetune of Qwen3.6-27B (arch
Qwen3_5ForConditionalGeneration, multimodal, mtp_num_hidden_layers=1). Unlike
Qwopus3.5-v3 it ships its OWN MTP head, so we don't graft anything — we just
preserve the model's nextn weights through extraction (keep_mtp=True strips the
vision tower but keeps mtp.*) so they bundle into the trunk GGUF and can be
served with `llama-server --spec-type draft-mtp`.

This is the riskiest + longest-fetch step, isolated so we confirm the
multimodal+MTP conversion works before investing in corpora/imatrix.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp041_setup_qwopus36.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract
from quant_tuner.quantize import convert

MODEL_ID = "Jackrong/Qwopus3.6-27B-Coder"

EXP41 = REPO / "out" / "exp-041"
HF_DIR = EXP41 / "model_extracted"
F16_GGUF = EXP41 / "model-f16.gguf"
LOGS = EXP41 / "logs"


def _inspect_nextn() -> None:
    """Print every nextn/mtp tensor in the F16 GGUF so we can pin them to Q8_0."""
    gguf_reader = REPO / "vendor" / "llama.cpp" / "gguf-py" / "gguf" / "scripts" / "gguf_dump.py"
    log("[exp-041] scanning F16 GGUF for nextn/MTP tensors:")
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; from gguf import GGUFReader; "
             "r=GGUFReader(sys.argv[1]); "
             "[print('  ', t.name, t.tensor_type.name) for t in r.tensors "
             " if 'nextn' in t.name.lower() or 'mtp' in t.name.lower()]",
             str(F16_GGUF)],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(REPO / "vendor" / "llama.cpp" / "gguf-py")},
        )
        print(out.stdout or "  (no nextn/mtp tensors found by name — inspect manually)")
        if out.stderr:
            print("  [stderr]", out.stderr[-500:])
    except Exception as e:  # noqa: BLE001
        log(f"  (nextn scan failed: {e}; inspect F16 GGUF manually)")


def main() -> int:
    EXP41.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase("[exp-041] extract HF (strip vision, KEEP MTP)"):
        step("extract HF", HF_DIR / "config.json",
             lambda: extract.extract_text_lm(
                 source=MODEL_ID, output_dir=HF_DIR, keep_mtp=True))

    # Confirm the extracted config still declares the MTP layer(s).
    import json
    cfg = json.loads((HF_DIR / "config.json").read_text())
    log(f"[exp-041] extracted config: mtp_num_hidden_layers="
        f"{cfg.get('mtp_num_hidden_layers')} "
        f"num_nextn_predict_layers={cfg.get('num_nextn_predict_layers')} "
        f"architectures={cfg.get('architectures')}")
    if not (cfg.get("mtp_num_hidden_layers") or cfg.get("num_nextn_predict_layers")):
        log("  !! WARNING: MTP layer count is 0/absent after extract — MTP was NOT kept.")

    with phase("[exp-041] convert -> F16 GGUF (nextn bundled into trunk)"):
        step("convert", F16_GGUF,
             lambda: convert.hf_to_f16_gguf(
                 HF_DIR, F16_GGUF, log=LOGS / "convert.log"))

    _inspect_nextn()

    log("")
    log("=== exp-041 setup complete ===")
    log(f"  HF (vision-stripped, MTP-kept): {HF_DIR}")
    log(f"  F16 GGUF (MTP bundled):         {F16_GGUF}")
    log("  Next: confirm the nextn tensor names above, then run corpora/imatrix")
    log("  + the 4 quant rows pinning those tensors to Q8_0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

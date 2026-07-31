"""exp-050 setup: fetch deepreinforce-ai/Ornith-1.0-9B, strip vision, convert to
F16 GGUF (text trunk only).

Ornith-1.0-9B is a Qwen3.5-family VLM (arch Qwen3_5ForConditionalGeneration,
multimodal). Its config declares `mtp_num_hidden_layers: 1`, BUT the model ships
NO MTP/nextn head weights (verified: 0 nextn tensors in the source safetensors).
So — unlike Qwopus3.6, which genuinely bundles its own nextn head — there is
nothing to preserve. We extract with keep_mtp=False: this drops the (nonexistent)
mtp.* weights AND zeroes the phantom mtp config fields, so the converter emits a
consistent block_count=32 GGUF. (keep_mtp=True leaves block_count=33 declaring a
blk.32 that has no weights → llama.cpp fails with "missing tensor blk.32.*".)

This script dumps any nextn/MTP tensor name after conversion as a sanity check
(expected: none) — Ornith is a plain 32-layer text LM once the vision tower is
stripped.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp050_setup_ornith.py
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

MODEL_ID = "deepreinforce-ai/Ornith-1.0-9B"

EXP50 = REPO / "out" / "exp-050"
HF_DIR = EXP50 / "model_extracted"
F16_GGUF = EXP50 / "model-f16.gguf"
LOGS = EXP50 / "logs"


def _inspect_nextn() -> None:
    """Print every nextn/mtp tensor in the F16 GGUF so we can pin them to Q8_0."""
    log("[exp-050] scanning F16 GGUF for nextn/MTP tensors:")
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
    EXP50.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase("[exp-050] extract HF (strip vision, drop phantom MTP)"):
        step("extract HF", HF_DIR / "config.json",
             lambda: extract.extract_text_lm(
                 source=MODEL_ID, output_dir=HF_DIR, keep_mtp=False))

    # Confirm the extracted config zeroed the phantom MTP fields.
    import json
    cfg = json.loads((HF_DIR / "config.json").read_text())
    log(f"[exp-050] extracted config: mtp_num_hidden_layers="
        f"{cfg.get('mtp_num_hidden_layers')} "
        f"num_nextn_predict_layers={cfg.get('num_nextn_predict_layers')} "
        f"num_hidden_layers={cfg.get('num_hidden_layers')} "
        f"architectures={cfg.get('architectures')}")
    if cfg.get("mtp_num_hidden_layers") or cfg.get("num_nextn_predict_layers"):
        log("  !! WARNING: MTP fields still set — GGUF block_count will be wrong.")

    with phase("[exp-050] convert -> F16 GGUF (text trunk, block_count=32)"):
        step("convert", F16_GGUF,
             lambda: convert.hf_to_f16_gguf(
                 HF_DIR, F16_GGUF, log=LOGS / "convert.log"))

    _inspect_nextn()

    log("")
    log("=== exp-050 setup complete ===")
    log(f"  HF (vision-stripped, text trunk): {HF_DIR}")
    log(f"  F16 GGUF (32-layer text LM):      {F16_GGUF}")
    log("  Expected nextn/MTP tensors: NONE (Ornith ships no draft head).")
    log("  Next: run exp050_quants_ornith.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

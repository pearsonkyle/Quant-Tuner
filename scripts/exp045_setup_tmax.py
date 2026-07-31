"""exp-045 setup: fetch allenai/tmax-27b, extract the text trunk, convert to F16 GGUF.

allenai/tmax-27b declares arch Qwen3_5ForConditionalGeneration (model_type qwen3_5,
config carries vision_config + mtp_num_hidden_layers=1), BUT the released checkpoint
contains ONLY the 64-layer text trunk (`model.language_model.*` + lm_head) — no vision
tower and **no MTP/nextn weights**. The mtp_num_hidden_layers=1 flag is vestigial
(inherited from the Qwen3.6 base config). So unlike Jackrong/Qwopus3.6-27B-Coder
(exp-041, which genuinely ships its nextn head), there is nothing to keep:
keep_mtp=False (default) drops the `mtp.` prefix AND zeros mtp_num_hidden_layers /
num_nextn_predict_layers in the extracted config — otherwise the GGUF converter writes
a header claiming a blk.64 nextn layer that has no weights, and llama.cpp then fails to
load with `missing tensor 'blk.64.attn_norm.weight'`.

This is the longest-fetch step, isolated so we confirm conversion before investing in
corpora/imatrix.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp045_setup_tmax.py
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

MODEL_ID = "allenai/tmax-27b"

EXP45 = REPO / "out" / "exp-045"
HF_DIR = EXP45 / "model_extracted"
F16_GGUF = EXP45 / "model-f16.gguf"
LOGS = EXP45 / "logs"


def _inspect_blocks() -> None:
    """Print the block count in the F16 GGUF; we expect a clean 64-block trunk
    (blk.0–63) with NO blk.64 / nextn / mtp tensors."""
    log("[exp-045] scanning F16 GGUF blocks (expect blk.0–63, no blk.64/nextn/mtp):")
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys, re; from gguf import GGUFReader; "
             "r=GGUFReader(sys.argv[1]); names=[t.name for t in r.tensors]; "
             "blks=sorted({int(m.group(1)) for n in names "
             "if (m:=re.match(r'blk\\.(\\d+)\\.', n))}); "
             "print('  blocks:', blks[:2], '...', blks[-2:], '| count', len(blks)); "
             "mtp=[n for n in names if 'nextn' in n.lower() or 'mtp' in n.lower() "
             "or n.startswith('blk.64.')]; "
             "print('  mtp/nextn/blk.64 tensors:', mtp or 'none (good)')",
             str(F16_GGUF)],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(REPO / "vendor" / "llama.cpp" / "gguf-py")},
        )
        print(out.stdout or "  (scan produced no output — inspect manually)")
        if out.stderr:
            print("  [stderr]", out.stderr[-500:])
    except Exception as e:  # noqa: BLE001
        log(f"  (block scan failed: {e}; inspect F16 GGUF manually)")


def main() -> int:
    EXP45.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase("[exp-045] extract HF text trunk (no MTP — zero the vestigial flag)"):
        step("extract HF", HF_DIR / "config.json",
             lambda: extract.extract_text_lm(
                 source=MODEL_ID, output_dir=HF_DIR, keep_mtp=False))

    # Confirm the extracted config zeroed the vestigial MTP flag.
    import json
    cfg = json.loads((HF_DIR / "config.json").read_text())
    log(f"[exp-045] extracted config: mtp_num_hidden_layers="
        f"{cfg.get('mtp_num_hidden_layers')} "
        f"num_nextn_predict_layers={cfg.get('num_nextn_predict_layers')} "
        f"architectures={cfg.get('architectures')}")
    if cfg.get("mtp_num_hidden_layers") or cfg.get("num_nextn_predict_layers"):
        log("  !! WARNING: MTP layer count is non-zero — the GGUF will declare a "
            "blk.64 nextn layer with no weights and fail to load. Expected 0.")

    with phase("[exp-045] convert -> F16 GGUF (64-block text trunk)"):
        step("convert", F16_GGUF,
             lambda: convert.hf_to_f16_gguf(
                 HF_DIR, F16_GGUF, log=LOGS / "convert.log"))

    _inspect_blocks()

    log("")
    log("=== exp-045 setup complete ===")
    log(f"  HF (text trunk):  {HF_DIR}")
    log(f"  F16 GGUF:         {F16_GGUF}")
    log("  Next: run exp045_quants_tmax.py (corpora/imatrix + 4 plain 2-bit quants).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

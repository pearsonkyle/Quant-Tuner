"""exp-060 stage 1: fetch Qwen3.8-27B, keep the MTP head, convert to F16 GGUF.

Mirrors exp-041 (Qwopus3.6-27B-Coder), generalized so nothing about the model is assumed:

* **MTP is detected, not declared.** ``config.json`` is read for a draft-layer count AND
  the safetensors index is checked for actual ``mtp.*`` weights. Those disagree in the
  wild — Ornith-1.0-9B's config claimed ``mtp_num_hidden_layers=1`` while shipping none,
  and extracting with ``keep_mtp=True`` produced a phantom block that crashed llama.cpp.
  Only "config says N **and** weights exist" turns on ``keep_mtp``.
* **The nextn layer index is read back out of the GGUF** (``models.mtp.describe``) instead
  of assuming ``blk.64``, and written to ``mtp_report.json`` for the quantize stage.
* **The chat template is verified against a tool-calling fixture** before anything
  expensive runs, since the whole calibration corpus is chat-templated text.

Qwen3.8-27B is unreleased at time of writing. Everything here is exercisable today against
a released Qwen-family sibling:

    # plumbing test (model that exists, same shape, ships its own MTP head)
    PYTHONPATH=src .venv/bin/python scripts/exp060_setup_qwen38.py \\
        --model Jackrong/Qwopus3.6-27B-Coder --run exp-060-dryrun

    # the real thing, once the repo is live
    PYTHONPATH=src .venv/bin/python scripts/exp060_setup_qwen38.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, mtp
from quant_tuner.quantize import convert

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RUN = "exp-060"


def _model_available(model_id: str) -> tuple[bool, str]:
    if Path(model_id).is_dir():
        return True, "local directory"
    from huggingface_hub import model_info
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    try:
        info = model_info(model_id)
    except RepositoryNotFoundError:
        return False, "repository not found (unreleased, renamed, or gated)"
    except HfHubHTTPError as e:  # noqa: BLE001
        return False, f"hub error: {e}"
    return True, f"{len(info.siblings or [])} files"


def _mtp_weight_count(model_path: Path) -> int:
    """Count real ``mtp.*`` tensors in the checkpoint — the claim-vs-weights cross-check."""
    idx = model_path / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text()).get("weight_map", {})
        return sum(1 for k in wm if k.startswith("mtp.") or ".mtp." in k)
    from safetensors import safe_open

    n = 0
    for sf in sorted(model_path.glob("*.safetensors")):
        with safe_open(sf, framework="pt") as f:
            n += sum(1 for k in f.keys() if k.startswith("mtp.") or ".mtp." in k)  # noqa: SIM118
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local dir")
    p.add_argument("--run", default=DEFAULT_RUN, help="out/<run> workspace name")
    p.add_argument("--force-keep-mtp", action="store_true",
                   help="keep mtp.* even if the weight cross-check finds none (don't — "
                        "this is what produced the phantom-block crash on Ornith)")
    p.add_argument("--skip-template-check", action="store_true")
    a = p.parse_args()

    root = REPO / "out" / a.run
    hf_dir = root / "model_extracted"
    f16 = root / "model-f16.gguf"
    logs = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    ok, detail = _model_available(a.model)
    if not ok:
        log(f"[{a.run}] {a.model} is not available: {detail}")
        log("")
        log("  Qwen3.8-27B has not been published yet. Until it is, exercise the whole")
        log("  chain against a released sibling of the same family:")
        log(f"    scripts/exp060_setup_qwen38.py --model Jackrong/Qwopus3.6-27B-Coder "
            f"--run {a.run}-dryrun")
        return 2
    log(f"[{a.run}] source model: {a.model} ({detail})")

    # --- config + weight cross-check ---------------------------------------------
    src = extract.fetch_model(a.model)
    config = json.loads((src / "config.json").read_text())
    declared = mtp.config_declares_mtp(config)
    weights = _mtp_weight_count(src)
    log(f"[{a.run}] architectures={config.get('architectures')} "
        f"declared_mtp_layers={declared} mtp_weight_tensors={weights}")
    keep_mtp = bool(declared and weights) or a.force_keep_mtp
    if declared and not weights:
        log("  !! config declares an MTP head but the checkpoint ships NO mtp.* weights.")
        log("     keep_mtp stays OFF (a phantom block crashes llama.cpp). To bundle a")
        log("     draft head anyway, graft one after extraction — see")
        log("     scripts/exp045_graft_mtp.py (requires a byte-identical sibling).")
    if not declared and not weights:
        log("  !! no MTP head in this model. The release can still ship, but speculative")
        log("     decoding will need a grafted head (exp045_graft_mtp.py) or a separate")
        log("     draft model.")
    log(f"[{a.run}] keep_mtp={keep_mtp}")

    # --- chat template (cheap, and everything downstream depends on it) -----------
    if not a.skip_template_check:
        from transformers import AutoTokenizer

        from quant_tuner.data.template_check import check_template

        rep = check_template(AutoTokenizer.from_pretrained(a.model, fix_mistral_regex=True),
                             a.model)
        log(rep.summary())
        (root / "template_report.json").write_text(json.dumps(rep.to_dict(), indent=2))
        if not rep.ok:
            log("  !! the chat template cannot carry tool calls; fix it before building "
                "corpora (see scripts/verify_chat_template.py --show)")
            return 1

    # --- extract + convert --------------------------------------------------------
    with phase(f"[{a.run}] extract HF (strip vision, keep_mtp={keep_mtp})"):
        step("extract HF", hf_dir / "config.json",
             lambda: extract.extract_text_lm(
                 source=a.model, output_dir=hf_dir, keep_mtp=keep_mtp))

    cfg_out = json.loads((hf_dir / "config.json").read_text())
    log(f"[{a.run}] extracted config: declared_mtp_layers="
        f"{mtp.config_declares_mtp(cfg_out)} architectures={cfg_out.get('architectures')}")

    with phase(f"[{a.run}] convert -> F16 GGUF"):
        step("convert", f16,
             lambda: convert.hf_to_f16_gguf(hf_dir, f16, log=logs / "convert.log"))

    # --- what actually landed in the GGUF ----------------------------------------
    info = mtp.describe(f16)
    (root / "mtp_report.json").write_text(json.dumps(info, indent=2))
    log(f"[{a.run}] GGUF block_count={info['block_count']} "
        f"mtp_layers={info['mtp_layer_indices']} "
        f"mtp_tensors={info['mtp_tensor_count']}")
    if info["has_mtp"]:
        log(f"[{a.run}] nextn pin for quantization: {info['pin']}")
    elif keep_mtp:
        log("  !! keep_mtp was on but the GGUF has no draft layer — the converter dropped "
            "it. Check logs/convert.log; llama.cpp may not support this arch's MTP yet.")

    log("")
    log(f"=== {a.run} stage 1 complete ===")
    log(f"  HF:   {hf_dir}")
    log(f"  F16:  {f16}")
    log(f"  MTP:  {root / 'mtp_report.json'}")
    log("  Next: scripts/build_universal_corpus.py, then scripts/exp060_quants_qwen38.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

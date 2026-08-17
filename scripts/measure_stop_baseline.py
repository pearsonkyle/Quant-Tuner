"""Measure the SHIPPED model's stop-probe readings, so the in-training probe means something.

``qat/stop_probe.py`` reports P(stop token) at five points in a generation prefix. On its
own that number is uninterpretable: "sentence_period = 0.95" is a catastrophe for
Ternary-Bonsai (whose shipped model reads **0.009** there) and would be unremarkable for a
model that always ends its turn after one sentence. Every conclusion in
`docs/ternary_qat_curriculum.md` is a comparison against that baseline.

The Bonsai/Qwen numbers were measured once and hard-coded. For a new family — gemma-4 —
they have to be measured before the probe can be read, and `PROBE_SPECS[...].vanilla` is
`None` until they are. This script produces them.

Run it on the **untrained** model, forward-only. It needs no GPU (E4B on CPU is ~20 s a
pass, and the probe prompts are short), but will use one if told to.

    python scripts/measure_stop_baseline.py \
        --model google/gemma-4-E4B-it-qat-q4_0-unquantized --device cpu

Then paste the printed `ProbeSpec(...)` line into `qat/stop_probe.py`. It is recorded in
source rather than a data file on purpose: it is a property of a specific released
checkpoint, and a run's log should be readable years later without the file alongside it.

**The control point does not mean the same thing in every family.** For Qwen, after
``</tool_call>`` the model should emit the stop token (0.99995). For gemma-4 the template
hands over to the harness after ``<tool_call|>`` — it emits an opening ``<|tool_response>``
as the generation prompt — so a low reading there is correct, not a regression. Read the
control as "unchanged from this baseline", never as "should be high".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quant_tuner.qat.dialect import detect as detect_dialect
from quant_tuner.qat.stop_probe import CONTROL, DIAGNOSTIC, StopProbe


def _load(model_id: str, device: str, dtype: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[dtype]
    cfg = AutoConfig.from_pretrained(model_id)
    arch = (cfg.architectures or ["AutoModelForCausalLM"])[0]
    # AutoModelForCausalLM silently picks a text-only class on a multimodal checkpoint —
    # for gemma-4 that class's module tree matches none of the checkpoint's tensors. Load
    # the class the config actually declares (same trap as vllm_export's --model-class).
    try:
        import transformers

        klass = getattr(transformers, arch)
    except AttributeError:
        klass = AutoModelForCausalLM
    print(f"[baseline] loading {model_id} as {klass.__name__} ({dtype}, {device})",
          flush=True)
    model = klass.from_pretrained(model_id, dtype=torch_dtype, device_map=device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id or local dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="float32 by default: these are probabilities read to 4 decimals "
                         "and bf16 has 8 mantissa bits")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    dialect = detect_dialect(tok)
    probe = StopProbe.build(tok, dialect)
    print(f"[baseline] dialect={dialect.name} stop_piece={dialect.stop_piece!r} "
          f"stop_id={probe.stop_id}", flush=True)

    model = _load(args.model, args.device, args.dtype)
    probs = probe.measure(model, args.device)

    print("\n=== shipped-model stop probabilities ===")
    for k, v in probs.items():
        tag = ("  <- DIAGNOSTIC" if k == DIAGNOSTIC else
               "  <- CONTROL" if k == CONTROL else "")
        print(f"  {k:18s} {v:.5f}{tag}")
    print(f"\npaste into qat/stop_probe.py PROBE_SPECS[{dialect.name!r}]:")
    print(f"    vanilla=({probs[DIAGNOSTIC]:.4g}, {probs[CONTROL]:.5g}),")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"model": args.model, "dialect": dialect.name,
             "stop_id": probe.stop_id, "dtype": args.dtype, "probs": probs}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

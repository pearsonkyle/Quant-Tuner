"""exp-057 step 0: verify the STE ternary wrapping reproduces the real model.

Loads prism-ml/Ternary-Bonsai-8B-unpacked (a plain trainable Qwen3ForCausalLM whose
fp16 weights are the shipped ternary values), records reference logits, then swaps
every nn.Linear for a TernaryLinear and re-runs. Because the shipped weights are
exactly ``s_g * c`` per 128-group and TWN recovers ``s_g`` exactly, the wrapped
logits must match the originals to float tolerance — proving the fine-tune starts
from the real model (no step-0 drift). Also prints peak MPS memory for one forward.

    PYTHONPATH=src .venv/bin/python scripts/exp057_qat_step0.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from quant_tuner.qat.ternary import TernaryLinear

MODEL = REPO / "out" / "exp-057" / "model"


def swap_linears(module: torch.nn.Module, *, trainable: bool = False, skip: tuple[str, ...] = ("lm_head",)) -> int:
    """Recursively replace nn.Linear children with TernaryLinear. Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and name not in skip:
            if child.in_features % 128 != 0:
                continue  # Q2_0 needs /128; skip anything odd (none in Qwen3-8B)
            setattr(module, name, TernaryLinear(child, trainable=trainable))
            n += 1
        else:
            n += swap_linears(child, trainable=trainable, skip=skip)
    return n


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = ap.parse_args()
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[step0] loading {MODEL} on {dev} ({args.dtype})...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt)
    model.eval().to(dev)
    print(f"[step0] loaded in {time.time()-t0:.0f}s | params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    ids = tok("def fibonacci(n):\n    if n < 2:\n        return n\n", return_tensors="pt").input_ids.to(dev)

    if dev == "mps":
        torch.mps.empty_cache()
    with torch.no_grad():
        ref = model(ids).logits.float().cpu()
    if dev == "mps":
        peak = torch.mps.current_allocated_memory() / 1024**3
        print(f"[step0] forward peak MPS alloc: {peak:.1f} GiB", flush=True)

    n = swap_linears(model, trainable=False)
    print(f"[step0] wrapped {n} linears with TernaryLinear", flush=True)

    with torch.no_grad():
        got = model(ids).logits.float().cpu()

    max_abs = (got - ref).abs().max().item()
    # argmax agreement over the sequence
    same_argmax = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"[step0] max|logit_wrapped - logit_ref| = {max_abs:.4e}")
    print(f"[step0] top-1 token agreement: {same_argmax*100:.2f}%")
    ok = max_abs < 1e-2 and same_argmax > 0.999
    print(f"[step0] STEP-0 REPRODUCTION: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

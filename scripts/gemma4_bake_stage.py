"""Bake a stage's trained latents into an HF model dir, for the next stage to start from.

A progressive schedule needs stage N+1 to see stage N's *result*. ``--resume`` cannot do
it: it requires the checkpoint to carry every trainable name, and each stage trains a
different layer set, so resuming stage 1's checkpoint into a stage-2 run exits on a
layer-set mismatch. Baking sidesteps that — the ternarized weights go into an ordinary
checkpoint and the next stage points ``--model-dir`` at it.

What comes out is on the ternary grid, so the next stage's ``wrap_model`` proves those
layers exact and leaves them frozen and unwrapped (a TernaryLinear there would be a
bit-exact no-op costing ~5 W-sized transients per forward). Storage is **bf16** to match
the base repo: a fp32 dir is 31.8 GB per stage and seven of them do not fit. bf16 cannot
represent an fp16 TWN scale exactly, so the next stage's exactness check may miss by a
rounding and re-wrap the layer — which re-derives the same codes and a scale equal to
within bf16, and is reported rather than silent (``frozen linear off-grid -> wrapping``).

The flip report is the point of the printout: a ternary model only learns by flipping
codes, and a stage that changed ~0% of them lowered its loss by drifting scales.

Usage::

    python scripts/gemma4_bake_stage.py --ckpt out/gemma4-ternary/stage1/ckpt-final.pt \
        --out out/gemma4-ternary/stage1/baked
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quant_tuner.qat.ternary import ternarize_group

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-q4_0-unquantized"


def bake(ckpt: Path, model_dir: str, out: Path, *, dtype=torch.bfloat16) -> dict:
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    print(f"[bake] loading base {model_dir}", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, device_map="cpu")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    latents = blob["latents"]
    sd = dict(model.named_parameters())

    stats: dict[str, dict] = {}
    n_flips = n_codes = 0
    with torch.no_grad():
        for k, v in latents.items():
            base_key = k.replace(".linear.weight", ".weight")
            if base_key not in sd:
                raise KeyError(f"checkpoint latent {k} has no weight {base_key} in "
                               f"{model_dir} — wrong base model for this stage")
            W0 = sd[base_key].data
            old_codes, _, _ = ternarize_group(W0)
            new_codes, new_scale, w_hat = ternarize_group(v.to(torch.float32))
            flips = int((new_codes != old_codes).sum())
            n_flips += flips
            n_codes += new_codes.numel()
            stats[base_key] = {"flips": flips, "numel": int(new_codes.numel()),
                               "flip_pct": 100.0 * flips / max(1, new_codes.numel()),
                               "zero_frac": float((new_codes == 0).float().mean())}
            W0.copy_(w_hat)

    pct = 100.0 * n_flips / max(1, n_codes)
    print(f"[bake] {len(latents)} tensors, {n_flips:,}/{n_codes:,} codes flipped "
          f"({pct:.3f}%) vs the base model", flush=True)
    if pct < 0.01:
        print("[bake] WARNING: essentially no codes moved. A ternary model learns only "
              "by flipping codes — this stage drifted scales and learned nothing, "
              "whatever the loss curve did.", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    print(f"[bake] saving {dtype} -> {out}", flush=True)
    model.to(dtype).save_pretrained(out)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(out)
    meta = {"base": model_dir, "ckpt": str(ckpt), "step": blob.get("step"),
            "corpus_fingerprint": blob.get("corpus_fingerprint"),
            "n_tensors": len(latents), "flip_pct": pct, "dtype": str(dtype),
            "per_tensor": stats}
    (out / "quant_tuner_bake.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL,
                    help="base this stage trained FROM (the previous stage's bake, "
                         "after stage 1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--threads", type=int, default=192)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    bake(args.ckpt, args.model_dir, args.out,
         dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype])


if __name__ == "__main__":
    main()

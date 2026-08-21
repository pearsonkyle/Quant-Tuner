"""Output-space damage of ONE ternarization stage, before and after training.

``gemma4_layer_damage.py`` answers "how much does ternarizing this group hurt, with no
training". This answers the question a progressive schedule actually turns on:

> Does QAT recover a stage's damage before the next stage compounds on it?

Both numbers come from the same probe in the same process, against the same dense
reference — the point is the *difference*, and a difference read across two runs of two
scripts is not a difference, it is two numbers.

Three measurements per invocation:

``dense``
    The reference against itself. Must be ~0; anything else means the probe is broken
    and the other two numbers are meaningless.
``untrained``
    The stage's layers ternarized straight from the shipped weights. This is NOT the
    matching row of ``layer_damage.json``: that row ternarized every linear in the
    layer, while a stage keeps ``--dense-kind`` tensors dense, and on gemma-4-E4B the
    kind held dense (``down_proj``, solo KLD 1.199) is the most damaging one there is.
    The comparable baseline has to be measured under the stage's own configuration.
``trained``
    The same wrapping, with the checkpoint's latents loaded into it.

The model is wrapped with the trainer's own :func:`wrap_model`, not a re-derived
ternarization, so what is measured is exactly what training deploys — including which
linears are skipped (``in_features % 128``) and which kinds are held dense. A
re-implementation that agreed to 99% would produce a plausible number that answers a
slightly different question.

Usage::

    python scripts/gemma4_stage_damage.py --ternary-layers 0,1,2,3,7,8 \
        --dense-kind down_proj --ckpt out/gemma4-ternary/stage1/ckpt-final.pt \
        --out out/gemma4-ternary/stage1/stage_damage.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Works whether this file is run as a script (sys.path[0] is scripts/) or imported as
# scripts.gemma4_stage_damage by the tests (sys.path[0] is the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The probe is imported, never re-implemented: same windows, same reductions, same
# fp16 storage as the no-training scan, so the numbers sit in one column.
from gemma4_layer_damage import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SFT,
    _logits,
    build_windows,
    compare,
)

from quant_tuner.qat.ternary import TernaryLinear  # noqa: E402
from quant_tuner.qat.train import (  # noqa: E402
    decoder_layers,
    parse_layers,
    wrap_model,
)


def latent_modules(model) -> dict[str, torch.nn.Parameter]:
    """Every wrapped ternary latent, keyed by its ``….linear.weight`` parameter name.

    Selected by MODULE TYPE, never by name. gemma-4's vision and audio encoders have
    submodules literally called ``linear``, so ``name.endswith(".linear.weight")`` finds
    280 parameters where the wrapping produced 48 — the same collision that once handed
    167.8 M params of frozen tower to the optimizer.
    """
    return {f"{n}.linear.weight": m.linear.weight
            for n, m in model.named_modules() if isinstance(m, TernaryLinear)}


def load_latents(model, ckpt: Path) -> tuple[int, int]:
    """Copy a checkpoint's latents into the wrapped model. Returns (loaded, expected).

    Refuses a partial load. A checkpoint whose keys half-match the wrapping silently
    leaves the rest of the stage at its shipped weights, and the resulting damage
    number would read as recovery.
    """
    blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    latent_sd = blob["latents"]
    live = latent_modules(model)
    missing = sorted(set(live) - set(latent_sd))
    extra = sorted(set(latent_sd) - set(live))
    if missing or extra:
        raise ValueError(
            f"checkpoint does not match the wrapping: {len(missing)} wrapped latents "
            f"absent from the checkpoint (e.g. {missing[:2]}), {len(extra)} checkpoint "
            f"tensors with no wrapped latent (e.g. {extra[:2]}). Check --ternary-layers "
            f"and --dense-kind match the run that produced it.")
    with torch.no_grad():
        for n, p in live.items():
            p.copy_(latent_sd[n].to(p.dtype))
    return len(live), len(latent_sd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    # Defaults reproduce layer_damage.json's probe exactly -- changing one makes the
    # result incomparable to the schedule it is supposed to be read against.
    ap.add_argument("--split", default="test")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--ternary-layers", required=True, help="e.g. 0,1,2,3,7,8")
    ap.add_argument("--dense-kind", action="append", default=[],
                    help="tensor kind held dense inside every ternarized layer")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="trained latents; omit to measure the untrained stage only")
    ap.add_argument("--threads", type=int, default=192)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.model)
    windows = build_windows(tok, args.sft, split=args.split,
                            window=args.window, n_windows=args.windows)

    print(f"[load] {args.model} fp32 on cpu", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        ref = _logits(model, windows)
    print(f"[ref] dense reference logits in {time.time()-t0:.0f}s", flush=True)

    rows: dict[str, dict] = {}
    with torch.no_grad():
        rows["dense"] = compare(ref, _logits(model, windows), windows)
    print(f"[dense] self-check kld={rows['dense']['kld']:.2e} "
          f"(must be ~0; a nonzero here invalidates everything below)", flush=True)

    idx = sorted(parse_layers(args.ternary_layers, len(decoder_layers(model))))
    wrap_model(model, 0, layer_spec=args.ternary_layers,
               ternary_spec=args.ternary_layers, dense_kinds=tuple(args.dense_kind))
    live = latent_modules(model)
    print(f"[wrap] layers {idx} -> {len(live)} ternary latents, "
          f"dense kinds {args.dense_kind or 'none'}", flush=True)

    with torch.no_grad():
        rows["untrained"] = compare(ref, _logits(model, windows), windows)
    print(f"[untrained] kld={rows['untrained']['kld']:.4f} "
          f"top1={rows['untrained']['top1_agree']:.3f} "
          f"ppl={rows['untrained']['ppl']:.2f}", flush=True)

    if args.ckpt:
        n, _ = load_latents(model, args.ckpt)
        print(f"[trained] loaded {n} latents from {args.ckpt}", flush=True)
        with torch.no_grad():
            rows["trained"] = compare(ref, _logits(model, windows), windows)
        u, t = rows["untrained"]["kld"], rows["trained"]["kld"]
        rows["recovered_frac"] = {"kld": (u - t) / u if u else float("nan")}
        print(f"[trained] kld={t:.4f} top1={rows['trained']['top1_agree']:.3f} "
              f"ppl={rows['trained']['ppl']:.2f}  |  recovered "
              f"{100*(u-t)/u:.1f}% of the stage's damage", flush=True)

    blob = {"model": args.model, "split": args.split, "window": args.window,
            "windows": args.windows, "ternary_layers": idx,
            "dense_kinds": args.dense_kind, "n_latents": len(live),
            "ckpt": str(args.ckpt) if args.ckpt else None, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

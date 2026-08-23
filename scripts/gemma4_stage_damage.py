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

from quant_tuner.qat.stop_probe import StopProbe, format_line  # noqa: E402
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
    """Copy a checkpoint's trained tensors into the wrapped model. Returns (ternary, dense).

    A stage's checkpoint holds TWO kinds of tensor and both were trained:

    * ``….linear.weight`` — the ternary latents, one per wrapped linear.
    * ``….weight`` — the ``--dense-kind`` tensors held OFF the grid inside a trainable
      layer. Letting them adapt to their ternarized neighbours is most of why a partial
      schedule beats all-at-once, so measuring the stage without them measures a model
      that was never trained. (Measured: 16 latents + 2 dense down_proj for a 2-layer
      stage.)

    Refuses a partial load in either direction. A checkpoint whose keys half-match the
    wrapping silently leaves the rest of the stage at its shipped weights, and the
    resulting damage number reads as recovery.
    """
    blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    ck = blob["latents"]
    live = latent_modules(model)
    dense_live = {n: q for n, q in model.named_parameters()
                  if n in ck and not n.endswith(".linear.weight")}
    known = set(live) | set(dense_live)
    missing = sorted(set(live) - set(ck))
    extra = sorted(set(ck) - known)
    if missing or extra:
        raise ValueError(
            f"checkpoint does not match the wrapping: {len(missing)} wrapped latents "
            f"absent from the checkpoint (e.g. {missing[:2]}), {len(extra)} checkpoint "
            f"tensors with no home in the model (e.g. {extra[:2]}). Check "
            f"--ternary-layers and --dense-kind match the run that produced it.")
    with torch.no_grad():
        for n, q in {**live, **dense_live}.items():
            q.copy_(ck[n].to(q.dtype))
    return len(live), len(dense_live)


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
    ap.add_argument("--ref-ckpt", type=Path, default=None,
                    help="measure against a DENSE fine-tune of the same layers instead "
                         "of the shipped model — isolates ternarization from training")
    ap.add_argument("--probe", action="store_true",
                    help="also run the stop probe at each stage — separates "
                         "'ternarization broke termination' from 'training did'")
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

    # The reference decides what "damage" MEANS, and the default answer is incomplete.
    # Against the shipped model, KLD sums two things a fine-tune does at once:
    # ternarization damage failing to recover, and the model legitimately learning the
    # training distribution. Measured: a CE-only ternary arm scored 0.3866 where the
    # untrained ternarization scored 0.0762 — training moved it 5x further than
    # ternarizing did, and no part of that number says which part was the quantization.
    # --ref-ckpt makes the reference a DENSE fine-tune of the same layers on the same
    # data, so what is left is the ternarization alone.
    if args.ref_ckpt:
        # The reference is a DENSE fine-tune, so its tensors are plain `….weight` and go
        # straight into the unwrapped model — no wrap_model call, which would otherwise
        # have to use the candidate's --dense-kind and then refuse the load. The shipped
        # values are kept and restored, so "untrained" keeps meaning what it means in
        # every other invocation: the ternarized SHIPPED weights.
        rc = torch.load(args.ref_ckpt, map_location="cpu", weights_only=False, mmap=True)
        ref_sd = rc["latents"]
        if any(k.endswith(".linear.weight") for k in ref_sd):
            raise ValueError(f"--ref-ckpt {args.ref_ckpt} contains ternary latents; the "
                             f"reference must be a DENSE fine-tune of the same layers")
        live = dict(model.named_parameters())
        missing = sorted(set(ref_sd) - set(live))
        if missing:
            raise ValueError(f"--ref-ckpt has {len(missing)} tensors absent from the "
                             f"model (e.g. {missing[:2]})")
        shipped = {k: live[k].detach().clone() for k in ref_sd}
        with torch.no_grad():
            for k, v in ref_sd.items():
                live[k].copy_(v.to(live[k].dtype))
        print(f"[ref] dense fine-tune reference: {len(ref_sd)} tensors from "
              f"{args.ref_ckpt}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        ref = _logits(model, windows)
    print(f"[ref] reference logits in {time.time()-t0:.0f}s "
          f"({'dense fine-tune' if args.ref_ckpt else 'shipped model'})", flush=True)

    probe = StopProbe.build(tok) if args.probe else None

    def run_probe(tag: str) -> dict | None:
        """The probe is the ONLY instrument that sees the termination failure; masked-CE
        cannot (sft32k's validation went flat for 225 steps while its diagnostic climbed
        to 0.97). Running it at each stage attributes a control collapse to the
        TERNARIZATION or to the TRAINING, which the in-training series cannot do on its
        own — its first reading is already 25 steps deep."""
        if probe is None:
            return None
        pr = probe.measure(model, "cpu")
        print(f"[probe/{tag}] {format_line(pr, probe.dialect)}", flush=True)
        return pr

    if args.ref_ckpt:
        with torch.no_grad():
            for k, v in shipped.items():
                dict(model.named_parameters())[k].copy_(v)
        del shipped
        print("[ref] shipped weights restored; 'untrained' below is the ternarized "
              "SHIPPED model, as in every other run", flush=True)

    rows: dict[str, dict] = {}
    probes: dict[str, dict] = {}
    # The reference scored against itself: KLD is 0 by construction, but the NLL/ppl is
    # the reference's OWN quality on these windows, and without it the file cannot answer
    # the question its KLD column invites — "is the ternary arm worse, or just different?"
    # Without it the file reports three candidate ppls and no yardstick, which is how the
    # first --ref-ckpt run ended up uninterpretable.
    with torch.no_grad():
        rows["reference"] = compare(ref, ref, windows)
    rows["dense"] = compare(ref, _logits(model, windows), windows)
    probes["dense"] = run_probe("dense")
    if args.ref_ckpt:
        # NOT a self-check here: the reference is the dense fine-tune, so this row is
        # KLD(dense-ft ‖ shipped) — how far the dense arm travelled, which is the floor
        # the ternary arms are read against. It SHOULD be large.
        print(f"[shipped] kld={rows['dense']['kld']:.4f} vs the dense fine-tune — this "
              f"is the training drift itself, not a self-check", flush=True)
    else:
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
    probes["untrained"] = run_probe("untrained")
    print(f"[untrained] kld={rows['untrained']['kld']:.4f} "
          f"top1={rows['untrained']['top1_agree']:.3f} "
          f"ppl={rows['untrained']['ppl']:.2f}", flush=True)

    if args.ckpt:
        n_t, n_d = load_latents(model, args.ckpt)
        print(f"[trained] loaded {n_t} ternary latents + {n_d} trained dense tensors "
              f"from {args.ckpt}", flush=True)
        with torch.no_grad():
            rows["trained"] = compare(ref, _logits(model, windows), windows)
        probes["trained"] = run_probe("trained")
        u, t = rows["untrained"]["kld"], rows["trained"]["kld"]
        # u is EXACTLY 0 for a dense control arm: nothing was ternarized, so the
        # "untrained" model is the reference and there is no damage to recover a
        # fraction of. Report the absolute drift instead of dividing by zero — that arm
        # exists precisely to say how much of a ternary arm's KLD is just fine-tuning.
        frac = (u - t) / u if u > 1e-9 else None
        if args.ref_ckpt:
            # The two rows do not sit on the same footing against a fine-tuned reference,
            # so their ratio is not a recovery fraction and printing one invites the
            # reader to quote it. `trained` is (trained ternary ‖ trained dense): both saw
            # the same data, so the fine-tuning drift largely cancels and what is left is
            # close to ternarization alone. `untrained` is (ternarized SHIPPED ‖ trained
            # dense), which still carries the whole drift — the reference moved and this
            # candidate did not. The honest before/after pairs each candidate with the
            # dense model AT ITS OWN point in training: use `stage_damage_untrained.json`
            # (ternarized shipped ‖ shipped) as the "before" and this file's `trained` as
            # the "after".
            frac = None
            rows["recovered_frac"] = {"kld": None, "why": "not defined against a "
                                      "fine-tuned reference; see stage_damage_untrained"}
        else:
            rows["recovered_frac"] = {"kld": frac}
        tail = (f"recovered {100 * frac:.1f}% of the stage's damage" if frac is not None
                else "recovery fraction undefined here — 'untrained' still carries the "
                     "full training drift against this reference; pair the 'before' with "
                     "the SHIPPED reference instead" if args.ref_ckpt
                else "no ternarization in this arm — its KLD is pure fine-tuning drift, "
                     "the floor to read the ternary arms against")
        print(f"[trained] kld={t:.4f} top1={rows['trained']['top1_agree']:.3f} "
              f"ppl={rows['trained']['ppl']:.2f}  |  {tail}", flush=True)

    blob = {"model": args.model, "split": args.split, "window": args.window,
            "windows": args.windows, "ternary_layers": idx,
            "dense_kinds": args.dense_kind, "n_latents": len(live),
            "ckpt": str(args.ckpt) if args.ckpt else None,
            "ref_ckpt": str(args.ref_ckpt) if args.ref_ckpt else None, "rows": rows,
            "probes": {k: v for k, v in probes.items() if v}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

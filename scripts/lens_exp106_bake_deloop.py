#!/usr/bin/env python3
"""exp-106: bake a jlens-discovered loop direction out of a GGUF (corrected quant).

Demonstrates the full "modify a GGUF from a jlens-driven edit" path:

  1. reproduce a strict repetition loop on the quant (the 2-bit Ornith model
     loops verbatim on certain prompts);
  2. `loop_direction` extracts the residual-space direction that separates the
     looping region from the run-up (saved as direction.npz);
  3. `orthogonalize_layers` projects that direction out of the FP16
     residual-writing tensors (attn_output + ffn_down, per-expert for MoE; the
     skip path is untouched), then `llama-quantize` requantizes to a real GGUF
     with the existing imatrix — never in-place block editing;
  4. a matched CONTROL is requantized from the *unedited* FP16 with the same
     recipe, so any difference is attributable to the orthogonalization alone;
  5. verify: run loop-prone + normal prompts on baked vs control and compare
     loop incidence and unique-token ratio.

This is an abliteration-style edit targeted by an interpretability signal rather
than a hand-picked refusal direction. Whether one direction meaningfully reduces
looping is an empirical question the verification answers honestly.

Usage::

    PYTHONPATH=src python scripts/lens_exp106_bake_deloop.py \
        --f16       out/exp-051/awq_iq2_m/model-f16-awq.gguf \
        --quant     out/exp-051/awq_iq2_m/Ornith-1.0-9B-IQ2_M-awq.gguf \
        --imatrix   out/exp-051/awq_iq2_m/imatrix-awq-hybrid_custom.gguf \
        --direction out/exp-051/lens/loop_direction.npz \
        --quant-type IQ2_M --out-dir out/exp-051/lens/bake
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.lens.backend import running_jlens_server  # noqa: E402
from quant_tuner.lens.bake import Direction, orthogonalize_layers  # noqa: E402
from quant_tuner.lens.loops import detect_repetition  # noqa: E402

# Prompts: two are loop-prone (verbatim repetition), two are normal generation.
VERIFY_PROMPTS = [
    ("loop:json", "Output a JSON array of 500 identical objects {\"status\":\"checking\"} with no other text."),
    ("loop:enum", "Print the numbers 1 to 300 as a comma separated list, then keep counting forever."),
    ("task:plan", "You are a coding agent. Briefly outline a plan to fix a Duration serialization "
                  "TypeError in msrest, then state the exact edit you would make."),
    ("task:explain", "Explain in three sentences why 2-bit quantized models sometimes repeat themselves."),
]


def loop_stats(client, text, n_predict=360):
    msgs = [{"role": "user", "content": text}]
    prompt = client.apply_template(msgs, add_assistant=True)
    tokens = client.tokenize(prompt, add_special=True, parse_special=True)
    fr = client.forward(tokens, capture=False, n_predict=n_predict, sampling={"greedy": True})
    gen = [g["token"] for g in fr.generated]
    loop = detect_repetition(gen, min_repeats=4, min_span=30)
    return {
        "looped": loop is not None,
        "loop": (f"p{loop.period}x{loop.n_repeats}" if loop else None),
        "n_gen": len(gen),
        "unique_ratio": round(len(set(gen)) / max(len(gen), 1), 3),
        "sample": fr.generated_text.strip().replace("\n", " ")[-120:],
    }


def verify(model, ngl):
    out = {}
    with running_jlens_server(Path(model), ctx=4096, ngl=ngl) as client:
        for name, text in VERIFY_PROMPTS:
            out[name] = loop_stats(client, text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f16", type=Path, required=True)
    ap.add_argument("--quant", type=Path, required=True, help="original quant (for reference verify)")
    ap.add_argument("--imatrix", type=Path, required=True)
    ap.add_argument("--direction", type=Path, required=True)
    ap.add_argument("--quant-type", default="IQ2_M")
    ap.add_argument("--out-dir", type=Path, default=_REPO / "out" / "exp-051" / "lens" / "bake")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--skip-build", action="store_true", help="reuse existing baked/control GGUFs")
    args = ap.parse_args()

    from quant_tuner.quantize import gguf as gguf_quant

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    direction = Direction.load(args.direction)
    baked = out / f"{args.quant_type}-deloop-baked.gguf"
    control = out / f"{args.quant_type}-control.gguf"
    edited_f16 = out / "edited-f16.gguf"

    if not args.skip_build:
        print(f"[exp-106] orthogonalizing {len(direction.layers)} layers of {args.f16.name} ...", flush=True)
        orthogonalize_layers(args.f16, direction, edited_f16, layers=direction.layers)
        print(f"[exp-106] requantizing baked -> {baked.name} ({args.quant_type}) ...", flush=True)
        gguf_quant.quantize(edited_f16, baked, args.quant_type, imatrix=args.imatrix,
                            log=out / "baked-quantize.log")
        print(f"[exp-106] requantizing control -> {control.name} (same recipe, unedited) ...", flush=True)
        gguf_quant.quantize(args.f16, control, args.quant_type, imatrix=args.imatrix,
                            log=out / "control-quantize.log")
        edited_f16.unlink(missing_ok=True)  # 17G scratch; drop once quantized

    print("[exp-106] verifying loop incidence ...", flush=True)
    results = {"control": verify(control, args.ngl), "baked": verify(baked, args.ngl)}

    (out / "exp106.json").write_text(json.dumps(results, indent=1))
    print(f"\n{'prompt':>14} | {'control':>26} | {'baked':>26}")
    for name, _ in VERIFY_PROMPTS:
        c, b = results["control"][name], results["baked"][name]
        cs = f"loop={c['loop'] or '-'} uniq={c['unique_ratio']}"
        bs = f"loop={b['loop'] or '-'} uniq={b['unique_ratio']}"
        print(f"{name:>14} | {cs:>26} | {bs:>26}")
    c_loops = sum(r["looped"] for r in results["control"].values())
    b_loops = sum(r["looped"] for r in results["baked"].values())
    print(f"\ncontrol loops: {c_loops}/{len(VERIFY_PROMPTS)}   baked loops: {b_loops}/{len(VERIFY_PROMPTS)}")
    print(f"[exp-106] wrote {out}/exp106.json ; baked GGUF: {baked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

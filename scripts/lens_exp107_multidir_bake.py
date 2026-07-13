#!/usr/bin/env python3
"""exp-107: multi-direction bake — combine loop + explore-vs-act directions.

Extends the single-direction bake (exp-106) two ways the writeup asked for:

  * **combine directions**: `orthogonalize_subspace` removes the *span* of
    several directions at once (SVD basis, so correlated directions don't
    over-project), letting us fold a loop direction and an explore-vs-act
    direction into one edit;
  * **stay realistic**: the explore-vs-act directions are contrastive
    mean-difference vectors from the *real* tool-call holdout — mean residual at
    "about to call an explore tool (read/grep/ls)" minus "about to call an act
    tool (edit/write)" — and the eval uses the real 25-session holdout.

Ablating the explore-vs-act axis removes the model's learned lean toward
"inspect more" over "edit"; ablating the loop direction removes a verbatim-loop
attractor. A matched CONTROL is requantized from the *unedited* FP16 with the
same recipe, so any difference is the orthogonalization alone.

Direction bundle is read from ``--dirs-dir`` (``*.npz`` written by the
collectors); typically 4 explore-act (L15/19/23/27) + >=1 loop.

Usage::

    PYTHONPATH=src python scripts/lens_exp107_multidir_bake.py \
        --f16 out/exp-051/awq_iq2_m/model-f16-awq.gguf \
        --imatrix out/exp-051/awq_iq2_m/imatrix-awq-hybrid_custom.gguf \
        --dirs-dir out/exp-051/lens/dirs --lens out/exp-051/lens/Ornith-1.0-9B.lens.gguf \
        --quant-type IQ2_M --out-dir out/exp-051/lens/bake_multi [--toolcall]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from quant_tuner.lens.backend import running_jlens_server  # noqa: E402
from quant_tuner.lens.bake import Direction, orthogonalize_subspace  # noqa: E402
from quant_tuner.lens.loops import detect_repetition  # noqa: E402

VERIFY_PROMPTS = [
    ("loop:json", "Output a JSON array of 500 identical objects {\"status\":\"checking\"} with no other text."),
    ("loop:enum", "Print the numbers 1 to 300 as a comma separated list, then keep counting forever."),
    ("task:plan", "You are a coding agent. Briefly outline a plan to fix a Duration serialization "
                  "TypeError in msrest, then state the exact edit you would make."),
    ("task:explain", "Explain in three sentences why 2-bit quantized models sometimes repeat themselves."),
]
# explore vs act probe tokens (exp-105) — measured at the commit decision
EXPLORE_TOK = [" Let", " inspect", " look", " understand"]
ACT_TOK = [" edit", " fix", " modify", " change", " apply"]
COMMIT_MSGS = [
    {"role": "system", "content": "You are a coding agent working in /testbed. Use the bash tool to "
     "inspect and edit files. When you have found the bug, EDIT the source to fix it, then stop."},
    {"role": "user", "content": "Issue: Duration serialization fails with 'TypeError: Expecting a string'. "
     "You have already inspected serialization.py and know Duration objects are passed to a method "
     "expecting a string. What is your next step?"},
]


def loop_stats(client, text, n_predict=360):
    prompt = client.apply_template([{"role": "user", "content": text}], add_assistant=True)
    tokens = client.tokenize(prompt, add_special=True, parse_special=True)
    fr = client.forward(tokens, capture=False, n_predict=n_predict, sampling={"greedy": True})
    gen = [g["token"] for g in fr.generated]
    loop = detect_repetition(gen, min_repeats=4, min_span=30)
    return {"looped": loop is not None, "loop": (f"p{loop.period}x{loop.n_repeats}" if loop else None),
            "unique_ratio": round(len(set(gen)) / max(len(gen), 1), 3)}


def commit_ranks(client, readout):
    from quant_tuner.lens.readout import LensReadout

    prompt = client.apply_template(COMMIT_MSGS, add_assistant=True)
    tokens = client.tokenize(prompt, add_special=True, parse_special=True)
    n_final = readout.weights.n_layers - 1
    fr = client.forward(tokens, capture_layers=[n_final], dtype="f32")
    logits = readout.lens_logits(fr.activations[n_final][-1][None, :], n_final, use_lens=False)
    out = {}
    for w in EXPLORE_TOK + ACT_TOK:
        t = client.tokenize(w, add_special=False, parse_special=False)
        if t:
            out[w] = int(LensReadout.ranks_of(logits, np.array([t[0]], dtype=np.int64))[0, 0])
    return out


def verify(model, lens_path, ngl):
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    lens = JacobianLensGGUF.load(str(lens_path))
    readout = LensReadout(ReadoutWeights.from_gguf(str(model)), lens)
    res = {"loops": {}, "commit_ranks": None}
    with running_jlens_server(Path(model), ctx=4096, ngl=ngl) as client:
        for name, text in VERIFY_PROMPTS:
            res["loops"][name] = loop_stats(client, text)
        res["commit_ranks"] = commit_ranks(client, readout)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f16", type=Path, required=True)
    ap.add_argument("--imatrix", type=Path, required=True)
    ap.add_argument("--dirs-dir", type=Path, required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--quant-type", default="IQ2_M")
    ap.add_argument("--out-dir", type=Path, default=_REPO / "out" / "exp-051" / "lens" / "bake_multi")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--toolcall", action="store_true", help="also run the real tool-call holdout eval")
    ap.add_argument("--holdout", type=Path, default=_REPO / "out" / "exp-002" / "toolcall_holdout.jsonl")
    args = ap.parse_args()

    from quant_tuner.quantize import gguf as gguf_quant

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Direction.load(p) for p in sorted(args.dirs_dir.glob("*.npz"))]
    if not dirs:
        raise SystemExit(f"no directions in {args.dirs_dir}")
    kinds = {}
    for p, d in zip(sorted(args.dirs_dir.glob("*.npz")), dirs, strict=True):
        kinds[p.stem] = d.method
    print(f"[exp-107] {len(dirs)} directions: {kinds}")

    baked = out / f"{args.quant_type}-multidir-baked.gguf"
    control = out / f"{args.quant_type}-control.gguf"
    edited = out / "edited-f16.gguf"
    if not args.skip_build:
        print(f"[exp-107] orthogonalizing subspace of {len(dirs)} directions ...", flush=True)
        orthogonalize_subspace(args.f16, dirs, edited, layers=list(range(32)))
        print(f"[exp-107] requant baked -> {baked.name}", flush=True)
        gguf_quant.quantize(edited, baked, args.quant_type, imatrix=args.imatrix, log=out / "baked.log")
        print(f"[exp-107] requant control -> {control.name}", flush=True)
        gguf_quant.quantize(args.f16, control, args.quant_type, imatrix=args.imatrix, log=out / "control.log")
        edited.unlink(missing_ok=True)

    print("[exp-107] verifying (loops + commit-decision ranks) ...", flush=True)
    results = {"control": verify(control, args.lens, args.ngl),
               "baked": verify(baked, args.lens, args.ngl)}

    if args.toolcall:
        from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval

        print("[exp-107] running real tool-call holdout eval (llama-server) ...", flush=True)
        for label, path in (("control", control), ("baked", baked)):
            s = run_toolcall_eval(args.holdout, model_path=path, sampling=Sampling())
            results[label]["toolcall"] = {
                "tool_selection_acc": round(getattr(s, "tool_selection_acc", float("nan")), 3),
                "param_acc_mean": round(getattr(s, "param_acc_mean", float("nan")), 3),
                "schema_valid_rate": round(getattr(s, "schema_valid_rate", float("nan")), 3),
            }

    (out / "exp107.json").write_text(json.dumps(results, indent=1))
    _report(results, args.quant_type)
    print(f"\n[exp-107] wrote {out}/exp107.json ; baked GGUF: {baked}")
    return 0


def _report(results, qt):
    print(f"\n=== loop incidence ({qt}) ===")
    print(f"{'prompt':>14} | {'control':>22} | {'baked':>22}")
    for name, _ in VERIFY_PROMPTS:
        c, b = results["control"]["loops"][name], results["baked"]["loops"][name]
        cs = f"loop={c['loop'] or '-'} uniq={c['unique_ratio']:.3f}"
        bs = f"loop={b['loop'] or '-'} uniq={b['unique_ratio']:.3f}"
        print(f"{name:>14} | {cs:>22} | {bs:>22}")
    c_l = sum(v["looped"] for v in results["control"]["loops"].values())
    b_l = sum(v["looped"] for v in results["baked"]["loops"].values())
    print(f"control loops: {c_l}/{len(VERIFY_PROMPTS)}   baked loops: {b_l}/{len(VERIFY_PROMPTS)}")

    print("\n=== commit-decision token ranks (lower = more likely to say it) ===")
    cr, br = results["control"]["commit_ranks"], results["baked"]["commit_ranks"]
    print(f"{'token':>12} | {'control':>9} | {'baked':>9} | delta")
    for w in EXPLORE_TOK + ACT_TOK:
        a, b = cr.get(w, -1), br.get(w, -1)
        print(f"{w:>12} | {a:>9} | {b:>9} | {b - a:+d}")

    if "toolcall" in results["control"]:
        print("\n=== tool-call holdout accuracy ===")
        for label in ("control", "baked"):
            print(f"  {label}: {results[label]['toolcall']}")


if __name__ == "__main__":
    raise SystemExit(main())

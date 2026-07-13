#!/usr/bin/env python3
"""exp-102: infinite-loop autopsy + intervention (Jacobian lens).

Replays a looping generation (from a swebench ``.traj.json`` or a prompt file)
on a gemma-4-31B quant and, if it loops, measures the per-position *collapse
depth* — the earliest layer at which the lens top-1 already commits to the loop
token — against the FP16 reference. Then sweeps ablations of the loop direction
across the late-third layers to find one that breaks the loop, saving the
winning ``direction.npz`` for ``quant-tuner lens bake``.

Usage::

    PYTHONPATH=src python scripts/lens_exp102_loop_autopsy.py \
        --model uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf \
        --lens out/lens/gemma-4-31B.lens.gguf \
        --reference out/exp-044/vanilla/model-f16.gguf \
        --traj out/swe-rebench/smoke/trajectories/<model>/<instance>.traj.json \
        --sweep --out out/lens/exp102
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.lens.backend import running_jlens_server  # noqa: E402
from quant_tuner.lens.capture import load_run  # noqa: E402
from quant_tuner.lens.loops import (  # noqa: E402
    intervention_sweep,
    load_trajectory_prompt,
    loop_autopsy,
    loop_direction,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--traj", type=Path, default=None)
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=_REPO / "out" / "lens" / "exp102")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--ngl", type=int, default=99)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"

    with running_jlens_server(args.model, ngl=args.ngl) as client:
        if args.traj is not None:
            tokens = load_trajectory_prompt(args.traj, client)
        elif args.prompt_file is not None:
            tokens = client.tokenize(args.prompt_file.read_text(), add_special=True, parse_special=True)
        else:
            raise SystemExit("pass --traj or --prompt-file")

    report = loop_autopsy(
        args.model, args.lens, tokens, runs_dir=runs_dir,
        reference_gguf=args.reference, n_predict=args.n_predict, ngl=args.ngl,
    )
    (out / "loop_report.json").write_text(json.dumps({
        "looped": report.looped, "notes": report.notes,
        "collapse_depth_quant": report.collapse_depth_quant,
        "collapse_depth_ref": report.collapse_depth_ref,
        "generated_text": report.generated_text[:3000],
    }, indent=1))
    print(f"[exp-102] {report.notes}")
    if not report.looped:
        print("[exp-102] no loop detected; nothing to autopsy")
        return 0
    if report.collapse_depth_ref:
        mq = sum(report.collapse_depth_quant) / max(len(report.collapse_depth_quant), 1)
        mr = sum(report.collapse_depth_ref) / max(len(report.collapse_depth_ref), 1)
        print(f"[exp-102] mean collapse depth: quant {mq:.3f} vs FP16 {mr:.3f} "
              f"({'quant commits earlier' if mq < mr else 'similar'})")

    if args.sweep and report.quant_run_id is not None:
        run = load_run(runs_dir / report.quant_run_id)
        layers = run.manifest.capture_layers
        late = layers[int(len(layers) * 2 / 3):]
        direction = loop_direction(run, report.loop, late[len(late) // 2])
        with running_jlens_server(args.model, ngl=args.ngl) as client:
            results = intervention_sweep(
                client, direction, layers=late, prompt_tokens=tokens,
                alphas=(0.5, 1.0, 1.5), n_predict=256,
            )
        winner = next((r for r in results if r.broke_loop), None)
        (out / "sweep.json").write_text(json.dumps([
            {"alpha": r.alpha, "broke_loop": r.broke_loop, "text": r.generated_text[:500]}
            for r in results
        ], indent=1))
        if winner is not None:
            direction.save(out / "direction.npz")
            print(f"[exp-102] loop broken by ablation alpha={winner.alpha} across {len(late)} layers")
            print(f"[exp-102] saved direction -> {out / 'direction.npz'} (feed to `quant-tuner lens bake`)")
        else:
            print("[exp-102] no ablation in the sweep broke the loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

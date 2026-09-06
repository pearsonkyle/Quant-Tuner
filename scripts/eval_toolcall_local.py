#!/usr/bin/env python3
"""Tool-call eval for local HF checkpoints, driven in-process.

`eval_toolcall.py` targets GGUF through llama-server. A stage-1 LoRA
checkpoint is neither, and converting one to GGUF just to measure it would
put quantization error inside a measurement about training. This runs the same
scorer against the same holdout with the model in-process instead.

Every arm shares the frozen stage-0 base and differs only by adapter, so the
turns are identical across arms and the comparison is paired.

    PYTHONPATH=src python scripts/eval_toolcall_local.py \\
        --holdout out/e4b-v65536/eval/toolcall_holdout_quick.jsonl \\
        --adapters .../checkpoint-3000 --max-turns-per-session 3 \\
        --out out/e4b-v65536/eval/toolcall_local.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval  # noqa: E402

STAGE0 = "/workspace/models/gemma4-e4b-stage0-32k-v65536/final"
PRUNED = "/workspace/models/gemma4-e4b-qat-v65536-text"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--base", default=STAGE0)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--include-pruned-base", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-turns-per-session", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-len", type=int, default=65536)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Halt a session at the model's first miss (the shared "
                        "harness's default). OFF here by default, because it "
                        "makes the arms unpaired: on the step-3000 run stage 0 "
                        "failed immediately and scored 36 turns while "
                        "checkpoint-3000 scored 89, so the two per-turn rates "
                        "had different denominators and were not comparable. "
                        "Scoring every turn costs more GPU time and buys a "
                        "number you can actually difference.")
    a = p.parse_args()

    from quant_tuner.eval.local_gemma4 import LocalGemma4Client

    arms: list[tuple[str, str, str | None]] = []
    if a.include_pruned_base:
        arms.append(("pruned base", PRUNED, None))
    arms.append(("stage 0 final", a.base, None))
    for ad in a.adapters:
        arms.append((Path(ad.rstrip("/")).name, a.base, ad))

    sampling = Sampling(temperature=a.temperature, max_tokens=a.max_tokens)
    out: dict = {"holdout": str(a.holdout),
                 "max_turns_per_session": a.max_turns_per_session,
                 "stop_on_fail": a.stop_on_fail, "runs": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    for label, base, adapter in arms:
        print(f"\n=== {label} ===", flush=True)
        client = LocalGemma4Client(base, adapter=adapter, device=a.device,
                                   max_len=a.max_len)
        t0 = time.time()
        summary = run_toolcall_eval(
            a.holdout, client=client, sampling=sampling, model_label=label,
            max_turns_per_session=a.max_turns_per_session,
            stop_on_fail=a.stop_on_fail,
            per_turn_log=a.out.with_suffix(f".{label.replace(' ', '_')}.turns.jsonl"),
            progress=a.progress,
        )
        d = asdict(summary) if is_dataclass(summary) else dict(summary)
        d["secs"] = time.time() - t0
        out["runs"].append(d)
        print(f"  {label}: {d.get('n_scored')} turns scored in "
              f"{d['secs']/60:.1f} min", flush=True)
        for k in ("tool_selection_acc", "param_acc_mean", "schema_valid_rate"):
            if k in d:
                print(f"    {k:22s} {d[k]:.4f}", flush=True)
        # Write after every arm: an eval that dies on arm 3 should not throw
        # away arms 1 and 2, which cost the same GPU minutes to produce.
        a.out.write_text(json.dumps(out, indent=2, default=str))
        del client
        gc.collect()
        if a.device != "cpu":
            import torch
            torch.cuda.empty_cache()

    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Quantize an HF checkpoint to a vLLM-servable compressed-tensors dir (W4A16),
calibrated on our corpora at long context.

Typical use (gemma-4-E4B for the HomeLab GPU-1 vLLM deployment):

    uv sync --extra vllm-ptq
    uv run python scripts/run_vllm_ptq.py \
        --model ~/Programs/llm/hf/gemma-4-E4B-it \
        --corpus out/<run>/corpus/corpus.cal.txt \
        --out out/e4b-w4a16-logs8k \
        --ctx 8192

Then serve:  vllm serve out/e4b-w4a16-logs8k --max-model-len 131072 ...

Corpora come from scripts/build_corpora.py (calibration = logtrain train slice
+ wiki, interleaved). Passing several --corpus files splits the token budget
across them proportionally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_tuner.vllm_export import (
    DEFAULT_IGNORE,
    PTQConfig,
    audit_ignore,
    dropped_tensors,
    run_ptq,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument(
        "--corpus",
        action="append",
        type=Path,
        help="calibration text file (repeatable; budget split proportionally)",
    )
    ap.add_argument("--out", type=Path, help="output checkpoint dir")
    ap.add_argument("--ctx", type=int, default=8192, help="calibration seq length")
    ap.add_argument("--budget-tokens", type=int, default=524_288)
    ap.add_argument(
        "--scheme",
        default="W4A16",
        choices=["W4A16", "W8A8", "W8A16", "FP8_DYNAMIC"],
    )
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "extra module pattern to leave unquantized, appended to DEFAULT_IGNORE "
            "(repeatable; llmcompressor syntax, e.g. 're:.*visual.*'). DEFAULT_IGNORE "
            "only covers gemma-style multimodal naming — a tower named something else "
            "(Qwen3.5's model.visual.*) or a top-level draft head (mtp.*) matches "
            "nothing and would be quantized silently. Audit with --dry-run-ignore."
        ),
    )
    ap.add_argument(
        "--dry-run-ignore",
        action="store_true",
        help=(
            "print how many tensors each ignore pattern matches against the "
            "checkpoint's weight map, then exit without quantizing"
        ),
    )
    ap.add_argument(
        "--model-class",
        default=None,
        help=(
            "transformers class to load with (default: AutoModelForCausalLM). "
            "On a multimodal checkpoint the Auto class resolves to the TEXT-ONLY "
            "class and drops every tower tensor from the export — pass e.g. "
            "Qwen3_5ForConditionalGeneration to keep it."
        ),
    )
    ap.add_argument("--device-map", default="auto")
    ap.add_argument(
        "--pipeline",
        default="sequential",
        choices=["sequential", "basic", "independent"],
        help="use 'basic' for models with cross-layer state (gemma-4 shared KV)",
    )
    args = ap.parse_args()

    ignore = DEFAULT_IGNORE + tuple(args.ignore)

    if args.dry_run_ignore:
        counts = audit_ignore(args.model, ignore, args.model_class)
        width = max(len(p) for p in counts)
        for pattern, n in counts.items():
            flag = "  <-- matches nothing" if n == 0 else ""
            print(f"{pattern:<{width}}  {n:5d}{flag}")
        print(f"\n{sum(counts.values())} modules left unquantized by this ignore list")
        dropped = dropped_tensors(args.model, args.model_class)
        if dropped:
            print(
                f"\n{len(dropped)} checkpoint tensors have NO module in "
                f"{args.model_class or 'AutoModelForCausalLM'} and will be ABSENT "
                f"from the export (no ignore entry can save them):"
            )
            prefixes: dict[str, int] = {}
            for name in dropped:
                prefixes[name.split(".")[0]] = prefixes.get(name.split(".")[0], 0) + 1
            for prefix, n in sorted(prefixes.items(), key=lambda kv: -kv[1]):
                print(f"  {prefix + '.*':<24} {n:5d}")
        return

    # Only optional so --dry-run-ignore can run without them.
    missing = [f"--{n}" for n in ("corpus", "out") if not getattr(args, n)]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    cfg = PTQConfig(
        model_id=args.model,
        out_dir=args.out,
        corpus_files=args.corpus,
        ctx=args.ctx,
        budget_tokens=args.budget_tokens,
        scheme=args.scheme,
        group_size=args.group_size,
        ignore=ignore,
        model_class=args.model_class,
        device_map=args.device_map,
        pipeline=args.pipeline,
    )
    out = run_ptq(cfg)
    print(f"quantized checkpoint written to {out}")


if __name__ == "__main__":
    main()

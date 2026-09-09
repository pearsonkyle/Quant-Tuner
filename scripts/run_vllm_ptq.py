#!/usr/bin/env python3
"""Quantize an HF checkpoint to a vLLM-servable compressed-tensors dir,
calibrated on our corpora at long context.

Two levers, both optional on top of the W4A16 preset:

  --group-size / --asymmetric / --observer / --actorder
      the weight grid. The preset is int4 group-128 symmetric minmax; the
      published INT4 cards use int4 **group-32 asymmetric** with the
      **imatrix-mse** observer and **static** act-ordering.

  --kv-cache-dtype fp8_e4m3
      calibrate static per-tensor fp8 KV scales in the *same* oneshot pass and
      bake them into the checkpoint. `vllm serve --kv-cache-dtype fp8_e4m3`
      picks them up automatically. This is the concurrency / long-context lever:
      weights shrink once, the KV cache shrinks per token per sequence.

Typical use (gemma-4-E4B for the HomeLab GPU-1 vLLM deployment):

    uv sync --extra vllm-ptq
    uv run python scripts/run_vllm_ptq.py \
        --model ~/Programs/llm/hf/gemma-4-E4B-it \
        --corpus out/<run>/corpus/corpus.cal.txt \
        --out out/e4b-w4a16-logs8k \
        --ctx 8192

The full published-card recipe (Qwen3.8-27B, int4 gs32 asym + fp8 KV):

    uv run python scripts/run_vllm_ptq.py \
        --model out/exp-060/model_extracted \
        --corpus out/exp-060-32k/corpora/corpus.cal.txt \
        --out out/exp-060-w4a16-fp8kv/checkpoint \
        --ctx 32768 --budget-tokens 4194304 \
        --group-size 32 --asymmetric --observer imatrix-mse --actorder static \
        --kv-cache-dtype fp8_e4m3 \
        --ignore 're:.*visual.*' --ignore 're:mtp.*'

Then serve:  vllm serve <out> --kv-cache-dtype fp8_e4m3 --max-model-len 262144

Corpora come from scripts/build_universal_corpus.py (or build_corpora.py for the
published two-source runs). Passing several --corpus files splits the token
budget across them proportionally.

**--ctx must match what the corpus was packed for** (top-level `ctx` in the
corpus's corpora_audit.json). It is a packing parameter, not just a flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_tuner.vllm_export import (
    DEFAULT_IGNORE,
    KNOWN_ACTORDER,
    KNOWN_OBSERVERS,
    KV_CACHE_SCHEMES,
    SUPPORTED_SCHEMES,
    VLLM_GROUP_SIZES,
    PTQConfig,
    audit_ignore,
    build_config_groups,
    build_kv_cache_scheme,
    dropped_tensors,
    run_ptq,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument(
        "--corpus",
        action="append",
        type=Path,
        help="calibration text file (repeatable; budget split proportionally)",
    )
    ap.add_argument("--out", type=Path, help="output checkpoint dir")
    ap.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help="calibration seq length; must match the corpus's packing ctx",
    )
    ap.add_argument(
        "--budget-tokens",
        type=int,
        default=524_288,
        help=(
            "total calibration tokens. Read it as sequences: at ctx 32768 the "
            "default is only 16, too few for a stable Hessian — use ~4194304"
        ),
    )
    ap.add_argument("--scheme", default="W4A16", choices=list(SUPPORTED_SCHEMES))

    grid = ap.add_argument_group("weight grid (deviating from the preset scheme)")
    grid.add_argument(
        "--group-size",
        type=int,
        default=128,
        choices=list(VLLM_GROUP_SIZES),
        help="quantization group along the input dim; -1 = per-output-channel",
    )
    grid.add_argument(
        "--asymmetric",
        action="store_true",
        help=(
            "store a zero-point per group (int4 asymmetric). More expressive and "
            "~3%% larger; what the published INT4 cards use at group 32"
        ),
    )
    grid.add_argument(
        "--observer",
        default=None,
        choices=list(KNOWN_OBSERVERS),
        help=(
            "how the per-group scale is picked. Default (unset) = llmcompressor's "
            "minmax. 'imatrix-mse' weights the MSE search by per-input-channel "
            "activation importance — the direct analogue of the GGUF imatrix"
        ),
    )
    grid.add_argument(
        "--actorder",
        default=None,
        choices=list(KNOWN_ACTORDER),
        help=(
            "GPTQ activation reordering. 'static' permutes columns once at "
            "quantization time, so serving pays no g_idx indirection"
        ),
    )
    grid.add_argument("--dampening-frac", type=float, default=0.01)
    grid.add_argument("--block-size", type=int, default=128)
    grid.add_argument(
        "--bypass-divisibility-checks",
        action="store_true",
        help="skip the group_size divisibility check (only after checking by hand)",
    )

    kv = ap.add_argument_group("KV cache")
    kv.add_argument(
        "--kv-cache-dtype",
        default=None,
        choices=list(KV_CACHE_SCHEMES),
        help=(
            "calibrate static per-tensor KV scales in the same pass and bake them "
            "into the checkpoint. Halves the cache at serve time (vllm serve "
            "--kv-cache-dtype fp8_e4m3), which is what buys context length and "
            "concurrency on a fixed card"
        ),
    )

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
            "print how many live modules each ignore pattern matches, the resolved "
            "recipe, and the tensors that would vanish from the export — then exit "
            "without quantizing"
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

    def make_cfg(corpus: list[Path], out: Path) -> PTQConfig:
        return PTQConfig(
            model_id=args.model,
            out_dir=out,
            corpus_files=corpus,
            ctx=args.ctx,
            budget_tokens=args.budget_tokens,
            scheme=args.scheme,
            group_size=args.group_size,
            symmetric=not args.asymmetric,
            observer=args.observer,
            actorder=args.actorder,
            dampening_frac=args.dampening_frac,
            block_size=args.block_size,
            bypass_divisibility_checks=args.bypass_divisibility_checks,
            kv_cache_dtype=args.kv_cache_dtype,
            ignore=ignore,
            model_class=args.model_class,
            device_map=args.device_map,
            pipeline=args.pipeline,
        )

    if args.dry_run_ignore:
        counts = audit_ignore(args.model, ignore, args.model_class)
        width = max(len(p) for p in counts)
        for pattern, n in counts.items():
            flag = "  <-- matches nothing" if n == 0 else ""
            print(f"{pattern:<{width}}  {n:5d}{flag}")
        print(f"\n{sum(counts.values())} modules left unquantized by this ignore list")

        # The resolved recipe, so a deviation from the preset is visible before
        # hours of calibration rather than only in the finished provenance.
        probe = make_cfg([Path(__file__)], Path("/dev/null"))
        groups = build_config_groups(probe)
        print("\nrecipe:")
        print(f"  scheme         {args.scheme}" + ("" if groups is None else "  (overridden)"))
        if groups is None:
            print("  weight grid    preset (int4 group-128 symmetric, minmax observer)")
        else:
            for k, v in groups["group_0"]["weights"].items():
                print(f"  {k:<14} {v}")
        kv_scheme = build_kv_cache_scheme(probe)
        print(f"  kv cache       {kv_scheme or 'not quantized'}")

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
            for prefix, n in sorted(prefixes.items(), key=lambda kv_: -kv_[1]):
                print(f"  {prefix + '.*':<24} {n:5d}")
        return

    # Only optional so --dry-run-ignore can run without them.
    missing = [f"--{n}" for n in ("corpus", "out") if not getattr(args, n)]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    out = run_ptq(make_cfg(args.corpus, args.out))
    print(f"quantized checkpoint written to {out}")
    if args.kv_cache_dtype:
        print(
            f"serve with:  vllm serve {out} --kv-cache-dtype {args.kv_cache_dtype} ..."
        )


if __name__ == "__main__":
    main()

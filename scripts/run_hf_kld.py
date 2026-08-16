#!/usr/bin/env python3
"""HF-side KLD: a quantized (compressed-tensors) checkpoint vs its bf16 reference.

The GGUF ladder's KLD comes from ``llama-perplexity --kl-divergence-base``,
which only speaks GGUF. This is the equivalent for the ``vllm_export`` W4A16
path. See :mod:`quant_tuner.bench.kld_hf` for the method and for why these
numbers share the GGUF table's *shape* but not its scale.

    PYTHONPATH=src .venv/bin/python scripts/run_hf_kld.py \
        --ref out/exp-060/model_extracted \
        --quant out/exp-060-w4a16-32k/checkpoint \
        --corpora-dir out/exp-060-32k/corpora \
        --out out/exp-060-w4a16-32k/kld_results.csv

Both models are held resident on the GPU (bf16 reference ~52 GB + W4A16
~13 GB). The quantized model is loaded FIRST so it gets contiguous memory
before the reference claims the rest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from quant_tuner.bench.kld_hf import (
    DEFAULT_EVAL_CORPORA,
    DEFAULT_EVAL_CTX,
    chunk_corpus,
    evaluate_corpus,
    evaluate_corpus_two_pass,
    iter_corpora,
    write_csv,
)
from quant_tuner.vllm_export import resolve_model_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, type=Path, help="bf16 reference checkpoint")
    ap.add_argument("--quant", required=True, type=Path, help="quantized checkpoint")
    ap.add_argument("--corpora-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="results CSV")
    ap.add_argument("--ctx", type=int, default=DEFAULT_EVAL_CTX)
    ap.add_argument(
        "--model-class",
        default="Qwen3_5ForConditionalGeneration",
        help="transformers class for BOTH models (must match the PTQ export)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--vocab-chunk",
        type=int,
        default=16384,
        help="vocab-dim block for the fp32 reductions; lower it if VRAM is tight",
    )
    ap.add_argument("--skip-first", type=int, default=1)
    ap.add_argument(
        "--corpora",
        nargs="*",
        default=None,
        help="subset of corpus labels (default: all six)",
    )
    ap.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="cap chunks per corpus — smoke-test knob, NOT for reported numbers",
    )
    ap.add_argument(
        "--two-pass",
        action="store_true",
        help=(
            "hold only one model on the GPU at a time, caching reference logits "
            "in CPU RAM (~4 GB per 8192-token chunk). Use when both models will "
            "not co-reside. Results are identical; only the schedule differs."
        ),
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    selected = DEFAULT_EVAL_CORPORA
    if args.corpora:
        wanted = set(args.corpora)
        selected = tuple(c for c in DEFAULT_EVAL_CORPORA if c[0] in wanted)
        unknown = wanted - {c[0] for c in DEFAULT_EVAL_CORPORA}
        if unknown:
            ap.error(f"unknown corpus label(s): {sorted(unknown)}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.ref))
    cls = resolve_model_class(args.model_class)

    # In two-pass mode the models are moved on and off the GPU by hand, so they
    # must load unsharded (device_map would pin them to a dispatch plan).
    load_device = None if args.two_pass else args.device
    print(f"loading quantized: {args.quant}", flush=True)
    quant_model = cls.from_pretrained(
        str(args.quant), torch_dtype="bfloat16", device_map=load_device
    ).eval()
    print(f"loading reference: {args.ref}", flush=True)
    ref_model = cls.from_pretrained(
        str(args.ref), torch_dtype="bfloat16", device_map=load_device
    ).eval()
    if args.device.startswith("cuda"):
        print(
            f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True
        )

    rows = []
    started = time.time()
    for label, path in iter_corpora(args.corpora_dir, selected):
        chunks = chunk_corpus(path, tokenizer, args.ctx)
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        print(f"{label}: {len(chunks)} chunks @ ctx {args.ctx}", flush=True)
        run = evaluate_corpus_two_pass if args.two_pass else evaluate_corpus
        row = run(
            ref_model,
            quant_model,
            chunks,
            corpus=label,
            device=args.device,
            vocab_chunk=args.vocab_chunk,
            skip_first=args.skip_first,
        )
        rows.append(row)
        print(
            f"  -> median KLD {row.median_kld:.5f} | top-1 {row.top1_agree:.2f}% "
            f"| ppl {row.ref_ppl:.3f} -> {row.quant_ppl:.3f}",
            flush=True,
        )
        # Rewrite after EVERY corpus, not once at the end. Each distribution costs
        # real GPU minutes, and a failure on corpus N previously discarded the
        # N-1 that had already succeeded.
        write_csv(rows, args.out)

    write_csv(rows, args.out)
    meta = {
        "ref": str(args.ref),
        "quant": str(args.quant),
        "model_class": args.model_class,
        "eval_ctx": args.ctx,
        "mode": "two-pass" if args.two_pass else "resident",
        "skip_first": args.skip_first,
        "limit_chunks": args.limit_chunks,
        "elapsed_s": round(time.time() - started, 1),
        "reference_is": "bf16 HF model (NOT the F16 GGUF the ladder used)",
        "special_tokens": "tokenized as single ids (llama-perplexity cannot)",
    }
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out}")
    print(f"{'corpus':10s} {'median KLD':>12s} {'top-1 %':>9s} {'top-5 %':>9s} {'ppl ratio':>10s}")
    for row in rows:
        print(
            f"{row.corpus:10s} {row.median_kld:12.5f} {row.top1_agree:9.2f} "
            f"{row.top5_agree:9.2f} {row.ppl_ratio:10.4f}"
        )


if __name__ == "__main__":
    main()

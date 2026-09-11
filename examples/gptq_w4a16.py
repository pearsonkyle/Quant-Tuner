"""Produce a W4A16 compressed-tensors checkpoint for vLLM serving.

The sibling of the GGUF pipeline: same calibration corpus, but the output is a
safetensors checkpoint that ``vllm serve`` reads directly (no llama.cpp
in the loop). The calibration context is 32K by default — the serving target
is long-context, so the Hessian statistics see long sequences.

    uv sync --extra vllm-ptq   # one-time; pulls llmcompressor
    uv run python examples/gptq_w4a16.py \
        --model Qwen/Qwen3.5-8B --corpus out/corpora/corpus.cal.txt \
        --out out/qwen3.5-w4a16

The checkpoint is then served with:

    vllm serve out/qwen3.5-w4a16 --max-model-len 32768
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local dir (bf16)")
    p.add_argument("--corpus", type=Path, required=True,
                   help="calibration text (corpus.cal.txt from build_universal_corpus.py)")
    p.add_argument("--out", type=Path, default=Path("./out/w4a16"))
    p.add_argument("--ctx", type=int, default=32768,
                   help="calibration sequence length (default 32K)")
    p.add_argument("--budget-tokens", type=int, default=524_288,
                   help="total calibration token budget (default 512K)")
    p.add_argument("--scheme", default="W4A16",
                   choices=["W4A16", "W8A8", "W8A16", "FP8_DYNAMIC"])
    p.add_argument("--model-class", default=None,
                   help="transformers class name; pass the conditional class for "
                        "multimodal checkpoints (e.g. Qwen3_5ForConditionalGeneration)")
    p.add_argument("--ignore", action="append", default=[],
                   help="extra module pattern to keep in bf16 (repeatable)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the PTQConfig and exit (no model load)")
    a = p.parse_args()

    from quant_tuner.vllm_export.w4a16 import DEFAULT_IGNORE, PTQConfig

    cfg = PTQConfig(
        model_id=a.model,
        out_dir=a.out,
        corpus_files=[a.corpus],
        ctx=a.ctx,
        budget_tokens=a.budget_tokens,
        scheme=a.scheme,
        model_class=a.model_class,
        ignore=tuple(DEFAULT_IGNORE + tuple(a.ignore)),
    )
    cfg.validate()

    if a.dry_run:
        import dataclasses
        print(dataclasses.asdict(cfg))
        return 0

    from quant_tuner.vllm_export.w4a16 import run_ptq

    out = run_ptq(cfg)
    print(f"\nWrote: {out}")
    print(f"Serve:  vllm serve {out} --max-model-len {a.ctx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

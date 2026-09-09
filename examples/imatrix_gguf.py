"""Calibrate a GGUF with the hybrid imatrix at 32K context.

The default path for "I have a HF model and a usage log, I want a GGUF that
runs on llama.cpp / Ollama / LM Studio". The recipe is
``iq2m_imatrix_32k`` (change ``--quant`` to IQ3_M, IQ4_XS, or Q5_K_M for the
other rungs; each has its own recipe).

    uv run python examples/imatrix_gguf.py \
        --model Qwen/Qwen3.5-8B --logs logs.jsonl --workspace out/run \
        --quant IQ2_M

For a 32K-context calibration corpus, point ``--corpus`` at a pre-built file
(see ``scripts/build_universal_corpus.py --ctx 32768``). Without it the
pipeline derives the corpus from ``--logs`` at the recipe's ``context_len``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.config import RunConfig  # noqa: E402
from quant_tuner.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local dir")
    p.add_argument("--logs", type=Path, default=None,
                   help="usage-log .jsonl(.gz). Required unless --corpus is given.")
    p.add_argument("--corpus", type=Path, default=None,
                   help="pre-built 32K-ctx calibration corpus (corpus.cal.txt). "
                        "Skips the logs-derived build.")
    p.add_argument("--workspace", type=Path, default=Path("./out/imatrix_32k"))
    p.add_argument("--quant", default="IQ2_M",
                   choices=["IQ2_M", "IQ3_M", "IQ4_XS", "Q5_K_M"],
                   help="the llama-quantize type (default IQ2_M)")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the recipe, print the merged config, exit")
    a = p.parse_args()

    # Recipe name is quant-lower + _imatrix_32k (IQ2_M -> iq2m_imatrix_32k).
    recipe_name = a.quant.lower().replace("_", "") + "_imatrix_32k"
    cfg = RunConfig.from_yaml(REPO / "src/quant_tuner/recipes" / f"{recipe_name}.yaml")
    cfg.model = a.model
    cfg.workspace = a.workspace
    if a.logs is not None:
        cfg.data.logs = a.logs
    if a.corpus is not None:
        cfg.data.corpus = a.corpus
    # Re-pin the quant type (the recipe was already written for it, but the
    # --quant flag is the source of truth for the recipe name).
    cfg.quantize.type = a.quant

    if a.dry_run:
        print(cfg.model_dump_json(indent=2))
        return 0

    out = run_pipeline(cfg)
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

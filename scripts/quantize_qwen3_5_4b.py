"""Driver: quantize Qwen/Qwen3.5-4B to Q4_K_M with the hybrid_custom imatrix.

Thin wrapper around the recipe-driven pipeline. Calibration is capped at
~250K train + ~50K eval tokens, with `calibration_supplement.txt` appended
to the train corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.config import RunConfig
from quant_tuner.pipeline import run_pipeline


RECIPE = REPO / "src" / "quant_tuner" / "recipes" / "q4_k_m_qwen3_5_4b.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True,
                        help="Usage-log JSONL for calibration corpus")
    parser.add_argument("--workspace", type=Path, default=None,
                        help="Override recipe workspace (default: ./out/q4_k_m_qwen3_5_4b)")
    parser.add_argument("--quant-type", type=str, default=None,
                        help="Override quantize.type (default: Q4_K_M from recipe)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override HF repo id (default: Qwen/Qwen3.5-4B from recipe)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit")
    args = parser.parse_args()

    cfg = RunConfig.from_yaml(RECIPE)
    cfg.data.logs = args.logs
    if args.workspace is not None:
        cfg.workspace = args.workspace
    if args.quant_type is not None:
        cfg.quantize.type = args.quant_type
    if args.model is not None:
        cfg.model = args.model

    if args.dry_run:
        print(cfg.model_dump_json(indent=2))
        return 0

    row = run_pipeline(cfg)
    print()
    print(f"DONE  bpw={row.bpw:.3f}  mean_kld={row.mean_kld}  same_top_p={row.same_top_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

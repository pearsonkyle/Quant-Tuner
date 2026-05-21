"""Driver: quantize Qwen/Qwen3.6-27B to Q4_K_M with MTP bundled.

Thin wrapper around the recipe-driven pipeline. The MTP head stays inside
the produced GGUF so the model can be served with speculative decoding:

    ./llama.cpp/llama-server -m <out>/gguf/Q4_K_M-imatrix.gguf \\
        --temp 0.7 --top-p 0.8 --top-k 20 \\
        --presence-penalty 1.5 --min-p 0.00 \\
        --spec-type draft-mtp --spec-draft-n-max 2 \\
        --chat-template-kwargs '{"enable_thinking":false}'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.config import RunConfig
from quant_tuner.pipeline import run_pipeline


RECIPE = REPO / "src" / "quant_tuner" / "recipes" / "q4_k_m_qwen3_6_mtp.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True,
                        help="Usage-log JSONL for calibration corpus")
    parser.add_argument("--workspace", type=Path, default=None,
                        help="Override recipe workspace (default: ./out/q4_k_m_qwen3_6_mtp)")
    parser.add_argument("--quant-type", type=str, default=None,
                        help="Override quantize.type (default: Q4_K_M from recipe)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override HF repo id (default: Qwen/Qwen3.6-27B from recipe)")
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
    quant_path = Path(cfg.workspace) / "gguf" / f"{cfg.quantize.type}-imatrix.gguf"
    print()
    print(f"DONE  bpw={row.bpw:.3f}  mean_kld={row.mean_kld}  same_top_p={row.same_top_p}")
    print()
    print("Serve with speculative MTP draft heads:")
    print(f"  ./llama.cpp/llama-server -m {quant_path} \\")
    print("      --temp 0.7 --top-p 0.8 --top-k 20 \\")
    print("      --presence-penalty 1.5 --min-p 0.00 \\")
    print("      --spec-type draft-mtp --spec-draft-n-max 2 \\")
    print("      --chat-template-kwargs '{\"enable_thinking\":false}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

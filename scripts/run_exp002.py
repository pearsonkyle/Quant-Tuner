"""Experiment 002: output_aware imatrix re-weighting on Jackrong/Qwopus3.5-9B-Coder.

Reuses exp-001 artifacts (F16 GGUF, vanilla custom imatrix, eval corpus,
F16 KLD baseline). Adds one new row to compare against the existing
exp-001 vanilla-custom and none rows for the same model.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import imatrix
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import gguf

MODEL = "Jackrong/Qwopus3.5-9B-Coder"
SLUG = MODEL.replace("/", "__")

EXP1 = REPO / "out" / "exp-001" / SLUG
F16 = EXP1 / "model-f16.gguf"
BASE_IMATRIX = EXP1 / "imatrix-custom.gguf"
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"

EXP2_ROOT = REPO / "out" / "exp-002"
WORK = EXP2_ROOT / SLUG
LOGS = WORK / "logs"
OUTPUT_AWARE_IMATRIX = WORK / "imatrix-output_aware.gguf"
QUANT = WORK / "Q4_K_M-output_aware.gguf"
PER_MODEL_CSV = WORK / "results.csv"
AGGREGATE_CSV = EXP2_ROOT / "results.csv"


def main() -> int:
    for required in (F16, BASE_IMATRIX, EVAL_DS, BASE_KLD):
        if not required.exists():
            print(f"missing prerequisite from exp-001: {required}", file=sys.stderr)
            return 2

    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase(f"model {MODEL}"):
        # "output_aware" is what we call it in this study; the underlying source
        # variant in calibrate/imatrix.py is `hybrid_custom` (kept as-is so other
        # recipes/scripts in the repo continue to resolve it).
        step("build output_aware imatrix from vanilla base", OUTPUT_AWARE_IMATRIX,
             lambda: imatrix.calibrate(
                 variant="hybrid_custom", f16_gguf=F16,
                 base_imatrix=BASE_IMATRIX, out_path=OUTPUT_AWARE_IMATRIX))

        step("quantize Q4_K_M (output_aware)", QUANT,
             lambda: gguf.quantize(
                 F16, QUANT, "Q4_K_M", imatrix=OUTPUT_AWARE_IMATRIX,
                 log=LOGS / "quantize-output_aware.log"))

        n_params = bpw_mod.n_params(F16)
        label = f"{MODEL}|imatrix|custom+output_aware"
        with phase(f"bench {label}"):
            row = runner.bench_one(
                QUANT, label,
                reference_n_params=n_params,
                eval_dataset=EVAL_DS,
                eval_baseline=BASE_KLD,
                eval_ctx=8192,
                log_dir=LOGS,
                suite="kld",
            )
            runner.append_row(PER_MODEL_CSV, row)
            runner.append_row(AGGREGATE_CSV, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p}")

    log("ALL DONE")
    log(f"aggregate CSV: {AGGREGATE_CSV.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Combine outlier_l4 (exp-002 winning variant) with the mixed8k base imatrix
from exp-001 (500k custom + full wiki, llama-imatrix at ctx=8192).

Re-weighting variant is `outlier_l4` (√E[a⁴]) — same as the exp-002 winner.
The novelty here is the base imatrix: mixed8k instead of custom-only. Forward
stats for E[a⁴] are collected on corpus.mixed8k.txt so both signals derive
from the same calibration data.

Writes everything under out/exp-002/.../mixed8k_outlier_l4/ to keep it
disjoint from the original exp-002 outlier run.
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
MODEL_DIR = EXP1 / "model_extracted"
BASE_IMATRIX = EXP1 / "imatrix-mixed8k.gguf"
CALIBRATION_TEXT = EXP1 / "corpus.mixed8k.txt"
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"

WORK = REPO / "out" / "exp-002" / SLUG / "mixed8k_outlier_l4"
LOGS = WORK / "logs"
FORWARD_STATS = WORK / "forward_stats.npz"
IMATRIX_OUT = WORK / "imatrix-mixed8k_outlier_l4.gguf"
QUANT_OUT = WORK / "Q4_K_M-mixed8k_outlier_l4.gguf"
PER_MODEL_CSV = WORK / "results.csv"
AGGREGATE_CSV = REPO / "out" / "exp-002" / "results.csv"

FORWARD_TOKENS = 50_000
FORWARD_CTX = 1024


def main() -> int:
    for required in (F16, MODEL_DIR, BASE_IMATRIX, CALIBRATION_TEXT, EVAL_DS, BASE_KLD):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 2

    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase(f"model {MODEL} — outlier_l4 + mixed8k base"):
        def _collect():
            stats = imatrix.collect_forward_stats(
                MODEL_DIR, CALIBRATION_TEXT, BASE_IMATRIX,
                tokens=FORWARD_TOKENS, ctx=FORWARD_CTX,
                device="mps", dtype="bfloat16",
            )
            stats.save(FORWARD_STATS)
            log(f"  saved forward stats: {FORWARD_STATS.name}")

        step(f"collect forward stats on mixed8k ({FORWARD_TOKENS} tok, ctx={FORWARD_CTX})",
             FORWARD_STATS, _collect)

        step("build outlier_l4 imatrix (mixed8k base)", IMATRIX_OUT,
             lambda: imatrix.calibrate(
                 variant="outlier_l4", f16_gguf=F16, base_imatrix=BASE_IMATRIX,
                 out_path=IMATRIX_OUT, forward_stats_path=FORWARD_STATS))

        step("quantize Q4_K_M (mixed8k_outlier_l4)", QUANT_OUT,
             lambda: gguf.quantize(
                 F16, QUANT_OUT, "Q4_K_M", imatrix=IMATRIX_OUT,
                 log=LOGS / "quantize.log"))

        label = f"{MODEL}|imatrix|mixed8k+outlier_l4"
        n_params = bpw_mod.n_params(F16)
        with phase(f"bench {label}"):
            row = runner.bench_one(
                QUANT_OUT, label,
                reference_n_params=n_params,
                eval_dataset=EVAL_DS, eval_baseline=BASE_KLD,
                eval_ctx=8192, log_dir=LOGS, suite="kld",
            )
            runner.append_row(PER_MODEL_CSV, row)
            runner.append_row(AGGREGATE_CSV, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p}")

    log("ALL DONE")
    log(f"next: scripts/run_toolcall_reps.py --models {QUANT_OUT.relative_to(REPO)} "
        "--holdout out/exp-002/toolcall_holdout.jsonl --reps 5 "
        "--results out/exp-002/toolcall_reps_mixed8k_outlier_l4_results.csv "
        "--aggregated out/exp-002/toolcall_reps_mixed8k_outlier_l4_aggregated.csv "
        "--log-dir out/exp-002/logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

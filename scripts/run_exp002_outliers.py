"""Exp-002 extension: outlier_l4 and outlier_max variants on Jackrong.

Motivation: exp-002 found that output_aware (`hybrid_custom`) regressed
schema_valid_rate on the tool-call eval by 5.6 pts vs no calibration,
despite improving KLD. Hypothesis: imatrix calibration concentrates bits
on high-`E[a²]` natural-language channels at the expense of rare/heavy-tail
channels that carry structural tokens (JSON brackets, quotes, tool-name
fragments). Heavy-tail-aware variants (`outlier_l4` uses `√E[a⁴]`,
`outlier_max` uses `max|a_c|`) directly target this failure mode.

Both variants require an HF forward pass over the calibration corpus to
record E[a⁴] and max|a|. Forward stats are computed once and shared.

Reuses exp-001 artifacts (F16, model_extracted, custom imatrix base,
corpus.train.txt, corpus.eval.txt, baseline.kld). Writes to out/exp-002/.
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
BASE_IMATRIX = EXP1 / "imatrix-custom.gguf"
CALIBRATION_TEXT = EXP1 / "corpus.train.txt"
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"

WORK = REPO / "out" / "exp-002" / SLUG
LOGS = WORK / "logs"
FORWARD_STATS = WORK / "forward_stats.npz"
AGGREGATE_CSV = REPO / "out" / "exp-002" / "results.csv"
PER_MODEL_CSV = WORK / "results.csv"

# 50K tokens at ctx=1024 → ~50 forward passes. Defaults are 4K/1024 which
# is too small for a stable E[a⁴] estimate.
FORWARD_TOKENS = 50_000
FORWARD_CTX = 1024

VARIANTS = [
    ("outlier_l4",  WORK / "imatrix-outlier_l4.gguf",  WORK / "Q4_K_M-outlier_l4.gguf"),
    ("outlier_max", WORK / "imatrix-outlier_max.gguf", WORK / "Q4_K_M-outlier_max.gguf"),
]


def main() -> int:
    for required in (F16, MODEL_DIR, BASE_IMATRIX, CALIBRATION_TEXT, EVAL_DS, BASE_KLD):
        if not required.exists():
            print(f"missing prerequisite from exp-001: {required}", file=sys.stderr)
            return 2

    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase(f"model {MODEL} — outlier variants"):
        # 1) Forward pass to capture E[a^4] and max|a|. Saved + reused across variants.
        def _collect():
            stats = imatrix.collect_forward_stats(
                MODEL_DIR, CALIBRATION_TEXT, BASE_IMATRIX,
                tokens=FORWARD_TOKENS, ctx=FORWARD_CTX,
                device="mps", dtype="bfloat16",
            )
            stats.save(FORWARD_STATS)
            log(f"  saved forward stats: {FORWARD_STATS.name}")

        step(f"collect forward stats ({FORWARD_TOKENS} tok, ctx={FORWARD_CTX})",
             FORWARD_STATS, _collect)

        n_params = bpw_mod.n_params(F16)

        for variant, imatrix_path, quant_path in VARIANTS:
            step(f"build {variant} imatrix", imatrix_path,
                 lambda v=variant, p=imatrix_path: imatrix.calibrate(
                     variant=v, f16_gguf=F16, base_imatrix=BASE_IMATRIX,
                     out_path=p, forward_stats_path=FORWARD_STATS))

            step(f"quantize Q4_K_M ({variant})", quant_path,
                 lambda v=variant, q=quant_path, i=imatrix_path: gguf.quantize(
                     F16, q, "Q4_K_M", imatrix=i,
                     log=LOGS / f"quantize-{v}.log"))

            label = f"{MODEL}|imatrix|custom+{variant}"
            with phase(f"bench {label}"):
                row = runner.bench_one(
                    quant_path, label,
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
    log("next: scripts/run_toolcall_reps.py --models "
        f"{VARIANTS[0][2].relative_to(REPO)} {VARIANTS[1][2].relative_to(REPO)} "
        "--holdout out/exp-002/toolcall_holdout.jsonl --reps 5 "
        "--results out/exp-002/toolcall_reps_results.csv "
        "--aggregated out/exp-002/toolcall_reps_aggregated.csv "
        "--log-dir out/exp-002/logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

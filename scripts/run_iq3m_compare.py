"""IQ3_M on Gemma-4-31B-it: imatrix-only vs AWQ+imatrix.

Reuses all upstream artifacts:
  - exp-009 model-f16.gguf + imatrix-mixed8k.gguf + baseline.kld + corpus.eval.txt
  - exp-010 model-f16-awq.gguf (already-folded AWQ F16)

Two quantize + two KLD bench passes. Writes:
  - imatrix-only row into out/exp-009/.../results.csv
  - awq row into out/exp-010/.../results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import gguf

REPO_ID = "google/gemma-4-31B-it"
SLUG = REPO_ID.replace("/", "__")

EXP09 = REPO / "out" / "exp-009" / SLUG
EXP10 = REPO / "out" / "exp-010" / SLUG
EXP09_LOGS = EXP09 / "logs"
EXP10_LOGS = EXP10 / "logs"

EVAL_CTX = 4096
QUANT = "IQ3_M"


def main() -> int:
    f16 = EXP09 / "model-f16.gguf"
    f16_awq = EXP10 / "model-f16-awq.gguf"
    imat = EXP09 / "imatrix-mixed8k.gguf"
    eval_ds = EXP09 / "corpus.eval.txt"
    base_kld = EXP09 / "baseline.kld"

    missing = [p for p in (f16, f16_awq, imat, eval_ds, base_kld) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing inputs:\n  " + "\n  ".join(str(p) for p in missing)
        )

    n_params = bpw_mod.n_params(f16)
    log(f"reference n_params = {n_params:,.0f}")

    # ---- imatrix-only ------------------------------------------------- #
    qpath_imat = EXP09 / f"{QUANT}-mixed8k.gguf"
    csv_imat = EXP09 / "results.csv"
    label_imat = f"{REPO_ID}|{QUANT}|imatrix|500k-custom+wiki (ctx=8192)"
    with phase(f"{QUANT} imatrix-only"):
        step(f"quantize {QUANT} (imatrix)", qpath_imat,
             lambda: gguf.quantize(
                 f16, qpath_imat, QUANT, imatrix=imat,
                 log=EXP09_LOGS / f"quantize-{QUANT}.log"))
        with phase(f"bench {label_imat}"):
            row = runner.bench_one(
                qpath_imat, label_imat,
                reference_n_params=n_params,
                eval_dataset=eval_ds,
                eval_baseline=base_kld,
                eval_ctx=EVAL_CTX,
                log_dir=EXP09_LOGS,
                suite="kld",
            )
            runner.append_row(csv_imat, row)
            log(f"  IMATRIX size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} same_top_p={row.same_top_p}")

    # ---- AWQ + imatrix ------------------------------------------------- #
    qpath_awq = EXP10 / f"{QUANT}-awq.gguf"
    csv_awq = EXP10 / "results.csv"
    label_awq = f"{REPO_ID}|{QUANT}|awq+imatrix|500k-custom+wiki (ctx=8192) + AWQ"
    with phase(f"{QUANT} AWQ"):
        step(f"quantize {QUANT} (awq)", qpath_awq,
             lambda: gguf.quantize(
                 f16_awq, qpath_awq, QUANT, imatrix=imat,
                 log=EXP10_LOGS / f"quantize-{QUANT}.log"))
        with phase(f"bench {label_awq}"):
            row = runner.bench_one(
                qpath_awq, label_awq,
                reference_n_params=n_params,
                eval_dataset=eval_ds,
                eval_baseline=base_kld,
                eval_ctx=EVAL_CTX,
                log_dir=EXP10_LOGS,
                suite="kld",
            )
            runner.append_row(csv_awq, row)
            log(f"  AWQ     size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} same_top_p={row.same_top_p}")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

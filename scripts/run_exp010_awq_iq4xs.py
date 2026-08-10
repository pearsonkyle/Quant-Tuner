"""Add IQ4_XS to the exp-010 AWQ sweep on Gemma-4-31B-it.

Reuses exp-010's already-folded AWQ F16 GGUF (`model-f16-awq.gguf`),
exp-009's mixed8k imatrix, and exp-009's FP16 KLD baseline. One
quantize + one KLD bench.

Comparison point: exp-009 IQ4_XS imatrix-only = PPL 212.13.
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
LOGS = EXP10 / "logs"

EVAL_CTX = 4096
DATASET_LABEL = "500k-custom+wiki (ctx=8192) + AWQ"
QUANT = "IQ4_XS"


def main() -> int:
    f16_awq = EXP10 / "model-f16-awq.gguf"
    imat = EXP09 / "imatrix-mixed8k.gguf"
    eval_ds = EXP09 / "corpus.eval.txt"
    base_kld = EXP09 / "baseline.kld"
    f16_ref = EXP09 / "model-f16.gguf"
    per_model_csv = EXP10 / "results.csv"

    missing = [p for p in (f16_awq, imat, eval_ds, base_kld, f16_ref) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing inputs:\n  " + "\n  ".join(str(p) for p in missing)
        )

    n_params = bpw_mod.n_params(f16_ref)
    log(f"reference n_params = {n_params:,.0f}")

    qpath = EXP10 / f"{QUANT}-awq.gguf"
    with phase(f"{QUANT} AWQ"):
        step(f"quantize {QUANT}", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=imat,
                 log=LOGS / f"quantize-{QUANT}.log"))
        label = f"{REPO_ID}|{QUANT}|awq+imatrix|{DATASET_LABEL}"
        with phase(f"bench {label}"):
            row = runner.bench_one(
                qpath, label,
                reference_n_params=n_params,
                eval_dataset=eval_ds,
                eval_baseline=base_kld,
                eval_ctx=EVAL_CTX,
                log_dir=LOGS,
                suite="kld",
            )
            runner.append_row(per_model_csv, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p}")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

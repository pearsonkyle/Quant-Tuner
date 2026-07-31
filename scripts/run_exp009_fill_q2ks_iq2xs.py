"""Fill in Q2_K_S + IQ2_XS imatrix-only rows for Gemma-4-31B-it (exp-009).

exp-009 originally swept Q6_K..IQ2_XXS but skipped Q2_K_S and IQ2_XS.
exp-010 produced AWQ versions of both — this script fills in the
imatrix-only baselines so the comparison is symmetric.

Reuses exp-009's F16, mixed8k imatrix, eval corpus, and FP16 KLD
baseline. Appends rows to out/exp-009/.../results.csv.
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
WORK = REPO / "out" / "exp-009" / SLUG
LOGS = WORK / "logs"

EVAL_CTX = 4096
DATASET_LABEL = "500k-custom+wiki (ctx=8192)"
QUANTS = ["Q2_K_S", "IQ2_XS"]


def main() -> int:
    f16 = WORK / "model-f16.gguf"
    imat = WORK / "imatrix-mixed8k.gguf"
    eval_ds = WORK / "corpus.eval.txt"
    base_kld = WORK / "baseline.kld"
    per_model_csv = WORK / "results.csv"

    missing = [p for p in (f16, imat, eval_ds, base_kld) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing exp-009 inputs:\n  " + "\n  ".join(str(p) for p in missing)
        )

    n_params = bpw_mod.n_params(f16)
    log(f"f16 n_params = {n_params:,.0f}")

    for q in QUANTS:
        qpath = WORK / f"{q}-mixed8k.gguf"
        with phase(f"{q} mixed8k"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16, p, qt, imatrix=imat,
                     log=LOGS / f"quantize-{qt}.log"))
            label = f"{REPO_ID}|{q}|imatrix|{DATASET_LABEL}"
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

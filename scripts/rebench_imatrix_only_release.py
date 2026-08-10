"""Regenerate IQ2_XS / IQ2_M / Q2_K_S imatrix-only quants and bench them on
the exp-019 eval corpus so they are directly comparable to the AWQ cv-gate
release numbers. Idempotent via experiments.step().
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

SLUG = "google__gemma-4-31B-it"
EXP09 = REPO / "out" / "exp-009" / SLUG
EXP19 = REPO / "out" / "exp-019" / SLUG
OUT_DIR = EXP19 / "imatrix-only-rebench"
LOGS = OUT_DIR / "logs"

SRC_F16 = EXP09 / "model-f16.gguf"
IMATRIX = EXP19 / "imatrix-cal.gguf"
EVAL = EXP19 / "corpora" / "corpus.eval.txt"
BASELINE = EXP19 / "baseline.kld"

QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]
EVAL_CTX = 4096
DATASET_LABEL = (
    "wiki+500k-logtrain (cal) / 10k-logtrain+supplement (val) / code+math+tools (eval)"
)


def main() -> int:
    for p in (SRC_F16, IMATRIX, EVAL, BASELINE):
        assert p.exists(), f"missing: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params = {n_params:,.0f}")
    per_model_csv = OUT_DIR / "results.csv"

    for q in QUANTS:
        qpath = OUT_DIR / f"{q}-imatrix.gguf"
        with phase(f"{q} imatrix-only"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     SRC_F16, p, qt, imatrix=IMATRIX,
                     log=LOGS / f"quantize-{qt}.log"))
            label = f"google/gemma-4-31B-it|{q}|imatrix|{DATASET_LABEL}"
            with phase(f"bench {label}"):
                row = runner.bench_one(
                    qpath, label,
                    reference_n_params=n_params,
                    eval_dataset=EVAL,
                    eval_baseline=BASELINE,
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

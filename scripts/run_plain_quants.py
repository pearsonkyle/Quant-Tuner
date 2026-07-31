"""Plain (no-imatrix) quants on Gemma-4-31B-it: Q2_K, Q3_K_S, Q3_K_M, Q5_K_S, Q5_K_M.

Writes a separate CSV at out/exp-011/google__gemma-4-31B-it/results.csv
so plain rows don't intermix with imatrix or AWQ rows.

Reuses exp-009 F16/eval_corpus/baseline.kld (same FP16 reference, so
KLD numbers compare 1:1 across all three experiments).
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
EXP11 = REPO / "out" / "exp-011" / SLUG
LOGS = EXP11 / "logs"

EVAL_CTX = 4096
QUANTS = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q5_K_S", "Q5_K_M"]
DATASET_LABEL = "no-calibration"


def main() -> int:
    EXP11.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    f16 = EXP09 / "model-f16.gguf"
    eval_ds = EXP09 / "corpus.eval.txt"
    base_kld = EXP09 / "baseline.kld"
    csv_path = EXP11 / "results.csv"

    missing = [p for p in (f16, eval_ds, base_kld) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing:\n  " + "\n  ".join(str(p) for p in missing))

    n_params = bpw_mod.n_params(f16)
    log(f"reference n_params = {n_params:,.0f}")

    for q in QUANTS:
        qpath = EXP11 / f"{q}-plain.gguf"
        label = f"{REPO_ID}|{q}|plain|{DATASET_LABEL}"
        with phase(f"{q} plain"):
            step(f"quantize {q} (plain)", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16, p, qt, imatrix=None,
                     log=LOGS / f"quantize-{qt}.log"))
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
                runner.append_row(csv_path, row)
                log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                    f"ppl={row.ppl} mean_kld={row.mean_kld} same_top_p={row.same_top_p}")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

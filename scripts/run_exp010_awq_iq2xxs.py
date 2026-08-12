"""Add IQ2_XXS AWQ to exp-010. Mirrors run_exp010_awq_iq4xs.py."""

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
QUANT = "IQ2_XXS"


def main() -> int:
    f16_awq = EXP10 / "model-f16-awq.gguf"
    imat = EXP09 / "imatrix-mixed8k.gguf"
    eval_ds = EXP09 / "corpus.eval.txt"
    base_kld = EXP09 / "baseline.kld"
    f16_ref = EXP09 / "model-f16.gguf"
    n_params = bpw_mod.n_params(f16_ref)

    qpath = EXP10 / f"{QUANT}-awq.gguf"
    label = f"{REPO_ID}|{QUANT}|awq+imatrix|500k-custom+wiki (ctx=8192) + AWQ"
    with phase(f"{QUANT} AWQ"):
        step(f"quantize {QUANT}", qpath,
             lambda: gguf.quantize(f16_awq, qpath, QUANT, imatrix=imat,
                                   log=LOGS / f"quantize-{QUANT}.log"))
        with phase(f"bench {label}"):
            row = runner.bench_one(
                qpath, label,
                reference_n_params=n_params,
                eval_dataset=eval_ds,
                eval_baseline=base_kld,
                eval_ctx=4096,
                log_dir=LOGS,
                suite="kld",
            )
            runner.append_row(EXP10 / "results.csv", row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} same_top_p={row.same_top_p}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

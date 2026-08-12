"""Add three Gemma-4-31B-it quants:
  1. Q4_K_M  AWQ+imatrix   (imatrix-only already in exp-009)
  2. IQ4_NL  imatrix-only
  3. IQ4_NL  AWQ+imatrix

Reuses exp-009 F16/imatrix/baseline + exp-010 model-f16-awq.gguf.
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


def _bench(qpath, label, n_params, eval_ds, base_kld, log_dir, csv_path):
    with phase(f"bench {label}"):
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=eval_ds,
            eval_baseline=base_kld,
            eval_ctx=EVAL_CTX,
            log_dir=log_dir,
            suite="kld",
        )
        runner.append_row(csv_path, row)
        log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
            f"ppl={row.ppl} mean_kld={row.mean_kld} same_top_p={row.same_top_p}")


def main() -> int:
    f16 = EXP09 / "model-f16.gguf"
    f16_awq = EXP10 / "model-f16-awq.gguf"
    imat = EXP09 / "imatrix-mixed8k.gguf"
    eval_ds = EXP09 / "corpus.eval.txt"
    base_kld = EXP09 / "baseline.kld"
    csv_imat = EXP09 / "results.csv"
    csv_awq = EXP10 / "results.csv"

    missing = [p for p in (f16, f16_awq, imat, eval_ds, base_kld) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing:\n  " + "\n  ".join(str(p) for p in missing))

    n_params = bpw_mod.n_params(f16)
    log(f"reference n_params = {n_params:,.0f}")

    # 1) Q4_K_M AWQ
    q = "Q4_K_M"
    qpath = EXP10 / f"{q}-awq.gguf"
    label = f"{REPO_ID}|{q}|awq+imatrix|500k-custom+wiki (ctx=8192) + AWQ"
    with phase(f"{q} AWQ"):
        step(f"quantize {q} (awq)", qpath,
             lambda: gguf.quantize(f16_awq, qpath, q, imatrix=imat,
                                   log=EXP10_LOGS / f"quantize-{q}.log"))
        _bench(qpath, label, n_params, eval_ds, base_kld, EXP10_LOGS, csv_awq)

    # 2) IQ4_NL imatrix-only
    q = "IQ4_NL"
    qpath = EXP09 / f"{q}-mixed8k.gguf"
    label = f"{REPO_ID}|{q}|imatrix|500k-custom+wiki (ctx=8192)"
    with phase(f"{q} imatrix-only"):
        step(f"quantize {q} (imatrix)", qpath,
             lambda: gguf.quantize(f16, qpath, q, imatrix=imat,
                                   log=EXP09_LOGS / f"quantize-{q}.log"))
        _bench(qpath, label, n_params, eval_ds, base_kld, EXP09_LOGS, csv_imat)

    # 3) IQ4_NL AWQ
    q = "IQ4_NL"
    qpath = EXP10 / f"{q}-awq.gguf"
    label = f"{REPO_ID}|{q}|awq+imatrix|500k-custom+wiki (ctx=8192) + AWQ"
    with phase(f"{q} AWQ"):
        step(f"quantize {q} (awq)", qpath,
             lambda: gguf.quantize(f16_awq, qpath, q, imatrix=imat,
                                   log=EXP10_LOGS / f"quantize-{q}.log"))
        _bench(qpath, label, n_params, eval_ds, base_kld, EXP10_LOGS, csv_awq)

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

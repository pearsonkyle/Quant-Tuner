"""Build mixed-corpus Q4_K_M variants of gemma-4-E4B-it at ctx=2048 and ctx=512.

One-off follow-up to exp-001: reuses the existing mixed corpus
(`corpus.mixed8k.txt`, 500k custom tokens + full wiki.test.raw) and the
F16 GGUF, varying only the imatrix context. Benches against the existing
KLD baseline (ctx=4096) and appends rows to the per-model + aggregate CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import gguf

WORK = REPO / "out" / "exp-001" / "google__gemma-4-E4B-it"
LOGS = WORK / "logs"
F16 = WORK / "model-f16.gguf"
MIXED = WORK / "corpus.mixed8k.txt"
BASE_KLD = WORK / "baseline.kld"
PER_MODEL_CSV = WORK / "results.csv"
AGGREGATE_CSV = REPO / "out" / "exp-001" / "results.csv"
EVAL_DS = WORK / "corpus.eval.txt"
EVAL_CTX = 4096  # gemma vocab busts ctx=8192 in llama-perplexity

CELLS = [
    ("mixed2k", 2048, "500k-custom+wiki (ctx=2048)"),
    ("mixed512", 512, "500k-custom+wiki (ctx=512)"),
]


def main() -> int:
    for required in (F16, MIXED, BASE_KLD, EVAL_DS):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 1
    LOGS.mkdir(parents=True, exist_ok=True)

    n_params = bpw_mod.n_params(F16)

    for cell, ctx, dataset_label in CELLS:
        imat = WORK / f"imatrix-{cell}.gguf"
        qpath = WORK / f"Q4_K_M-{cell}.gguf"
        with phase(f"cell {cell} (ctx={ctx})"):
            step(f"llama-imatrix {cell} (ctx={ctx})", imat,
                 lambda i=imat, c=ctx, n=cell: llama_cpp.imatrix(
                     F16, MIXED, i, ctx=c, log=LOGS / f"imatrix-{n}.log"))
            step(f"quantize Q4_K_M ({cell})", qpath,
                 lambda q=qpath, s=imat, n=cell: gguf.quantize(
                     F16, q, "Q4_K_M", imatrix=s,
                     log=LOGS / f"quantize-{n}.log"))

            label = f"google/gemma-4-E4B-it|imatrix|{dataset_label}"
            with phase(f"bench {label}"):
                row = runner.bench_one(
                    qpath, label,
                    reference_n_params=n_params,
                    eval_dataset=EVAL_DS,
                    eval_baseline=BASE_KLD,
                    eval_ctx=EVAL_CTX,
                    log_dir=LOGS,
                    suite="kld",
                )
                runner.append_row(PER_MODEL_CSV, row)
                runner.append_row(AGGREGATE_CSV, row)
                log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                    f"ppl={row.ppl} mean_kld={row.mean_kld} "
                    f"same_top_p={row.same_top_p}")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

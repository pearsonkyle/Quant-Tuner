"""exp-044 addendum: IQ4_XS + imatrix for vanilla and QAT sources.

Reuses all exp-044 inputs (F16s, imatrices, corpora, baseline.kld).
Appends two rows to out/exp-044/results.csv.

    PYTHONPATH=src .venv/bin/python scripts/exp044_iq4xs.py
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

EXP44 = REPO / "out" / "exp-044"
CSV = EXP44 / "results.csv"

VANILLA_F16     = EXP44 / "vanilla" / "model-f16.gguf"
VANILLA_IMATRIX = EXP44 / "vanilla" / "imatrix-cal.gguf"
EVAL_CORPUS     = EXP44 / "corpora" / "corpus.eval.txt"
BASELINE_KLD    = EXP44 / "baseline.kld"

_EXP22 = REPO / "out" / "exp-022" / "google__gemma-4-31B-it-qat-q4_0-unquantized"
QAT_F16     = _EXP22 / "model-f16.gguf"
QAT_IMATRIX = EXP44 / "qat" / "imatrix-cal.gguf"

VANILLA_ID = "google/gemma-4-31B-it"
QAT_ID     = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QUANT      = "IQ4_XS"
EVAL_CTX   = 4096

ROWS = [
    ("vanilla", VANILLA_ID, VANILLA_F16, VANILLA_IMATRIX,
     EXP44 / "v_iq4xs_im" / f"gemma-4-31B-it-{QUANT}-imatrix.gguf"),
    ("qat",    QAT_ID,     QAT_F16,     QAT_IMATRIX,
     EXP44 / "q_iq4xs_im" / f"gemma-4-31B-it-qat-{QUANT}-imatrix.gguf"),
]


def _in_csv(qpath: Path) -> bool:
    return CSV.exists() and any(str(qpath) in ln for ln in CSV.open())


def main() -> int:
    for p in (VANILLA_F16, VANILLA_IMATRIX, QAT_F16, QAT_IMATRIX,
              EVAL_CORPUS, BASELINE_KLD):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    n_params_vanilla = bpw_mod.n_params(VANILLA_F16)
    n_params_qat     = bpw_mod.n_params(QAT_F16)

    for src, model_id, f16, imatrix, qpath in ROWS:
        tag = f"[exp-044][{src}_iq4xs_im]"
        log_dir = qpath.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        n_params = n_params_vanilla if src == "vanilla" else n_params_qat

        with phase(f"{tag} quantize {QUANT}"):
            step("quantize", qpath,
                 lambda f16=f16, qp=qpath, im=imatrix: gguf.quantize(
                     f16, qp, QUANT, imatrix=im,
                     log=log_dir / "quantize.log"))

        with phase(f"{tag} bench"):
            if _in_csv(qpath):
                log(f"  {qpath.name}: already in CSV — skipping")
            else:
                label = f"{model_id}|{QUANT}|imatrix // baseline=vanilla-FP16"
                row = runner.bench_one(
                    qpath, label, reference_n_params=n_params,
                    eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
                    eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld")
                runner.append_row(CSV, row)
                log(f"  {src}: PPL={row.ppl:.4f} medKLD={row.median_kld:.5f} "
                    f"top_p={row.same_top_p:.4f}%")

    log("")
    log("=== exp-044 IQ4_XS complete ===")
    log(f"  results: {CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""exp-040c: does collecting the imatrix on the FOLDED F16 lift AWQ over imatrix-only?

exp-040b proved the rmsnorm_plus_one fix recovers AWQ (IQ2_XS top_p 0.15%→78.69%),
but it still trails imatrix-only (81.70%). Suspected cause: exp-040's AWQ rows
quantized with the PRE-FOLD imatrix (`imatrix=IMATRIX`, collected on the original
F16). CLAUDE.md is explicit that the AWQ branch must collect its imatrix on the
FOLDED F16 — an unfolded imatrix over-weights exactly the channels AWQ rescaled,
mis-allocating 2-bit precision. That pre-fold imatrix is correct for imatrix-only
(weights unchanged) but wrong for AWQ, which fits the asymmetry we measured.

This isolates that one variable: re-use the folded F16 from exp-040b
(out/exp-040/iq2xs_awq_fix/model-f16-awq.gguf — already on disk, no re-fold),
collect a fresh imatrix ON IT, re-quantize IQ2_XS, bench. Everything else
(scales, per-tensor α, proxy) is identical to exp-040b.

Compare:
  imatrix-only            top_p 81.70%  med_KLD 0.076
  AWQ + pre-fold imatrix  top_p 78.69%  med_KLD 0.122  (exp-040b)
  AWQ + folded imatrix    top_p   ??     med_KLD  ??    (this run)

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040c_folded_imatrix.py
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

EXP40 = REPO / "out" / "exp-040"
F16_GGUF = EXP40 / "model-f16.gguf"            # original, for n_params only
BASELINE_KLD = EXP40 / "baseline.kld"
EVAL_CORPUS = EXP40 / "corpora" / "corpus.eval.txt"
CAL_CORPUS = EXP40 / "corpora" / "corpus.cal.txt"

FOLDED_F16 = EXP40 / "iq2xs_awq_fix" / "model-f16-awq.gguf"  # reuse, no re-fold

OUT = EXP40 / "iq2xs_awq_foldedimat"
QUANT = "IQ2_XS"
EVAL_CTX = 4096
IMATRIX_CTX = 4096


def main() -> int:
    for p in (FOLDED_F16, F16_GGUF, BASELINE_KLD, EVAL_CORPUS, CAL_CORPUS):
        if not p.exists():
            raise FileNotFoundError(f"exp-040c missing reusable input: {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    folded_imatrix = OUT / "imatrix-folded.gguf"
    qpath = OUT / f"{QUANT}-awq-foldedimat.gguf"
    csv_path = OUT / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase("[exp-040c] collect imatrix ON the folded F16"):
        step("imatrix-folded", folded_imatrix,
             lambda: llama_cpp.imatrix(
                 FOLDED_F16, CAL_CORPUS, folded_imatrix,
                 ctx=IMATRIX_CTX, log=log_dir / "imatrix-folded.log"))

    with phase(f"[exp-040c] quantize {QUANT} (folded F16 + folded imatrix)"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 FOLDED_F16, qpath, QUANT, imatrix=folded_imatrix,
                 log=log_dir / "quantize.log"))

    with phase("[exp-040c] bench"):
        label = (f"Jackrong/Qwopus3.5-27B-v3|{QUANT}|"
                 f"awq+folded-imatrix // baseline=FP16")
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS,
            eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX,
            log_dir=log_dir,
            suite="kld",
        )
        runner.append_row(csv_path, row)

    log("")
    log("=== exp-040c verdict (IQ2_XS) ===")
    log(f"  imatrix-only            : top_p 81.70%  med_KLD 0.076")
    log(f"  AWQ + pre-fold imatrix  : top_p 78.69%  med_KLD 0.122  (exp-040b)")
    log(f"  AWQ + folded imatrix    : top_p {row.same_top_p:6.2f}%  med_KLD {row.median_kld:.3f}  (this run)")
    log("  If folded-imatrix AWQ > 81.70%, the pre-fold imatrix was the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

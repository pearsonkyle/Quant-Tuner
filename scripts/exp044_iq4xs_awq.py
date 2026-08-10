"""exp-044 addendum: IQ4_XS AWQ cv-gate for vanilla and QAT sources.

Runs after exp044_iq4xs.py (imatrix-only baseline). AWQ is expected to be
marginal at 4-bit vs 2-bit — this confirms (or surprises) that assumption.

Proxy: int4_g128 (default for 4-bit targets). Plain base imatrix on the
UNFOLDED F16 (Gemma pattern — no folded imatrix re-weight).

    PYTHONPATH=src .venv/bin/python scripts/exp044_iq4xs_awq.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import convert, gguf

EXP44 = REPO / "out" / "exp-044"
CSV   = EXP44 / "results.csv"

VANILLA_HF      = EXP44 / "vanilla" / "model_extracted"
VANILLA_F16     = EXP44 / "vanilla" / "model-f16.gguf"
VANILLA_IMATRIX = EXP44 / "vanilla" / "imatrix-cal.gguf"
CAL_CORPUS      = EXP44 / "corpora" / "corpus.cal.txt"
VAL_CORPUS      = EXP44 / "corpora" / "corpus.val.txt"
EVAL_CORPUS     = EXP44 / "corpora" / "corpus.eval.txt"
BASELINE_KLD    = EXP44 / "baseline.kld"

_EXP22      = REPO / "out" / "exp-022" / "google__gemma-4-31B-it-qat-q4_0-unquantized"
QAT_HF      = _EXP22 / "model_extracted"
QAT_F16     = _EXP22 / "model-f16.gguf"
QAT_IMATRIX = EXP44 / "qat" / "imatrix-cal.gguf"

VANILLA_ID = "google/gemma-4-31B-it"
QAT_ID     = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QUANT      = "IQ4_XS"
EVAL_CTX   = 4096
PROXY_TOKENS          = 1024
CTX                   = 4096
PER_TENSOR_GRID_RADIUS = 0.15
SANITY_MAX_REL        = 1.20
RMSNORM_PLUS_ONE      = False   # Gemma

ROWS = [
    ("vanilla", VANILLA_ID, VANILLA_HF, VANILLA_F16, VANILLA_IMATRIX,
     EXP44 / "v_iq4xs_awq"),
    ("qat",    QAT_ID,     QAT_HF,     QAT_F16,     QAT_IMATRIX,
     EXP44 / "q_iq4xs_awq"),
]


def _in_csv(qpath: Path) -> bool:
    return CSV.exists() and any(str(qpath) in ln for ln in CSV.open())


def run_row(src, model_id, hf, f16, imatrix, sub):
    tag     = f"[exp-044][{src}_iq4xs_awq]"
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle    = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq   = sub / "model-f16-awq.gguf"
    stem      = "gemma-4-31B-it" if src == "vanilla" else "gemma-4-31B-it-qat"
    qpath     = sub / f"{stem}-{QUANT}-awq-cv-gate.gguf"
    proxy     = awq.proxy_for_quant_type(QUANT)   # int4_g128

    with phase(f"{tag} AWQ calibrate (proxy={proxy})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 hf, CAL_CORPUS, bundle,
                 proxy=proxy,
                 proxy_mix=None,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=PER_TENSOR_GRID_RADIUS,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase(f"{tag} AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 hf, bundle, model_awq,
                 device="auto", dtype="bfloat16",
                 rmsnorm_plus_one=RMSNORM_PLUS_ONE,
                 sanity_max_rel=SANITY_MAX_REL,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq,
                                            log=log_dir / "convert.log"))

    with phase(f"{tag} quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(f16_awq, qpath, QUANT, imatrix=imatrix,
                                   log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        if _in_csv(qpath):
            log(f"  {qpath.name}: already in CSV — skipping")
        else:
            n_params = bpw_mod.n_params(f16)
            label = (f"{model_id}|{QUANT}|awq-cv-gate+imatrix|"
                     f"proxy={proxy},mix=None,tokens={PROXY_TOKENS},ctx={CTX}"
                     f" // baseline=vanilla-FP16")
            row = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld")
            runner.append_row(CSV, row)
            log(f"  {src}: PPL={row.ppl:.4f} medKLD={row.median_kld:.5f} "
                f"top_p={row.same_top_p:.4f}%")

    # free the ~115 GB intermediates; keep GGUF + bundle
    for child in (model_awq, f16_awq):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)
    log(f"  freed AWQ intermediates for {src}")


def main() -> int:
    for p in (VANILLA_HF / "config.json", VANILLA_F16, VANILLA_IMATRIX,
              QAT_HF / "config.json", QAT_F16, QAT_IMATRIX,
              CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    proxy = awq.proxy_for_quant_type(QUANT)
    log(f"[exp-044][iq4xs_awq] proxy={proxy}  proxy_tokens={PROXY_TOKENS}  ctx={CTX}")

    for args in ROWS:
        run_row(*args)

    log("")
    log("=== exp-044 IQ4_XS AWQ complete ===")
    log(f"  results: {CSV}")
    log("  Compare medKLD/top_p vs imatrix-only rows to gauge 4-bit AWQ benefit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

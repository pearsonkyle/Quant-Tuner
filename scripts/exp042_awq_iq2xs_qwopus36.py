"""exp-042: does AWQ help IQ2_XS on Qwopus3.6-27B-Coder *now* (windowed corpus)?

We skipped AWQ-on-IQ2_XS before because it barely moved the needle. The tool-call
calibration corpus has since been rebuilt with the stub+multi-window packer
(tool-turn share 88.9%), so the activation statistics AWQ scales on are different
— this re-tests it on the new distribution.

Flow (mirrors the release AWQ branch, exp-040/exp-034, on the existing exp-041 inputs):
  1. AWQ calibrate on the HF model   → awq.pt  (iq2_xs codebook proxy, cv-gate on
     the val corpus, per-tensor α, rmsnorm_plus_one=True for Qwen3.6)
  2. AWQ apply (fold into RMSNorm)    → model_awq/  (sanity_max_rel=0.60)
  3. convert folded HF                → model-f16-awq.gguf
  4. imatrix on the FOLDED F16        → base, re-weighted to hybrid_custom
     (folding rescales activations, so the imatrix MUST be collected post-fold)
  5. quantize → IQ2_XS               + nextn/blk.64 pinned Q8_0
  6. bench vs FP16 on BOTH evals      → corpus.eval.tools.txt + corpus.eval.general.txt
     (reusing exp-041's baseline.tools.kld / baseline.general.kld)

Compare results.csv here against the shipped IQ2_XS imatrix-only row:
  tools  medKLD 0.0462  top_p 74.88%   |   general medKLD 0.0950  top_p 78.87%
If AWQ beats those, it's a ship candidate.

    PYTHONPATH=src .venv/bin/python scripts/exp042_awq_iq2xs_qwopus36.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.calibrate import imatrix as imatrix_cal
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import convert, gguf

MODEL_ID = "Jackrong/Qwopus3.6-27B-Coder"
EXP41 = REPO / "out" / "exp-041"
EXP42 = REPO / "out" / "exp-042"

HF_DIR = EXP41 / "model_extracted"
F16_GGUF = EXP41 / "model-f16.gguf"
CORPORA = EXP41 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
TOOLS_CORPUS = CORPORA / "corpus.eval.tools.txt"
GEN_CORPUS = CORPORA / "corpus.eval.general.txt"
TOOLS_BASELINE = EXP41 / "baseline.tools.kld"
GEN_BASELINE = EXP41 / "baseline.general.kld"

QUANT = "IQ2_XS"
CTX = 4096
EVAL_CTX = 4096
PROXY_TOKENS = 1024
NEXTN_PIN = {"blk.64.": "q8_0"}

SUB = EXP42 / "iq2xs_awq"
BUNDLE = SUB / "awq.pt"
MODEL_AWQ = SUB / "model_awq"
F16_AWQ = SUB / "model-f16-awq.gguf"
BASE_IMATRIX = SUB / "imatrix-awq-base.gguf"
HYBRID_IMATRIX = SUB / "imatrix-awq-hybrid_custom.gguf"
QPATH = SUB / "Qwopus3.6-27B-Coder-IQ2_XS-awq-mtp.gguf"
CSV = EXP42 / "results.csv"


def _csv_has(path: Path, needle: str) -> bool:
    return path.exists() and any(needle in ln for ln in path.open())


def _bench(eval_name: str, corpus: Path, baseline: Path, n_params: int) -> None:
    label = (f"{MODEL_ID}|{QUANT}|awq-cv-gate+imatrix+nextn@Q8 // eval={eval_name} "
             f"// baseline=FP16")
    if _csv_has(CSV, f"eval={eval_name} ") and _csv_has(CSV, str(QPATH)):
        log(f"  {eval_name}: already in CSV — skipping")
        return
    r = runner.bench_one(
        QPATH, label, reference_n_params=n_params,
        eval_dataset=corpus, eval_baseline=baseline,
        eval_ctx=EVAL_CTX, log_dir=SUB / "logs", suite="kld",
    )
    runner.append_row(CSV, r)
    log(f"  [{eval_name}] PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} "
        f"top_p={r.same_top_p:.2f}%")


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, CAL_CORPUS, VAL_CORPUS,
              TOOLS_CORPUS, GEN_CORPUS, TOOLS_BASELINE, GEN_BASELINE):
        if not p.exists():
            raise FileNotFoundError(f"exp-042 missing input: {p}")
    (SUB / "logs").mkdir(parents=True, exist_ok=True)
    log_dir = SUB / "logs"
    n_params = bpw_mod.n_params(F16_GGUF)
    proxy = awq.proxy_for_quant_type(QUANT)
    log(f"[exp-042] AWQ IQ2_XS  proxy={proxy}  proxy_tokens={PROXY_TOKENS}  ctx={CTX}")

    with phase("[exp-042] AWQ calibrate (cv-gate, per-tensor α, plus_one=True)"):
        step("AWQ calibrate", BUNDLE,
             lambda: awq.calibrate(
                 HF_DIR, CAL_CORPUS, BUNDLE,
                 proxy=proxy,
                 proxy_mix=None,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase("[exp-042] AWQ apply (fold into RMSNorm)"):
        step("AWQ apply", MODEL_AWQ / "config.json",
             lambda: awq.apply(
                 HF_DIR, BUNDLE, MODEL_AWQ,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=True,   # Qwen3.6 (1+γ) norm — see exp-040
                 sanity_max_rel=0.60,
             ))

    with phase("[exp-042] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", F16_AWQ,
             lambda: convert.hf_to_f16_gguf(MODEL_AWQ, F16_AWQ,
                                            log=log_dir / "convert.log"))

    with phase("[exp-042] base imatrix on FOLDED F16"):
        step("imatrix-base", BASE_IMATRIX,
             lambda: llama_cpp.imatrix(F16_AWQ, CAL_CORPUS, BASE_IMATRIX,
                                       ctx=CTX, log=log_dir / "imatrix-base.log"))

    with phase("[exp-042] re-weight imatrix -> hybrid_custom"):
        step("imatrix-hybrid", HYBRID_IMATRIX,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=F16_AWQ,
                 base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))

    with phase(f"[exp-042] quantize {QUANT} (nextn@Q8)"):
        step("quantize", QPATH,
             lambda: gguf.quantize(F16_AWQ, QPATH, QUANT, imatrix=HYBRID_IMATRIX,
                                   tensor_types=NEXTN_PIN,
                                   log=log_dir / "quantize.log"))

    with phase("[exp-042] bench vs FP16 (tools + general)"):
        _bench("TOOLS", TOOLS_CORPUS, TOOLS_BASELINE, n_params)
        _bench("GENERAL", GEN_CORPUS, GEN_BASELINE, n_params)

    log("")
    log("=== exp-042 complete ===")
    log(f"  results: {CSV}")
    log("  Compare vs shipped IQ2_XS imatrix-only: tools medKLD 0.0462 / top_p 74.88%,")
    log("                                          general medKLD 0.0950 / top_p 78.87%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

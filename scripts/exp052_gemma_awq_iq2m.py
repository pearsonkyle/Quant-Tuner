"""exp-052: does 2-bit AWQ beat imatrix agentically on gemma-4-31B?

exp-051 found that on Ornith-1.0-9B, IQ2_M AWQ dramatically out-tool-called the
plain imatrix IQ2_M (param-acc 5%→33%) even though static KLD slightly favored
imatrix. Gemma is a better/cleaner AWQ test than Ornith: its 60 layers are all
standard attention (full_attention every 6th + sliding_attention elsewhere, all
with q/k/v/o_proj), so AWQ scales EVERY layer — not the partial coverage Ornith's
linear-attention trunk allowed.

This builds ONE quant — AWQ IQ2_M on the fixed strided calibration — and benches
it. The imatrix IQ2_M baseline is the already-shipped
`gemma-4-31B-it-IQ2_M.gguf`; the agentic head-to-head is a separate step
(run_toolcall_reps over both GGUFs).

Reuses exp-044 vanilla F16 + extracted HF + calibration corpus. With strided
ingestion (awq.calibrate now samples the whole corpus, tokens=65536 default), the
existing wiki-prepended corpus still yields good tool-log coverage.

⚠️ rmsnorm_plus_one for gemma is ambiguous: exp-044 used False, but gemma's (1+γ)
norm theoretically wants True. The AWQ apply step (which runs BEFORE the expensive
folded-imatrix pass) reports the F16 logit-drift `rel`; a wrong fold collapses to
rel≈1.0. We set sanity_max_rel=0.6 so a bad fold ABORTS cheaply — flip PLUS_ONE
and re-run (calibrate is cached) if that happens.

    PYTHONPATH=src .venv/bin/python scripts/exp052_gemma_awq_iq2m.py
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

MODEL_ID = "google/gemma-4-31B-it"

# --- reuse exp-044 vanilla artifacts ---------------------------------------
EXP44 = REPO / "out" / "exp-044"
HF_DIR = EXP44 / "vanilla" / "model_extracted"
F16_GGUF = EXP44 / "vanilla" / "model-f16.gguf"
CAL_CORPUS = EXP44 / "corpora" / "corpus.cal.txt"
EVAL_CORPUS = EXP44 / "corpora" / "corpus.eval.txt"
EVAL_BASELINE = EXP44 / "baseline.kld"

# imatrix IQ2_M baseline (already shipped) — used by the agentic head-to-head.
SHIPPED_IMATRIX_IQ2M = (
    REPO / "uploads" / "pearsonkyle" / "gemma-4-31b-it-imatrix-GGUF"
    / "gemma-4-31B-it-IQ2_M.gguf"
)

# --- workspace -------------------------------------------------------------
EXP = REPO / "out" / "exp-052"
LOGS = EXP / "logs"

QUANT = "IQ2_M"
PROXY = "q2k_b16"        # recipe pin for IQ2_M (pure iq2_s regresses)
PROXY_MIX = "IQ2_M"      # route v_proj→Q4_K, o_proj/down-first-eighth→IQ3_S
CTX = 4096
EVAL_CTX = 4096
# Gemma convention (exp-044). If AWQ apply aborts on drift, flip to True.
PLUS_ONE = False
SANITY_MAX_REL = 0.6     # catch a collapsed fold before the 1.5h imatrix pass


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, CAL_CORPUS, EVAL_CORPUS, EVAL_BASELINE):
        if not p.exists():
            raise FileNotFoundError(f"exp-052 missing input: {p}")
    LOGS.mkdir(parents=True, exist_ok=True)
    if not SHIPPED_IMATRIX_IQ2M.exists():
        log(f"[exp-052] WARN: shipped imatrix baseline not found at "
            f"{SHIPPED_IMATRIX_IQ2M} — agentic head-to-head will need it")
    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"[exp-052] reference n_params = {n_params:,.0f}")

    bundle = EXP / "awq.pt"
    model_awq = EXP / "model_awq"
    f16_awq = EXP / "model-f16-awq.gguf"
    base_im = EXP / "imatrix-awq-base.gguf"
    hybrid_im = EXP / "imatrix-awq-hybrid_custom.gguf"
    qpath = EXP / f"gemma-4-31B-it-{QUANT}-awq.gguf"
    csv_path = EXP / "results.csv"

    with phase(f"[exp-052] AWQ calibrate (strided, proxy={PROXY}, mix={PROXY_MIX})"):
        step("awq-calibrate", bundle,
             lambda: awq.calibrate(
                 HF_DIR, CAL_CORPUS, bundle,
                 proxy=PROXY, proxy_mix=PROXY_MIX, ctx=CTX,
                 device="auto", dtype="bfloat16",
                 per_tensor_alpha=False, cv_strategy="off"))

    # GATE: a wrong PLUS_ONE collapses the F16 fold -> apply raises here, BEFORE
    # the expensive imatrix pass. Flip PLUS_ONE=True and re-run if it aborts.
    with phase(f"[exp-052] AWQ apply (rmsnorm_plus_one={PLUS_ONE}, sanity≤{SANITY_MAX_REL})"):
        step("awq-apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, bundle, model_awq, device="auto", dtype="bfloat16",
                 rmsnorm_plus_one=PLUS_ONE, sanity_max_rel=SANITY_MAX_REL))

    with phase("[exp-052] convert AWQ-folded HF -> F16 GGUF"):
        step("awq-convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq, log=LOGS / "convert.log"))

    with phase("[exp-052] imatrix on FOLDED F16 (~1.5h on 31B)"):
        step("awq-imatrix-base", base_im,
             lambda: llama_cpp.imatrix(f16_awq, CAL_CORPUS, base_im,
                                       ctx=CTX, log=LOGS / "imatrix.log"))
    with phase("[exp-052] re-weight folded -> hybrid_custom"):
        step("awq-imatrix-hybrid", hybrid_im,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=f16_awq,
                 base_imatrix=base_im, out_path=hybrid_im))

    with phase(f"[exp-052] quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(f16_awq, qpath, QUANT, imatrix=hybrid_im,
                                   log=LOGS / "quantize.log"))

    with phase("[exp-052] KLD bench (exp-044 external eval)"):
        r = runner.bench_one(
            qpath, f"{MODEL_ID}|{QUANT}|awq+imatrix // eval=external // baseline=FP16",
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS, eval_baseline=EVAL_BASELINE,
            eval_ctx=EVAL_CTX, log_dir=LOGS, suite="kld")
        runner.append_row(csv_path, r)
        log(f"  AWQ {QUANT}: bpw={r.bpw:.3f} PPL={r.ppl:.4f} "
            f"medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-052 complete ===")
    log(f"  AWQ quant: {qpath}")
    log(f"  imatrix baseline (shipped): {SHIPPED_IMATRIX_IQ2M}")
    log("  Next: agentic head-to-head via run_toolcall_reps over both GGUFs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

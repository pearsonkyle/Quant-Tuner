"""Experiment 023 (smoke): does doubling AWQ proxy_tokens fix QAT-IQ2_XS?

Hypothesis. In exp-022, AWQ-cv-gate on QAT-source IQ2_XS landed at KLD 1.881
— marginally *worse* than imatrix-only at the same bit budget (1.859). The
α grid is asking the 256-token proxy to discriminate per-tensor reconstruction
differences that may be inside its noise floor (std-error ≈ 1/√256 per channel).
The cv-gate prevents *bad* α picks from going through, but it can't promote a
*better* α candidate that was missed because of cal-side variance.

Smoke test: rerun the same exp-022 cv-gate pipeline with proxy_tokens=512
(double), IQ2_XS only, QAT source only. All other inputs identical (imatrix,
baseline, corpora — all reused from exp-022 outputs).

Decision criterion (written down before the run):
  - KLD < 1.86  → noise hypothesis right; queue full sweep at 512 (+1024).
  - KLD ≈ 1.88  → structural overcorrection; close the door on AWQ-for-QAT-IQ2_XS.
  - KLD > 1.88  → AWQ liked the noise; write up as surprise.

Idempotent via experiments.step(). Reuses ~all of exp-022 except AWQ scales
and downstream IQ2_XS quant.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import convert, gguf

QAT_REPO_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QAT_SLUG = QAT_REPO_ID.replace("/", "__")
VANILLA_SLUG = "google__gemma-4-31B-it"

EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP22 = REPO / "out" / "exp-022" / QAT_SLUG
EXP23 = REPO / "out" / "exp-023" / QAT_SLUG
LOGS = EXP23 / "logs"

EVAL_CTX = 4096
PROXY_TOKENS = 512  # was 256 in exp-022
QUANT = "IQ2_XS"

SRC_MODEL = EXP22 / "model_extracted"     # reuse the stripped QAT HF dir
IMATRIX = EXP22 / "imatrix-cal.gguf"      # reuse QAT imatrix at ctx=4096
CAL_CORPUS = EXP20 / "corpora" / "corpus.cal.txt"
VAL_CORPUS = EXP20 / "corpora" / "corpus.val.txt"
EVAL_CORPUS = EXP20 / "corpora" / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"
VANILLA_F16 = REPO / "out" / "exp-009" / VANILLA_SLUG / "model-f16.gguf"


def _check_inputs() -> None:
    missing = [p for p in (SRC_MODEL / "config.json", IMATRIX, CAL_CORPUS,
                            VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD, VANILLA_F16)
               if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "exp-023 reuses exp-022 QAT model+imatrix and exp-020 corpora+baseline:\n  "
            + names
        )


def main() -> int:
    _check_inputs()
    EXP23.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    awq_bundle = EXP23 / "awq.pt"
    model_awq = EXP23 / "model_awq"
    f16_awq = EXP23 / "model-f16-awq.gguf"
    qpath = EXP23 / f"{QUANT}-awq.gguf"
    csv_path = EXP23 / "results.csv"

    with phase(f"[exp-023] AWQ calibrate (proxy_tokens={PROXY_TOKENS}, cv_strategy=gate)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, CAL_CORPUS, awq_bundle,
                 proxy_tokens=PROXY_TOKENS,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase("[exp-023] AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase("[exp-023] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq,
                 log=LOGS / "convert.log"))

    with phase(f"[exp-023] quantize {QUANT}"):
        step(f"quantize {QUANT}", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=LOGS / f"quantize-{QUANT}.log"))

    n_params = bpw_mod.n_params(VANILLA_F16)
    label = (
        f"{QAT_REPO_ID}|{QUANT}|awq-cv-gate+imatrix|"
        f"proxy_tokens={PROXY_TOKENS} // baseline=vanilla-FP16"
    )

    with phase(f"[exp-023] bench {QUANT}"):
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS,
            eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX,
            log_dir=LOGS,
            suite="kld",
        )
        runner.append_row(csv_path, row)

    log("")
    log(f"=== exp-023 smoke result (proxy_tokens={PROXY_TOKENS}, QAT-IQ2_XS) ===")
    log(f"  size={row.size_gib:.2f} GiB  bpw={row.bpw:.3f}")
    log(f"  PPL          = {row.ppl:.4f}")
    log(f"  mean KLD     = {row.mean_kld:.5f}")
    log(f"  same_top_p   = {row.same_top_p:.4f}%")
    log("")
    log("  exp-022 reference rows (QAT-IQ2_XS, proxy_tokens=256):")
    log("    AWQ cv-gate :  PPL 153.29   KLD 1.881   top_p 48.78%")
    log("    imatrix only:  PPL 209.00   KLD 1.859   top_p 47.09%")
    log("")
    if row.mean_kld < 1.86:
        log("  -> KLD beats imatrix-only floor. Noise hypothesis supported;")
        log("     consider running 1024 + adding Q2_K_S.")
    elif row.mean_kld <= 1.89:
        log("  -> KLD within exp-022 noise band. Likely structural overcorrection.")
        log("     Close the door on AWQ-for-QAT-IQ2_XS; ship imatrix-only as IQ2_XS pick.")
    else:
        log("  -> KLD worse than exp-022. AWQ grid liked the noise. Surprise — write up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

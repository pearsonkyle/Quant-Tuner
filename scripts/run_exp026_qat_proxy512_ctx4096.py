"""Experiment 026: diagnostic — QAT source, AWQ proxy=512, ctx=4096.

exp-025 adopted the project-wide AWQ defaults proxy=2048, ctx=4096 and saw
the QAT-Q2_K_S "headline SOTA" row regress from KLD 1.736 → 2.105 (versus
exp-022's proxy=256, ctx=1024). The regression could be caused by either
the ctx change, the proxy_tokens change, or their combination.

This diagnostic fills the missing cell of the 2×2 grid. After it lands we'll
have, for QAT-IQ2_XS:

      ctx       proxy=256     proxy=512    proxy=1024    proxy=2048
      ----      ---------     ---------    ----------    ----------
      1024      1.881         **1.782**    1.822         (would cap)
      4096      —             ? (new)      1.876         1.852

And for QAT-Q2_K_S:

      ctx       proxy=256     proxy=512    proxy=2048
      ----      ---------     ---------    ----------
      1024      **1.736**     —            —
      4096      —             ? (new)      2.105

Decision tree (write down before the run):
  - IQ2_XS at proxy=512,ctx=4096 lands near 1.78 → ctx=4096 isn't the root cause;
    proxy=2048 overshoots. Adopt proxy=512+ctx=4096 as the project default.
  - IQ2_XS lands near 1.85 → ctx=4096 itself is worse than ctx=1024.
    Revert ctx default to 1024.
  - IQ2_XS lands in between (1.80-1.84) → both factors contribute; recommend
    proxy=512+ctx=1024 as the empirical optimum we already knew about.

Reuses everything from exp-022 + exp-020. Single AWQ calibrate, two quants.
Wall-time estimate: ~50 min (calibrate 7 + apply 5 + convert 2 + 2×(quant+bench)).

NOTE: This script explicitly overrides the new project defaults because we're
running a controlled experiment, not a production calibrate.
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
EXP26 = REPO / "out" / "exp-026" / QAT_SLUG
LOGS = EXP26 / "logs"

EVAL_CTX = 4096
PROXY_TOKENS = 512
CALIBRATE_CTX = 4096
QUANTS = ["IQ2_XS", "Q2_K_S"]

SRC_MODEL = EXP22 / "model_extracted"
IMATRIX = EXP22 / "imatrix-cal.gguf"
CAL_CORPUS = EXP20 / "corpora" / "corpus.cal.txt"
VAL_CORPUS = EXP20 / "corpora" / "corpus.val.txt"
EVAL_CORPUS = EXP20 / "corpora" / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"
VANILLA_F16 = REPO / "out" / "exp-009" / VANILLA_SLUG / "model-f16.gguf"


def _check_inputs() -> None:
    missing = [p for p in (
        SRC_MODEL / "config.json", IMATRIX,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD, VANILLA_F16,
    ) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError("exp-026 missing inputs:\n  " + names)


def main() -> int:
    _check_inputs()
    EXP26.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    awq_bundle = EXP26 / "awq.pt"
    model_awq = EXP26 / "model_awq"
    f16_awq = EXP26 / "model-f16-awq.gguf"
    csv_path = EXP26 / "results.csv"

    with phase(f"[exp-026] AWQ calibrate (proxy={PROXY_TOKENS}, ctx={CALIBRATE_CTX})"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, CAL_CORPUS, awq_bundle,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CALIBRATE_CTX,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase("[exp-026] AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase("[exp-026] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=LOGS / "convert.log"))

    n_params = bpw_mod.n_params(VANILLA_F16)

    rows: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = EXP26 / f"{q}-awq.gguf"
        with phase(f"[exp-026] quantize {q}"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=IMATRIX,
                     log=LOGS / f"quantize-{qt}.log"))
        label = (
            f"{QAT_REPO_ID}|{q}|awq-cv-gate+imatrix|"
            f"proxy={PROXY_TOKENS},ctx={CALIBRATE_CTX} // baseline=vanilla-FP16"
        )
        with phase(f"[exp-026] bench {q}"):
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
            log(f"  {q}: PPL={row.ppl:.4f} KLD={row.mean_kld:.5f} "
                f"top_p={row.same_top_p:.4f}%")
            rows[q] = row

    log("")
    log(f"=== exp-026 (QAT, proxy={PROXY_TOKENS}, ctx={CALIBRATE_CTX}) ===")
    log("                              KLD       top_p     PPL")
    log("  exp-022 IQ2_XS  proxy=256,ctx=1024:  1.881   48.78%   153.29")
    log("  exp-023 IQ2_XS  proxy=512,ctx=1024:  1.782   49.72%   112.57   <-- prior empirical best")
    log("  exp-024 IQ2_XS  proxy=1024,ctx=4096: 1.876   49.08%   143.51")
    log("  exp-024 IQ2_XS  proxy=2048,ctx=4096: 1.852   49.47%   123.57")
    log(f"  exp-026 IQ2_XS  proxy={PROXY_TOKENS},ctx={CALIBRATE_CTX}: "
        f"{rows['IQ2_XS'].mean_kld:.3f}   {rows['IQ2_XS'].same_top_p:.2f}%   "
        f"{rows['IQ2_XS'].ppl:.2f}")
    log("")
    log("  exp-022 Q2_K_S  proxy=256,ctx=1024:  1.736   51.26%    90.28   <-- prior SOTA")
    log("  exp-025 Q2_K_S  proxy=2048,ctx=4096: 2.105   48.39%    81.31")
    log(f"  exp-026 Q2_K_S  proxy={PROXY_TOKENS},ctx={CALIBRATE_CTX}: "
        f"{rows['Q2_K_S'].mean_kld:.3f}   {rows['Q2_K_S'].same_top_p:.2f}%   "
        f"{rows['Q2_K_S'].ppl:.2f}")
    log("")
    iq_kld = rows["IQ2_XS"].mean_kld
    if iq_kld < 1.80:
        log("  -> IQ2_XS near 1.78. ctx=4096 isn't the root cause; proxy=2048 overshoots.")
        log("     Recommend adopting proxy=512+ctx=4096 as the new project default.")
    elif iq_kld > 1.84:
        log("  -> IQ2_XS near 1.85. ctx=4096 itself is worse than ctx=1024.")
        log("     Recommend reverting ctx default to 1024 (or revisiting the consistency call).")
    else:
        log("  -> IQ2_XS between 1.80 and 1.84: both factors contribute.")
        log("     proxy=512+ctx=1024 remains the empirical optimum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

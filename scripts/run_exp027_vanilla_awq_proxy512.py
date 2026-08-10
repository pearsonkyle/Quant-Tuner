"""Experiment 027: vanilla-source AWQ cv-gate at the new defaults.

Following exp-026 we adopted proxy_tokens=512, ctx=4096 as the project-wide
AWQ defaults (best KLD on QAT-IQ2_XS). The vanilla `google/gemma-4-31B-it`
AWQ-cv-gate quants in the upload dir were produced under the previous
defaults (proxy=256, ctx=1024 — exp-020), so they're no longer aligned with
the recipe documented in the HF README.

This regenerates the vanilla AWQ-cv-gate quants under the new defaults:
IQ2_XS, IQ2_M, Q2_K_S — all three.

Reuses everything else:
  - imatrix:        exp-020 (ctx=4096, unchanged)
  - corpora:        exp-020
  - baseline.kld:   exp-020

Compute estimate ~125 min: calibrate 7 + apply 5 + convert 2 + 3*(quant ~12
+ bench ~9) = ~85 min. Quantize+bench dominate.

QAT-source AWQ at the new defaults is already in exp-026 (IQ2_XS, Q2_K_S);
the upload step (separate) will copy from there.
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

VANILLA_REPO_ID = "google/gemma-4-31B-it"
VANILLA_SLUG = VANILLA_REPO_ID.replace("/", "__")

EXP09 = REPO / "out" / "exp-009" / VANILLA_SLUG
EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP27 = REPO / "out" / "exp-027" / VANILLA_SLUG
LOGS = EXP27 / "logs"

EVAL_CTX = 4096
QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]

SRC_MODEL = EXP09 / "model_extracted"
SRC_F16 = EXP09 / "model-f16.gguf"
IMATRIX = EXP20 / "imatrix-cal.gguf"
CAL_CORPUS = EXP20 / "corpora" / "corpus.cal.txt"
VAL_CORPUS = EXP20 / "corpora" / "corpus.val.txt"
EVAL_CORPUS = EXP20 / "corpora" / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"


def _check_inputs() -> None:
    missing = [p for p in (
        SRC_MODEL / "config.json", SRC_F16, IMATRIX,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError("exp-027 missing inputs:\n  " + names)


def main() -> int:
    _check_inputs()
    EXP27.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    awq_bundle = EXP27 / "awq.pt"
    model_awq = EXP27 / "model_awq"
    f16_awq = EXP27 / "model-f16-awq.gguf"
    csv_path = EXP27 / "results.csv"

    with phase("[exp-027] AWQ calibrate (defaults: proxy=512, ctx=4096)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, CAL_CORPUS, awq_bundle,
                 # Defaults explicitly listed for self-documentation.
                 proxy_tokens=512,
                 ctx=4096,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase("[exp-027] AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase("[exp-027] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=LOGS / "convert.log"))

    n_params = bpw_mod.n_params(SRC_F16)
    rows: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = EXP27 / f"{q}-awq.gguf"
        with phase(f"[exp-027] quantize {q}"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=IMATRIX,
                     log=LOGS / f"quantize-{qt}.log"))
        label = (
            f"{VANILLA_REPO_ID}|{q}|awq-cv-gate+imatrix|"
            f"proxy=512,ctx=4096 // baseline=vanilla-FP16"
        )
        with phase(f"[exp-027] bench {q}"):
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
    log("=== exp-027 (vanilla source, AWQ cv-gate at proxy=512, ctx=4096) ===")
    log("                              KLD       top_p     PPL")
    log("  exp-020 (published, proxy=256, ctx=1024):")
    log("    IQ2_XS:  3.709   46.18%   237.63")
    log("    IQ2_M :  2.905   49.38%   299.07")
    log("    Q2_K_S:  4.514   47.64%    51.52")
    log("  exp-025 (proxy=2048, ctx=4096):")
    log("    IQ2_XS:  3.606   40.82%  1493.08")
    log("    IQ2_M :  3.119   45.60%  1206.74")
    log("    Q2_K_S:  3.520   49.38%    90.26")
    log("  exp-027 (proxy=512, ctx=4096) — NEW:")
    for q in QUANTS:
        r = rows[q]
        log(f"    {q:7s}: {r.mean_kld:.3f}   {r.same_top_p:.2f}%   {r.ppl:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

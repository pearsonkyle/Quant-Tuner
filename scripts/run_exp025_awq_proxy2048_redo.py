"""Experiment 025: re-run all AWQ-cv-gate arms at the new defaults.

After exp-024 we adopted project-wide AWQ defaults:
    ctx          = 4096   (was 1024; aligned with imatrix/eval ctx=4096)
    proxy_tokens = 2048   (was 256;  widest setting tested in exp-024)

Even though exp-024 showed proxy=512 + ctx=1024 was the empirical optimum on
QAT-Gemma-4-31B-IQ2_XS, the user chose the consistent project-wide defaults
over per-quant tuning. This experiment regenerates the published AWQ-cv-gate
release at the new defaults so the table in the HF README matches.

Two sources:
  - vanilla  google/gemma-4-31B-it           → IQ2_XS, IQ2_M, Q2_K_S
  - QAT      google/gemma-4-31B-it-qat-q4_0-unquantized → IQ2_XS, Q2_K_S
                                                          (IQ2_M is broken on QAT)

QAT-IQ2_XS at proxy=2048,ctx=4096 was already produced by exp-024 — we link
to that GGUF rather than redoing the calibrate/quantize cycle.

Everything else reuses existing artifacts:
  - vanilla  : exp-020 imatrix, baseline.kld, corpora
  - QAT      : exp-022 imatrix; exp-020 baseline.kld + corpora

Idempotent via experiments.step(). Wall-time estimate ~3.5h:
  - vanilla calibrate (proxy=2048) ~60min + apply 5 + convert 2 + 3×(quant 13 + bench 9) ~125min
  - QAT calibrate ~60min + apply 5 + convert 2 + 1×(quant 13 + bench 9) ~85min
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

VANILLA_REPO_ID = "google/gemma-4-31B-it"
QAT_REPO_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
VANILLA_SLUG = VANILLA_REPO_ID.replace("/", "__")
QAT_SLUG = QAT_REPO_ID.replace("/", "__")

EXP09 = REPO / "out" / "exp-009" / VANILLA_SLUG
EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP22 = REPO / "out" / "exp-022" / QAT_SLUG
EXP24 = REPO / "out" / "exp-024"
EXP25 = REPO / "out" / "exp-025"

EVAL_CTX = 4096

VANILLA_SRC_MODEL = EXP09 / "model_extracted"
VANILLA_IMATRIX = EXP20 / "imatrix-cal.gguf"
QAT_SRC_MODEL = EXP22 / "model_extracted"
QAT_IMATRIX = EXP22 / "imatrix-cal.gguf"

CAL_CORPUS = EXP20 / "corpora" / "corpus.cal.txt"
VAL_CORPUS = EXP20 / "corpora" / "corpus.val.txt"
EVAL_CORPUS = EXP20 / "corpora" / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"
VANILLA_F16 = EXP09 / "model-f16.gguf"


def _check_inputs() -> None:
    missing = [p for p in (
        VANILLA_SRC_MODEL / "config.json", VANILLA_F16, VANILLA_IMATRIX,
        QAT_SRC_MODEL / "config.json", QAT_IMATRIX,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError("exp-025 missing inputs:\n  " + names)


def _calibrate_and_apply(src_model: Path, src_label: str, arm_dir: Path) -> Path:
    """Run AWQ calibrate + apply + convert. Returns the F16-awq GGUF path."""
    logs = arm_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    awq_bundle = arm_dir / "awq.pt"
    model_awq = arm_dir / "model_awq"
    f16_awq = arm_dir / "model-f16-awq.gguf"

    with phase(f"[{src_label}] AWQ calibrate (defaults: proxy=2048, ctx=4096)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 src_model, CAL_CORPUS, awq_bundle,
                 # proxy_tokens=2048 and ctx=4096 are the new defaults
                 # (exp-024). Listed explicitly here so the recipe is
                 # self-documenting if the defaults change again later.
                 proxy_tokens=2048,
                 ctx=4096,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase(f"[{src_label}] AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 src_model, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase(f"[{src_label}] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=logs / "convert.log"))

    return f16_awq


def _quantize_and_bench(
    f16_awq: Path, imatrix: Path, quant: str, arm_dir: Path,
    src_label: str, repo_id: str, n_params: float, csv_path: Path,
) -> runner.BenchRow:
    qpath = arm_dir / f"{quant}-awq.gguf"
    logs = arm_dir / "logs"
    with phase(f"[{src_label}] quantize {quant}"):
        step(f"quantize {quant}", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, quant, imatrix=imatrix,
                 log=logs / f"quantize-{quant}.log"))
    label = (
        f"{repo_id}|{quant}|awq-cv-gate+imatrix|"
        f"proxy=2048,ctx=4096 // baseline=vanilla-FP16"
    )
    with phase(f"[{src_label}] bench {quant}"):
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS,
            eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX,
            log_dir=logs,
            suite="kld",
        )
        runner.append_row(csv_path, row)
        log(f"  {quant}: PPL={row.ppl:.4f} KLD={row.mean_kld:.5f} "
            f"top_p={row.same_top_p:.4f}%")
    return row


def _link_existing_qat_iq2xs(arm_dir: Path) -> Path:
    """exp-024 already produced QAT-IQ2_XS at proxy=2048, ctx=4096. Reuse it."""
    src = EXP24 / QAT_SLUG / "proxy2048" / "IQ2_XS-awq.gguf"
    dst = arm_dir / "IQ2_XS-awq.gguf"
    if not dst.exists():
        if not src.exists():
            raise FileNotFoundError(
                f"Expected exp-024 QAT-IQ2_XS at proxy=2048: {src}"
            )
        arm_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log(f"reused exp-024 QAT-IQ2_XS proxy=2048 -> {dst.relative_to(REPO)}")
    return dst


def main() -> int:
    _check_inputs()
    EXP25.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(VANILLA_F16)
    log(f"reference n_params (from vanilla FP16) = {n_params:,.0f}")

    csv_path = EXP25 / "results.csv"

    # ---- Vanilla source: AWQ calibrate once, 3 quants ----
    vanilla_arm = EXP25 / "vanilla"
    vanilla_f16_awq = _calibrate_and_apply(
        VANILLA_SRC_MODEL, "vanilla", vanilla_arm,
    )
    vanilla_rows: dict[str, runner.BenchRow] = {}
    for q in ("IQ2_XS", "IQ2_M", "Q2_K_S"):
        vanilla_rows[q] = _quantize_and_bench(
            vanilla_f16_awq, VANILLA_IMATRIX, q, vanilla_arm,
            "vanilla", VANILLA_REPO_ID, n_params, csv_path,
        )

    # ---- QAT source: AWQ calibrate once, Q2_K_S only; reuse exp-024 IQ2_XS ----
    qat_arm = EXP25 / "qat"
    qat_f16_awq = _calibrate_and_apply(
        QAT_SRC_MODEL, "qat", qat_arm,
    )
    qat_rows: dict[str, runner.BenchRow] = {}
    qat_rows["Q2_K_S"] = _quantize_and_bench(
        qat_f16_awq, QAT_IMATRIX, "Q2_K_S", qat_arm,
        "qat", QAT_REPO_ID, n_params, csv_path,
    )

    # QAT-IQ2_XS already exists from exp-024 at the same recipe.
    qat_iq2xs = _link_existing_qat_iq2xs(qat_arm)
    label = (
        f"{QAT_REPO_ID}|IQ2_XS|awq-cv-gate+imatrix|"
        f"proxy=2048,ctx=4096 // baseline=vanilla-FP16 // reused from exp-024"
    )
    with phase("[qat] bench IQ2_XS (reused from exp-024)"):
        row = runner.bench_one(
            qat_iq2xs, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS,
            eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX,
            log_dir=qat_arm / "logs",
            suite="kld",
        )
        runner.append_row(csv_path, row)
        log(f"  IQ2_XS: PPL={row.ppl:.4f} KLD={row.mean_kld:.5f} "
            f"top_p={row.same_top_p:.4f}%")
        qat_rows["IQ2_XS"] = row

    log("")
    log("=== exp-025 results (all AWQ cv-gate at proxy=2048, ctx=4096) ===")
    log("  vanilla google/gemma-4-31B-it source:")
    for q, r in vanilla_rows.items():
        log(f"    {q:7s}: PPL={r.ppl:>10.4f}  KLD={r.mean_kld:.5f}  "
            f"top_p={r.same_top_p:.4f}%")
    log("  QAT google/gemma-4-31B-it-qat-q4_0-unquantized source:")
    for q, r in qat_rows.items():
        log(f"    {q:7s}: PPL={r.ppl:>10.4f}  KLD={r.mean_kld:.5f}  "
            f"top_p={r.same_top_p:.4f}%")
    log("")
    log("Compare against the published README numbers to decide whether to "
        "republish (small wins may not justify the file swap).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

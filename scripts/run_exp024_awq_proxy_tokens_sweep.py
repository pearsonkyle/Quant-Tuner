"""Experiment 024: AWQ proxy_tokens sweep at 1024 and 2048 (QAT-IQ2_XS).

Follow-on to exp-023. exp-023 confirmed that doubling proxy_tokens from 256 to
512 on QAT-IQ2_XS dropped mean KLD from 1.881 → 1.782 (below the imatrix-only
floor of 1.859) and lifted same_top_p from 48.78% → 49.72%. The cal-side noise
floor was real; tightening it lets per-tensor α refinement land cleaner picks
that the cv-gate already lets through.

This experiment maps the rest of the proxy_tokens curve. Two arms:

  1. proxy_tokens=1024 (4× exp-022 default)
  2. proxy_tokens=2048 (8× exp-022 default)

IQ2_XS only — we want a clean shape on the curve before expanding to Q2_K_S.
QAT source only — that is where the symptom is.

Decision criteria (write down before the run):
  - 1024 KLD < 1.78 - 0.02 → curve still climbing past 512; 2048 likely worth it.
  - 1024 KLD ≈ 1.78 ± 0.01  → plateau hit at 512; 2048 should confirm.
  - 2048 < 1024 < 512 monotonically → noise floor not yet reached; queue 4096.
  - 2048 ≥ 1024 → plateau confirmed. Pick the cheapest value that's within the
    plateau and adopt it as the new default for the QAT recipe.

Wall-time estimate (extrapolating from exp-023): ~50 min for 1024, ~80 min
for 2048. Total ~130 min. AWQ apply / convert / quantize / bench are constant;
calibrate is the linearly-scaling cost.

Idempotent via experiments.step(). Reuses exp-022 QAT HF model + imatrix and
exp-020 corpora + baseline; nothing else.
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
EXP24 = REPO / "out" / "exp-024" / QAT_SLUG

EVAL_CTX = 4096
QUANT = "IQ2_XS"
PROXY_TOKEN_SWEEP = [1024, 2048]

SRC_MODEL = EXP22 / "model_extracted"
IMATRIX = EXP22 / "imatrix-cal.gguf"
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
            "exp-024 reuses exp-022 QAT model+imatrix and exp-020 corpora+baseline:\n  "
            + names
        )


def _run_arm(proxy_tokens: int, n_params: float, csv_path: Path) -> runner.BenchRow:
    """Run a single (calibrate → apply → convert → quantize → bench) arm."""
    arm_dir = EXP24 / f"proxy{proxy_tokens}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    logs = arm_dir / "logs"
    logs.mkdir(exist_ok=True)

    awq_bundle = arm_dir / "awq.pt"
    model_awq = arm_dir / "model_awq"
    f16_awq = arm_dir / "model-f16-awq.gguf"
    qpath = arm_dir / f"{QUANT}-awq.gguf"

    with phase(f"[proxy={proxy_tokens}] AWQ calibrate (cv_strategy=gate)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, CAL_CORPUS, awq_bundle,
                 proxy_tokens=proxy_tokens,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase(f"[proxy={proxy_tokens}] AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase(f"[proxy={proxy_tokens}] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq,
                 log=logs / "convert.log"))

    with phase(f"[proxy={proxy_tokens}] quantize {QUANT}"):
        step(f"quantize {QUANT}", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=logs / f"quantize-{QUANT}.log"))

    label = (
        f"{QAT_REPO_ID}|{QUANT}|awq-cv-gate+imatrix|"
        f"proxy_tokens={proxy_tokens} // baseline=vanilla-FP16"
    )
    with phase(f"[proxy={proxy_tokens}] bench {QUANT}"):
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
        log(f"  proxy={proxy_tokens}: PPL={row.ppl:.4f} KLD={row.mean_kld:.5f} "
            f"top_p={row.same_top_p:.4f}%")
    return row


def main() -> int:
    _check_inputs()
    EXP24.mkdir(parents=True, exist_ok=True)
    csv_path = EXP24 / "results.csv"

    n_params = bpw_mod.n_params(VANILLA_F16)
    log(f"reference n_params (from vanilla FP16) = {n_params:,.0f}")

    rows: dict[int, runner.BenchRow] = {}
    for pt in PROXY_TOKEN_SWEEP:
        rows[pt] = _run_arm(pt, n_params, csv_path)

    log("")
    log("=== exp-024 sweep results (QAT-IQ2_XS, AWQ cv-gate, varying proxy_tokens) ===")
    log("  reference (from exp-022 / exp-023):")
    log("    imatrix only:           KLD 1.85881  top_p 47.09%  PPL 209.00")
    log("    AWQ proxy_tokens=256:   KLD 1.88059  top_p 48.78%  PPL 153.29")
    log("    AWQ proxy_tokens=512:   KLD 1.78160  top_p 49.72%  PPL 112.57")
    log("  exp-024 new arms:")
    for pt, r in rows.items():
        log(f"    AWQ proxy_tokens={pt}:  "
            f"KLD {r.mean_kld:.5f}  top_p {r.same_top_p:.4f}%  PPL {r.ppl:.4f}")

    klds = {256: 1.88059, 512: 1.78160, **{pt: r.mean_kld for pt, r in rows.items()}}
    sorted_pts = sorted(klds.keys())
    log("")
    log("  KLD vs proxy_tokens:")
    for pt in sorted_pts:
        log(f"    {pt:>5}: {klds[pt]:.5f}")

    # Crude verdict
    kld_512 = 1.78160
    kld_1024 = rows[1024].mean_kld
    kld_2048 = rows[2048].mean_kld
    log("")
    if kld_2048 < kld_1024 - 0.005 and kld_1024 < kld_512 - 0.005:
        log("  -> curve still climbing. Queue proxy_tokens=4096 next.")
    elif abs(kld_2048 - kld_1024) < 0.01 and kld_1024 <= kld_512 + 0.01:
        log(f"  -> plateau at proxy_tokens={1024 if kld_1024 <= kld_512 else 512}. "
            f"Adopt as new QAT default; expand to Q2_K_S to confirm.")
    elif kld_1024 > kld_512 + 0.01:
        log("  -> non-monotonic: 1024 worse than 512. Re-run 1024 to check variance "
            "before drawing a conclusion.")
    else:
        log("  -> unclear shape; inspect numbers individually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

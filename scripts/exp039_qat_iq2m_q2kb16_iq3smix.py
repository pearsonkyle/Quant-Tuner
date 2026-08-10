"""Experiment 039: QAT IQ2_M AWQ with q2k_b16 base + mix=IQ2_M (iq3_s overrides).

exp-038 showed vanilla IQ2_M with q2k_b16+iq3s mix wins materially over the
prior q2k_b16-no-mix baseline (medKLD 1.548 vs 1.746, PPL 652 vs 1276,
top_p 48.79% vs 46.75%). This applies the same recipe to the QAT source —
the only IQ2_M proxy/mix combination not yet tested on QAT.

Prior QAT IQ2_M failures (all PPL≈2e10, medKLD≈23, top_p=0%):
  exp-032 row 1: proxy=iq2_s, mix=None         — uniform iq2_s scoring (toxic)
  exp-032 row 2: proxy=iq2_s, mix=IQ2_M        — same toxic base, irrelevant mix
  exp-033       : proxy=q2k_b16, mix=None      — works on vanilla, broken on QAT
  exp-037 (bug) : actually QAT, proxy=iq2_s, mix=IQ2_M (iq3_s) — toxic base again

This experiment is the last untested combination: q2k_b16 base (the safe
choice) + the new iq3_s codebook mix for the bumped tensors. If even this
fails, the QAT+IQ2_M pairing is structurally broken and we'll keep it out
of the published table.
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

QAT_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QAT_SLUG = QAT_ID.replace("/", "__")
VANILLA_SLUG = "google__gemma-4-31B-it"

EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP22 = REPO / "out" / "exp-022" / QAT_SLUG
EXP39 = REPO / "out" / "exp-039"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"

SRC_HF = EXP22 / "model_extracted"
SRC_F16 = EXP22 / "model-f16.gguf"
SRC_IMATRIX = EXP22 / "imatrix-cal.gguf"

QUANT = "IQ2_M"
PROXY = "q2k_b16"
PROXY_MIX = QUANT
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


def _check_inputs() -> None:
    required = [SRC_HF / "config.json", SRC_F16, SRC_IMATRIX,
                CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-039 missing inputs:\n  " + "\n  ".join(missing))


def _csv_has_qpath(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        for line in fh:
            if needle in line:
                return True
    return False


def _bench(qpath: Path, label: str, n_params: int, csv_path: Path, log_dir: Path) -> None:
    if _csv_has_qpath(csv_path, qpath):
        log(f"  {qpath.name}: bench row already in CSV — skipping")
        return
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
    log(f"  {qpath.name}: PPL={row.ppl:.4f} mean_KLD={row.mean_kld:.5f} "
        f"med_KLD={row.median_kld:.5f} top_p={row.same_top_p:.4f}%")


def _free_intermediates(sub: Path) -> None:
    for child in (sub / "model_awq", sub / "model-f16-awq.gguf"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)


def main() -> int:
    _check_inputs()
    EXP39.mkdir(parents=True, exist_ok=True)
    csv_path = EXP39 / "results.csv"

    sub = EXP39 / "qat-iq2_m-q2kb16-iq3smix"
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / "gemma-4-31B-it-qat-IQ2_M-awq-q2kb16-iq3smix.gguf"
    n_params = bpw_mod.n_params(SRC_F16)

    tag = "[exp-039][qat-iq2m-q2kb16-iq3smix]"

    with phase(f"{tag} AWQ calibrate (proxy={PROXY}, mix={PROXY_MIX})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 SRC_HF, CAL_CORPUS, bundle,
                 proxy=PROXY,
                 proxy_mix=PROXY_MIX,
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

    with phase(f"{tag} AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_HF, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"{tag} quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=SRC_IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        label = (f"{QAT_ID}|{QUANT}|awq-cv-gate+imatrix|"
                 f"proxy={PROXY},mix={PROXY_MIX},tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_intermediates(sub)

    log("")
    log("=== exp-039 complete ===")
    log("References:")
    log("  vanilla IQ2_M imatrix                    : medKLD=1.496 top_p=47.61% PPL=2060.73")
    log("  vanilla IQ2_M AWQ q2k_b16+iq3s mix (38)  : medKLD=1.548 top_p=48.79% PPL=652.81")
    log("  QAT IQ2_M (prior tests, all BROKEN)      : medKLD≈23 top_p=0% PPL≈2e10")
    log(f"  results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

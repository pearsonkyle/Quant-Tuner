"""Experiment 038: vanilla IQ2_M AWQ with q2k_b16 base + mix=IQ2_M (iq3_s overrides).

exp-037 had a script bug — it loaded the QAT model instead of vanilla, so its
collapse reproduces the known QAT-IQ2_M failure, not a vanilla data point.
This re-runs cleanly with explicit vanilla paths, and uses the recipe-pinned
q2k_b16 base proxy (known-working for vanilla IQ2_M, exp-034 medKLD 1.746)
plus the new proxy_mix=IQ2_M overrides that route v_proj/o_proj/down-first-
eighth through the iq3_s codebook proxy.

Question: does adding faithful iq3_s scoring for the bumped tensors improve
over the no-mix q2k_b16 baseline (medKLD 1.746, top_p 46.75%, PPL 1276)?
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

VANILLA_ID = "google/gemma-4-31B-it"
VANILLA_SLUG = VANILLA_ID.replace("/", "__")

EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP34 = REPO / "out" / "exp-034"
EXP38 = REPO / "out" / "exp-038"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"

SRC_HF = EXP34 / "vanilla" / "model_extracted"
SRC_F16 = EXP34 / "vanilla" / "model-f16.gguf"
SRC_IMATRIX = EXP20 / "imatrix-cal.gguf"

QUANT = "IQ2_M"
PROXY = "q2k_b16"  # recipe-pinned; iq2_s base regresses (steep α landscape)
PROXY_MIX = QUANT  # routes v_proj/o_proj/down-first-eighth to iq3_s
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


def _check_inputs() -> None:
    required = [SRC_HF / "config.json", SRC_F16, SRC_IMATRIX,
                CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-038 missing inputs:\n  " + "\n  ".join(missing))


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
    EXP38.mkdir(parents=True, exist_ok=True)
    csv_path = EXP38 / "results.csv"

    sub = EXP38 / "vanilla-iq2_m-q2kb16-iq3smix"
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / "gemma-4-31B-it-IQ2_M-awq-q2kb16-iq3smix.gguf"
    n_params = bpw_mod.n_params(SRC_F16)

    tag = "[exp-038][vanilla-iq2m-q2kb16-iq3smix]"

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
        label = (f"{VANILLA_ID}|{QUANT}|awq-cv-gate+imatrix|"
                 f"proxy={PROXY},mix={PROXY_MIX},tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_intermediates(sub)

    log("")
    log("=== exp-038 complete ===")
    log("References:")
    log("  vanilla IQ2_M imatrix       : medKLD=1.496 top_p=47.61% PPL=2060.73")
    log("  vanilla IQ2_M AWQ (q2k_b16, no mix, exp-034): medKLD=1.746 top_p=46.75% PPL=1276.27")
    log(f"  results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

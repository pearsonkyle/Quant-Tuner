"""Experiment 035: QAT Q2_K_S AWQ with the new q2k_super proxy.

exp-034 showed QAT Q2_K_S AWQ regressing vs imatrix-only on PPL (158 vs
111) and top_p (44.6 vs 47.6). Hypothesis: the q2k_b16 proxy is too rough
(per-row normalization scope) to faithfully reproduce Q2_K's super-block
quantization error, so AWQ's α search optimizes against the wrong loss
landscape and picks α values that don't transfer to the real quantizer.

q2k_super is a new proxy mirroring llama.cpp's block_q2_K exactly:
  * 256-weight super-block → 16 sub-blocks × 16 weights
  * Per-sub-block (scale, ymin) stored as 4-bit uints
  * Per-super-block (d, dmin) fp16 normalization (NOT per-row)
  * 2-bit unsigned weight codes [0, 3]

Settings match exp-034 (winner config from exp-031):
  * proxy_tokens = 1024
  * ctx          = 4096
  * proxy_mix    = None
  * cv_strategy  = gate, cv_weight = 1.0
  * sanity_max_rel = 1.20

Single row to validate. If it beats exp-034's QAT Q2_K_S AWQ on top_p
and PPL, we'll re-run the full Q2_K_S AWQ pair (vanilla + QAT).
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
EXP36 = REPO / "out" / "exp-036"
EXP34 = REPO / "out" / "exp-034"
VANILLA_HF = EXP34 / "vanilla" / "model_extracted"
VANILLA_F16 = EXP34 / "vanilla" / "model-f16.gguf"
VANILLA_IMATRIX = EXP20 / "imatrix-cal.gguf"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"

QAT_HF = EXP22 / "model_extracted"
VANILLA_IMATRIX = EXP22 / "imatrix-cal.gguf"
VANILLA_F16 = EXP22 / "model-f16.gguf"

QUANT = "Q2_K_S"
PROXY = "q2k_super"  # exp-036: new super-block proxy
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


def _check_inputs() -> None:
    required = [
        QAT_HF / "config.json", VANILLA_IMATRIX, VANILLA_F16,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-036 missing inputs:\n  " + "\n  ".join(missing))


def _csv_has_qpath(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        for line in fh:
            if needle in line:
                return True
    return False


def _bench(qpath: Path, label: str, n_params: int, csv_path: Path,
           log_dir: Path) -> None:
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
    EXP36.mkdir(parents=True, exist_ok=True)
    csv_path = EXP36 / "results.csv"

    sub = EXP36 / "vanilla-q2k_s-q2ksuper"
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / "gemma-4-31B-it-Q2_K_S-awq-q2ksuper.gguf"
    n_params = bpw_mod.n_params(VANILLA_F16)

    tag = "[exp-036][vanilla-q2ks-q2ksuper]"

    with phase(f"{tag} AWQ calibrate (proxy={PROXY})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 VANILLA_HF, CAL_CORPUS, bundle,
                 proxy=PROXY,
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

    with phase(f"{tag} AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 VANILLA_HF, bundle, model_awq,
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
                 f16_awq, qpath, QUANT, imatrix=VANILLA_IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        label = (f"google/gemma-4-31B-it|{QUANT}|awq-cv-gate+imatrix|"
                 f"proxy={PROXY},mix=None,tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_intermediates(sub)

    log("")
    log("=== exp-036 complete ===")
    log("Reference (exp-034):")
    log("  vanilla Q2_K_S imatrix : medKLD=2.138 top_p=42.60% PPL=1436.40")
    log("  vanilla Q2_K_S AWQ (q2k_b16): medKLD=1.657 top_p=49.09% PPL=177.03")
    log("Reference (exp-035, QAT, q2k_super): medKLD=1.081 top_p=49.40% PPL=88.18")
    log(f"  results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

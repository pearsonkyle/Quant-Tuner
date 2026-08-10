"""Experiment 032: IQ2_M mix-vs-nomix at proxy=1024 ctx=4096 (QAT source).

exp-031 showed proxy=1024 mix=None beats every prior IQ2_XS run (medKLD
1.151, top_p 49.00%, PPL 108.66). The mix=ON path regressed because the
GQA detection in `_gqa_or_moe_ge4` returned False for Gemma-4, so only the
ffn_down tier-up fired and the v_proj override didn't.

This validates the same setup on IQ2_M before locking in defaults for the
full re-quant table. Two rows, sequential:

  1. proxy=iq2_s, mix=None        — uniform iq2_s scoring
  2. proxy=iq2_s, mix=IQ2_M       — per-member proxy (ffn_down tier-up to
                                    iq3_s, attn_output to iq3_s, etc.)

Note: the shipped iq2_m_awq recipe pins proxy=q2k_b16 because uniform iq2_s
regressed IQ2_M. We're NOT testing q2k_b16 here — the question is just
whether the same proxy=1024 ctx=4096 trick that won for IQ2_XS also wins
for IQ2_M, with the auto-selected codebook proxy.

Everything else matches exp-031:
  * source       : google/gemma-4-31B-it-qat-q4_0-unquantized
  * α grid       : 0.0..1.0 step 0.25, per-tensor refinement radius 0.15
  * cv_strategy  : gate, cv_weight=1.0
  * sanity_max_rel: 1.20
  * imatrix      : reused from exp-022 (collected on QAT F16 at ctx=4096
                   with --parse-special)
  * corpora      : exp-020 cal/val/eval
  * eval         : ctx=4096, baseline.kld from exp-020
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
EXP37 = REPO / "out" / "exp-037"
EXP34 = REPO / "out" / "exp-034"
VANILLA_HF_DIR = EXP34 / "vanilla" / "model_extracted"
VANILLA_F16_GGUF = EXP34 / "vanilla" / "model-f16.gguf"
VANILLA_IMATRIX = EXP20 / "imatrix-cal.gguf"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"

VANILLA_HF_DIR = EXP22 / "model_extracted"
VANILLA_IMATRIX = EXP22 / "imatrix-cal.gguf"
VANILLA_F16 = VANILLA_F16_GGUF

QUANT = "IQ2_M"
PROXY = awq.proxy_for_quant_type(QUANT)  # iq2_s
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096

# (mix_setting,) per row.
RUNS: tuple[str | None, ...] = (QUANT,)  # row 1 (mix=None) collapsed (PPL 2e10); skip directly to mix=IQ2_M


def _check_inputs() -> None:
    required = [
        VANILLA_HF_DIR / "config.json", VANILLA_IMATRIX, VANILLA_F16,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-037 missing inputs:\n  " + "\n  ".join(missing))


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
           log_dir: Path) -> runner.BenchRow | None:
    if _csv_has_qpath(csv_path, qpath):
        log(f"  {qpath.name}: bench row already in CSV — skipping")
        return None
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
    return row


def _free_intermediates(sub: Path) -> None:
    for child in (sub / "model_awq", sub / "model-f16-awq.gguf"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)


def _run_one(proxy_mix: str | None, *, csv_path: Path) -> None:
    suffix = "-mix" if proxy_mix else "-nomix"
    sub = EXP37 / f"vanilla-iq2_m-proxy-{PROXY_TOKENS}{suffix}"
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / f"gemma-4-31B-it-IQ2_M-awq-proxy-{PROXY_TOKENS}{suffix}.gguf"
    n_params = bpw_mod.n_params(VANILLA_F16)

    tag = f"[exp-037][proxy={PROXY_TOKENS},ctx={CTX},mix={proxy_mix}]"

    with phase(f"{tag} AWQ calibrate (proxy={PROXY})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 VANILLA_HF_DIR, CAL_CORPUS, bundle,
                 proxy=PROXY,
                 proxy_mix=proxy_mix,
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
                 VANILLA_HF_DIR, bundle, model_awq,
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
                 f"proxy={PROXY},mix={proxy_mix},tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_intermediates(sub)


def main() -> int:
    _check_inputs()
    EXP37.mkdir(parents=True, exist_ok=True)
    csv_path = EXP37 / "results.csv"

    for proxy_mix in RUNS:
        _run_one(proxy_mix, csv_path=csv_path)

    log("")
    log("=== exp-037 complete ===")
    log("References:")
    log("  vanilla IQ2_M imatrix       : medKLD=1.496 top_p=47.61% PPL=2060.73")
    log("  vanilla IQ2_M AWQ (q2k_b16) : medKLD=1.746 top_p=46.75% PPL=1276.27")
    log("  exp-031 IQ2_XS proxy=1024, mix=None:")
    log("  medKLD=1.151 mean=1.853 top_p=49.00% PPL=108.66")
    log("")
    log(f"  results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

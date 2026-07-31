"""exp-040d: does GROUP-α-only AWQ (per_tensor_alpha=False) cut the fold drift
enough to beat imatrix-only on Qwopus3.5-27B-v3 IQ2_XS?

Story so far:
  imatrix-only             top_p 81.70%  med_KLD 0.076
  AWQ per-tensor α (fixed) top_p 78.69%  med_KLD 0.122  (exp-040b, plus_one=True)
  AWQ + folded imatrix     top_p 78.78%  med_KLD 0.123  (exp-040c — imatrix wasn't the gap)

Hypothesis for the residual ~3-pt gap: per-tensor α bakes a per-channel residual
into the fused attn_qkv (q/k/v share one input-norm but get different scales),
widening dynamic range right where 2-bit can't follow. Group-α-only folds cleanly
into the shared norm with no residual → less injected drift (apply-rel should drop
below the per-tensor 0.40). If the model truly lacks outlier channels, milder AWQ
should land closer to — at best at — imatrix-only.

Single changed variable vs exp-040b: per_tensor_alpha True→False (so cv-gate off,
no holdout). Same pre-fold imatrix (exp-040c showed folded imatrix ≈ no change),
same proxy (iq2_xs), same plus_one=True fix. Needs a fresh calibrate (the saved
bundle holds per-tensor scales).

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040d_group_alpha.py
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

EXP40 = REPO / "out" / "exp-040"
HF_DIR = EXP40 / "model_extracted"
F16_GGUF = EXP40 / "model-f16.gguf"
IMATRIX = EXP40 / "imatrix-cal.gguf"          # pre-fold; exp-040c showed folded ≈ same
BASELINE_KLD = EXP40 / "baseline.kld"
EVAL_CORPUS = EXP40 / "corpora" / "corpus.eval.txt"
CAL_CORPUS = EXP40 / "corpora" / "corpus.cal.txt"

OUT = EXP40 / "iq2xs_awq_groupalpha"
QUANT = "IQ2_XS"
PROXY = "iq2_xs"
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, IMATRIX, BASELINE_KLD,
              EVAL_CORPUS, CAL_CORPUS):
        if not p.exists():
            raise FileNotFoundError(f"exp-040d missing input: {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = OUT / "awq.pt"
    model_awq = OUT / "model_awq"
    f16_awq = OUT / "model-f16-awq.gguf"
    qpath = OUT / f"{QUANT}-awq-groupalpha.gguf"
    csv_path = OUT / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase("[exp-040d] AWQ calibrate (GROUP-α only, per_tensor_alpha=False)"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 HF_DIR, CAL_CORPUS, bundle,
                 proxy=PROXY,
                 proxy_mix=None,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=False,   # the variable under test
                 cv_strategy="off",        # gate requires per-tensor α; off here
             ))

    with phase("[exp-040d] AWQ apply (plus_one=True, group scales fold cleanly)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=True,
                 sanity_max_rel=0.60,
             ))

    with phase("[exp-040d] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"[exp-040d] quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase("[exp-040d] bench"):
        label = (f"Jackrong/Qwopus3.5-27B-v3|{QUANT}|"
                 f"awq-group-alpha(per_tensor=False)+imatrix // baseline=FP16")
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

    log("")
    log("=== exp-040d verdict (IQ2_XS) ===")
    log(f"  imatrix-only              : top_p 81.70%  med_KLD 0.076")
    log(f"  AWQ per-tensor α (fixed)  : top_p 78.69%  med_KLD 0.122  (exp-040b)")
    log(f"  AWQ group-α only          : top_p {row.same_top_p:6.2f}%  med_KLD {row.median_kld:.3f}  (this run)")
    log("  If group-α ≥ imatrix-only, milder AWQ is the answer; if still below,")
    log("  AWQ has no headroom on this model and imatrix-only should ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

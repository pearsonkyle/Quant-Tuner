"""exp-040b: isolate the AWQ collapse on Qwopus3.5-27B-v3 to the RMSNorm fold form.

exp-040's AWQ rows all collapsed (PPL 2e5-1.5e6, top_p ~0%) while imatrix-only
was excellent. The runner copied the gemma `apply` kwargs verbatim:
    rmsnorm_plus_one=False, sanity_max_rel=1.20
But this model is `Qwen3_5ForConditionalGeneration`, whose RMSNorm applies the
`(1 + γ)` unit-offset gain. `awq.fold_rmsnorm_gain(plus_one=True)` is the correct
form for it (and is the `apply` default); `plus_one=False` leaves the unit term
unscaled, so the folded F16 forward already drifts hard (exp-040 logged apply
rel ≈ 1.0 vs gemma's healthy ~0.34) and 2-bit quant amplifies it to garbage.

Controlled test: re-use the *exact* saved IQ2_XS scale bundle from exp-040
(out/exp-040/iq2xs_awq/awq.pt — same group + per-tensor α scales) and re-run
ONLY `apply` with rmsnorm_plus_one=True, then convert + quantize IQ2_XS + bench.
The single changed variable is the fold form. Skips the ~1.5 h calibrate.

PASS criteria: apply sanity `rel` drops well below 1.0, and IQ2_XS top_p jumps
from 0.15% back toward the imatrix-only baseline (81.7%).

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040b_awq_rmsnorm_fix.py
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
IMATRIX = EXP40 / "imatrix-cal.gguf"
BASELINE_KLD = EXP40 / "baseline.kld"
EVAL_CORPUS = EXP40 / "corpora" / "corpus.eval.txt"

BUNDLE = EXP40 / "iq2xs_awq" / "awq.pt"   # reuse exp-040's saved scales

OUT = EXP40 / "iq2xs_awq_fix"
QUANT = "IQ2_XS"
EVAL_CTX = 4096


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, IMATRIX, BASELINE_KLD,
              EVAL_CORPUS, BUNDLE):
        if not p.exists():
            raise FileNotFoundError(f"exp-040b missing reusable input: {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_awq = OUT / "model_awq"
    f16_awq = OUT / "model-f16-awq.gguf"
    qpath = OUT / f"{QUANT}-awq-rmsnorm-fix.gguf"
    csv_path = OUT / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase("[exp-040b] AWQ apply (rmsnorm_plus_one=TRUE, tight gate)"):
        # sanity_max_rel left permissive (1.20) so a still-elevated rel doesn't
        # abort the test — the printed rel + final bench are the real signals.
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, BUNDLE, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=True,
                 sanity_max_rel=1.20,
             ))

    with phase("[exp-040b] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"[exp-040b] quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase("[exp-040b] bench"):
        label = (f"Jackrong/Qwopus3.5-27B-v3|{QUANT}|"
                 f"awq-rmsnorm-fix(plus_one=True)+imatrix // baseline=FP16")
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

    log("")
    log("=== exp-040b verdict ===")
    log(f"  IQ2_XS imatrix-only baseline : top_p 81.70%  med_KLD 0.076")
    log(f"  IQ2_XS AWQ plus_one=False    : top_p  0.15%  med_KLD 12.03  (exp-040, broken)")
    log(f"  IQ2_XS AWQ plus_one=True     : top_p {row.same_top_p:6.2f}%  med_KLD {row.median_kld:.3f}  (this run)")
    log("  If top_p recovered, the fix is rmsnorm_plus_one=True for Qwen3.5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

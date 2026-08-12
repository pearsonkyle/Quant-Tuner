"""exp-040g: best-shot AWQ on IQ2_M — the one quant type where gemma's AWQ won.

IQ2_M imatrix-only is the strongest artifact in the whole sweep
(top_p 86.03% / med_KLD 0.044). gemma's AWQ specifically beat imatrix at IQ2_M,
so this is the last place AWQ might add value on Qwopus3.5-27B-v3.

Config = the IQ2_M recipe proxy + the two lessons learned from the IQ2_XS sweep:
  * proxy = q2k_b16, proxy_mix = "IQ2_M"  (recipe: q2k_b16 base for the iq2_s
    tensors, routes the Q4_K-bumped v_proj + IQ3_S-bumped o_proj/down-first-eighth
    through their real targets — avoids the fictitious 2-bit error that drags α)
  * per_tensor_alpha = False  (group-α: per-tensor α injected fused-QKV fold drift
    and *hurt* IQ2_XS; group-α folds clean at rel~0.01)
  * rmsnorm_plus_one = True   (Qwen3.5 (1+γ) RMSNorm — the original collapse fix)
  * proxy_tokens = 1024, ctx = 4096, pre-fold imatrix for the final quantize

PASS: top_p > 86.03% => AWQ finally earns a row, ship it for IQ2_M.
TIE/BELOW: publish imatrix-only — AWQ has no headroom anywhere on this model.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040g_iq2m_awq.py
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
CAL_CORPUS = EXP40 / "corpora" / "corpus.cal.txt"

OUT = EXP40 / "iq2m_awq_groupalpha"
QUANT = "IQ2_M"
PROXY = "q2k_b16"          # recipe-pinned base (pure iq2_s scoring regresses IQ2_M)
PROXY_MIX = "IQ2_M"        # routes v_proj→Q4_K, o_proj/down-first-eighth→IQ3_S
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, IMATRIX, BASELINE_KLD,
              EVAL_CORPUS, CAL_CORPUS):
        if not p.exists():
            raise FileNotFoundError(f"exp-040g missing input: {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = OUT / "awq.pt"
    model_awq = OUT / "model_awq"
    f16_awq = OUT / "model-f16-awq.gguf"
    qpath = OUT / f"{QUANT}-awq-groupalpha.gguf"
    csv_path = OUT / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase(f"[exp-040g] AWQ calibrate (group-α, proxy={PROXY}, mix={PROXY_MIX})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 HF_DIR, CAL_CORPUS, bundle,
                 proxy=PROXY,
                 proxy_mix=PROXY_MIX,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=False,
                 cv_strategy="off",
             ))

    with phase("[exp-040g] AWQ apply (plus_one=True)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, bundle, model_awq,
                 device="auto", dtype="bfloat16",
                 rmsnorm_plus_one=True, sanity_max_rel=0.60,
             ))

    with phase("[exp-040g] convert -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"[exp-040g] quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase("[exp-040g] bench"):
        label = (f"Jackrong/Qwopus3.5-27B-v3|{QUANT}|"
                 f"awq-group-alpha+mix(IQ2_M)+imatrix // baseline=FP16")
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
        )
        runner.append_row(csv_path, row)

    log("")
    log("=== exp-040g verdict (IQ2_M) ===")
    log(f"  imatrix-only          : top_p 86.03%  med_KLD 0.044")
    log(f"  AWQ group-α + mix     : top_p {row.same_top_p:6.2f}%  med_KLD {row.median_kld:.4f}  (this run)")
    log("  > 86.03% => AWQ ships for IQ2_M; else publish imatrix-only (AWQ has no")
    log("  headroom anywhere on this model after 6 configs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

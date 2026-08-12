"""exp-040e: can proxy_mix=IQ2_XS push group-α AWQ ABOVE imatrix-only?

Story so far (Qwopus3.5-27B-v3 IQ2_XS):
  imatrix-only          top_p 81.70%  med_KLD 0.076
  AWQ group-α, mix=None top_p 81.63%  med_KLD 0.080  (exp-040d — TIED)

Why only a tie: this model is GQA 24/4 (ratio 6 ≥ 4), so llama-quantize bumps
attn_v → Q4_K and first-eighth ffn_down up a tier even under IQ2_XS
(proxy_for_member). exp-040d scored those with the iq2_xs codebook proxy — a
FICTITIOUS 2-bit error on tensors that are really Q4_K/Q3_K, which drags the
shared attention-group α down (the exact artifact CLAUDE.md cites for iq2_m_awq).

Fix under test: proxy_mix="IQ2_XS" routes attn_v → int4_g128 (Q4_K) and
first-eighth ffn_down → q2k_b16 during the α search, so the group α is chosen
against the REAL per-member targets. Single changed variable vs exp-040d:
proxy_mix None→IQ2_XS. Still group-α (per_tensor_alpha=False), plus_one=True,
same pre-fold imatrix for the final quantize.

PASS: top_p > 81.70% (AWQ finally out-benches imatrix-only). TIE/BELOW: the model
has no exploitable structure and imatrix-only ships.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040e_proxy_mix.py
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

OUT = EXP40 / "iq2xs_awq_proxymix"
QUANT = "IQ2_XS"
PROXY = "iq2_xs"
PROXY_MIX = "IQ2_XS"        # the variable under test (was None in exp-040d)
PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, IMATRIX, BASELINE_KLD,
              EVAL_CORPUS, CAL_CORPUS):
        if not p.exists():
            raise FileNotFoundError(f"exp-040e missing input: {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = OUT / "awq.pt"
    model_awq = OUT / "model_awq"
    f16_awq = OUT / "model-f16-awq.gguf"
    qpath = OUT / f"{QUANT}-awq-proxymix.gguf"
    csv_path = OUT / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase(f"[exp-040e] AWQ calibrate (group-α, proxy_mix={PROXY_MIX})"):
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

    with phase("[exp-040e] AWQ apply (plus_one=True)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=True,
                 sanity_max_rel=0.60,
             ))

    with phase("[exp-040e] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"[exp-040e] quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase("[exp-040e] bench"):
        label = (f"Jackrong/Qwopus3.5-27B-v3|{QUANT}|"
                 f"awq-group-alpha+proxymix(IQ2_XS)+imatrix // baseline=FP16")
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
    log("=== exp-040e verdict (IQ2_XS) ===")
    log(f"  imatrix-only              : top_p 81.70%  med_KLD 0.076")
    log(f"  AWQ group-α, mix=None     : top_p 81.63%  med_KLD 0.080  (exp-040d)")
    log(f"  AWQ group-α + proxy_mix   : top_p {row.same_top_p:6.2f}%  med_KLD {row.median_kld:.3f}  (this run)")
    log("  > 81.70% => AWQ finally beats imatrix-only; else imatrix-only ships.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

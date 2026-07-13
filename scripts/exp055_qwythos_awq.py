"""exp-055: AWQ IQ2_M for Qwythos-9B-v2 (does activation-aware scaling fix the
garbled 2-bit agent output?).

exp-054's imatrix IQ2_M doesn't loop (good) but only patches 40% on SWE-rebench —
it emits malformed agent output ("ASSISTANT (message):" echoes) instead of valid
tool calls. AWQ lifted Ornith's tool-argument accuracy 5%->33%, so try it here.

Reuses exp-054's extracted HF + F16 + corpora + eval baselines. Qwen3.5 norm =>
rmsnorm_plus_one=True (same as Ornith, per project_awq_qwen35_hybrid). Same arch
as Ornith (8 full-attn layers + MLPs scaled; 24 linear-attn layers pass through).

    PYTHONPATH=src .venv/bin/python scripts/exp055_qwythos_awq.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.calibrate import imatrix as imatrix_cal
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import convert, gguf

MODEL_ID = "empero-ai/Qwythos-9B-v2"

EXP54 = REPO / "out" / "exp-054"
HF_DIR = EXP54 / "model_extracted"
F16_GGUF = EXP54 / "model-f16.gguf"
CAL_CORPUS = EXP54 / "corpora" / "corpus.cal.txt"
GEN_EVAL = EXP54 / "corpora" / "corpus.eval.general.txt"
TOOLS_EVAL = EXP54 / "corpora" / "corpus.eval.tools.txt"
GEN_BASE = EXP54 / "baseline.general.kld"
TOOLS_BASE = EXP54 / "baseline.tools.kld"

EXP = REPO / "out" / "exp-055"
LOGS = EXP / "logs"

QUANT = "IQ2_M"
PROXY = "q2k_b16"        # recipe pin for IQ2_M
PROXY_MIX = "IQ2_M"
CTX = 4096
EVAL_CTX = 4096
PLUS_ONE = True          # Qwen3.5 (1+gamma) norm
SANITY_MAX_REL = 0.6


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, CAL_CORPUS, GEN_BASE, TOOLS_BASE):
        if not p.exists():
            raise FileNotFoundError(f"exp-055 missing input (run exp-054 first): {p}")
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16_GGUF)

    bundle = EXP / "awq.pt"
    model_awq = EXP / "model_awq"
    f16_awq = EXP / "model-f16-awq.gguf"
    base_im = EXP / "imatrix-awq-base.gguf"
    hybrid_im = EXP / "imatrix-awq-hybrid_custom.gguf"
    qpath = EXP / f"Qwythos-9B-v2-{QUANT}-awq.gguf"
    csv_path = EXP / "results.csv"

    with phase(f"[exp-055] AWQ calibrate (proxy={PROXY}, mix={PROXY_MIX})"):
        step("awq-calibrate", bundle,
             lambda: awq.calibrate(HF_DIR, CAL_CORPUS, bundle,
                                   proxy=PROXY, proxy_mix=PROXY_MIX, ctx=CTX,
                                   device="auto", dtype="bfloat16",
                                   per_tensor_alpha=False, cv_strategy="off"))

    with phase(f"[exp-055] AWQ apply (rmsnorm_plus_one={PLUS_ONE}, sanity<={SANITY_MAX_REL})"):
        step("awq-apply", model_awq / "config.json",
             lambda: awq.apply(HF_DIR, bundle, model_awq, device="auto", dtype="bfloat16",
                               rmsnorm_plus_one=PLUS_ONE, sanity_max_rel=SANITY_MAX_REL))

    with phase("[exp-055] convert -> F16 GGUF"):
        step("awq-convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq, log=LOGS / "convert.log"))

    with phase("[exp-055] imatrix on FOLDED F16"):
        step("awq-imatrix-base", base_im,
             lambda: llama_cpp.imatrix(f16_awq, CAL_CORPUS, base_im,
                                       ctx=CTX, log=LOGS / "imatrix.log"))
    with phase("[exp-055] re-weight folded -> hybrid_custom"):
        step("awq-imatrix-hybrid", hybrid_im,
             lambda: imatrix_cal.calibrate(variant="hybrid_custom", f16_gguf=f16_awq,
                                           base_imatrix=base_im, out_path=hybrid_im))

    with phase(f"[exp-055] quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(f16_awq, qpath, QUANT, imatrix=hybrid_im,
                                   log=LOGS / "quantize.log"))

    with phase("[exp-055] KLD bench (general + tools)"):
        for corpus, base, tag in ((GEN_EVAL, GEN_BASE, "general"), (TOOLS_EVAL, TOOLS_BASE, "tools")):
            r = runner.bench_one(qpath, f"{MODEL_ID}|{QUANT}|awq // eval={tag} // baseline=FP16",
                                 reference_n_params=n_params, eval_dataset=corpus,
                                 eval_baseline=base, eval_ctx=EVAL_CTX, log_dir=LOGS, suite="kld")
            runner.append_row(csv_path, r)
            log(f"    {tag}: bpw={r.bpw:.3f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-055 complete ===")
    log(f"  AWQ quant: {qpath}")
    log("  Next: SWE-rebench (1 rep, 10 instances) vs the imatrix IQ2_M (40% patch).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

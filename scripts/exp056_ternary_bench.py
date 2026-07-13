"""exp-056: static bench of prism-ml Ternary-Bonsai-8B Q2_0 (native 1.58-bit).

Ternary-Bonsai-8B is a Qwen3-8B natively TRAINED ternary model ({-1,0,+1} + a
per-128 fp16 scale, ggml type Q2_0 = 42) — NOT a post-hoc quant. Its Q2_0 format
is not in mainline llama.cpp, so this bench must run against the PrismML fork:

    LLAMA_CPP_DIR=vendor/llama.cpp-prism   (built with -DGGML_METAL=ON)

We bench Q2_0 vs the shipped Ternary F16 on the SAME external general-English eval
corpus used for Ornith/Qwythos (out/exp-054), so the general KLD/PPL/top_p land on
a comparable axis (each KLD is vs its own model's F16). PQ2_0 and Q2_0_g64 don't
load on the prism branch (types 142 / g64 layout) — Q2_0 is the recommended one.

    LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src \
        .venv/bin/python scripts/exp056_ternary_bench.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.experiments import log, phase, step

GGUF = REPO / "out" / "exp-056" / "gguf"
F16 = GGUF / "Ternary-Bonsai-8B-F16.gguf"
Q2_0 = GGUF / "Ternary-Bonsai-8B-Q2_0.gguf"

# reuse the external general-English eval corpus (model-agnostic text)
GEN_EVAL = REPO / "out" / "exp-054" / "corpora" / "corpus.eval.general.txt"

EXP = REPO / "out" / "exp-056"
LOGS = EXP / "logs"
GEN_BASELINE = EXP / "baseline.general.kld"
CSV = EXP / "results.general.csv"
EVAL_CTX = 4096


def main() -> int:
    for p in (F16, Q2_0, GEN_EVAL):
        if not p.exists():
            raise FileNotFoundError(p)
    LOGS.mkdir(parents=True, exist_ok=True)

    n_params = bpw_mod.n_params(F16)
    log(f"[exp-056] Ternary F16 n_params = {n_params:,.0f}")

    with phase("[exp-056] FP16 baseline (general eval)"):
        step("baseline-general", GEN_BASELINE,
             lambda: kld.build_baseline(F16, GEN_EVAL, GEN_BASELINE,
                                        ctx=EVAL_CTX, log=LOGS / "baseline-general.log"))

    with phase("[exp-056] bench Q2_0 (native ternary)"):
        label = "prism-ml/Ternary-Bonsai-8B|Q2_0|native-ternary // eval=general // baseline=FP16"
        r = runner.bench_one(Q2_0, label, reference_n_params=n_params,
                             eval_dataset=GEN_EVAL, eval_baseline=GEN_BASELINE,
                             eval_ctx=EVAL_CTX, log_dir=LOGS, suite="kld")
        runner.append_row(CSV, r)
        log(f"    Q2_0: bpw={r.bpw:.3f} PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} "
            f"top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-056 complete ===")
    log(f"  results: {CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

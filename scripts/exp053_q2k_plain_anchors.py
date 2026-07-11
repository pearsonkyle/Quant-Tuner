"""exp-053: naive Q2_K-plain 2-bit anchors (no calibration) for the article.

To tell the "how to make 2-bit usable for agents" story we need the naive
baseline: a 2-bit quant with NO calibration at all. IQ2_* can't run without an
imatrix, so the no-calibration anchor is plain Q2_K (llama-quantize default,
no imatrix). Builds it for gemma-4-31B (vanilla) and Ornith-1.0-9B and
static-benches on the SAME evals as the imatrix/AWQ IQ2_M builds so the 3-way
table is apples-to-apples.

Agentic tool-call replay on these is a separate step (run_toolcall_reps).

    PYTHONPATH=src .venv/bin/python scripts/exp053_q2k_plain_anchors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import gguf

EVAL_CTX = 4096

GEMMA_F16 = REPO / "out" / "exp-044" / "vanilla" / "model-f16.gguf"
GEMMA_Q2K = REPO / "out" / "exp-052" / "gemma-4-31B-it-Q2_K-plain.gguf"
GEMMA_EVAL = REPO / "out" / "exp-044" / "corpora" / "corpus.eval.txt"
GEMMA_BASELINE = REPO / "out" / "exp-044" / "baseline.kld"

ORNITH_F16 = REPO / "out" / "exp-050" / "model-f16.gguf"
ORNITH_Q2K = REPO / "out" / "exp-051" / "Ornith-1.0-9B-Q2_K-plain.gguf"
SHIP = REPO / "uploads" / "pearsonkyle" / "Ornith-1.0-9B-imatrix-GGUF" / "calibration_data"
ORNITH_GEN_EVAL = SHIP / "corpus.eval.general.txt"
ORNITH_GEN_BASE = REPO / "out" / "exp-051" / "baseline.general.kld"
ORNITH_TOOLS_EVAL = SHIP / "corpus.eval.tools.txt"
ORNITH_TOOLS_BASE = REPO / "out" / "exp-051" / "baseline.tools.kld"

CSV = REPO / "out" / "exp-053" / "results.csv"


def _bench(label, qpath, f16, eval_ds, baseline, log_dir):
    r = runner.bench_one(
        qpath, label, reference_n_params=bpw_mod.n_params(f16),
        eval_dataset=eval_ds, eval_baseline=baseline,
        eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld")
    runner.append_row(CSV, r)
    log(f"  {label}: bpw={r.bpw:.3f} PPL={r.ppl:.4f} "
        f"medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")


def main() -> int:
    out = REPO / "out" / "exp-053"
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with phase("[exp-053] gemma Q2_K-plain (no imatrix)"):
        step("quantize", GEMMA_Q2K,
             lambda: gguf.quantize(GEMMA_F16, GEMMA_Q2K, "Q2_K", imatrix=None,
                                   log=log_dir / "gemma-q2k.log"))
        _bench("google/gemma-4-31B-it|Q2_K|plain // eval=external",
               GEMMA_Q2K, GEMMA_F16, GEMMA_EVAL, GEMMA_BASELINE, log_dir)

    with phase("[exp-053] Ornith Q2_K-plain (no imatrix)"):
        step("quantize", ORNITH_Q2K,
             lambda: gguf.quantize(ORNITH_F16, ORNITH_Q2K, "Q2_K", imatrix=None,
                                   log=log_dir / "ornith-q2k.log"))
        _bench("deepreinforce-ai/Ornith-1.0-9B|Q2_K|plain // eval=general",
               ORNITH_Q2K, ORNITH_F16, ORNITH_GEN_EVAL, ORNITH_GEN_BASE, log_dir)
        _bench("deepreinforce-ai/Ornith-1.0-9B|Q2_K|plain // eval=tools",
               ORNITH_Q2K, ORNITH_F16, ORNITH_TOOLS_EVAL, ORNITH_TOOLS_BASE, log_dir)

    log("")
    log("=== exp-053 complete ===")
    log(f"  gemma Q2_K: {GEMMA_Q2K}")
    log(f"  ornith Q2_K: {ORNITH_Q2K}")
    log(f"  results: {CSV}")
    log("  Next: agentic tool-call replay on both (run_toolcall_reps).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

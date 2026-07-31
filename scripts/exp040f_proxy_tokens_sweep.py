"""exp-040f: sweep proxy_tokens for group-α AWQ IQ2_XS — can more/fewer α-search
tokens nudge top_p above imatrix-only?

Baselines (Qwopus3.5-27B-v3 IQ2_XS, group-α, plus_one=True, pre-fold imatrix):
  imatrix-only              top_p 81.70%  med_KLD 0.076
  AWQ group-α, tokens=1024  top_p 81.63%  med_KLD 0.080  (exp-040d)

proxy_tokens sets how many captured activation tokens feed the α-search proxy
loss (capped at ctx=4096). More tokens = a less-noisy α estimate; fewer = the
gemma-SOTA default. Single changed variable vs exp-040d: proxy_tokens. This
sweeps {512, 2048} to bracket 1024 → a clean 512/1024/2048 picture.

Each cell: calibrate (group-α, mix=None) → apply (plus_one=True) → convert →
quantize IQ2_XS → bench, then delete the ~54 GB folded F16 + HF intermediates.
Idempotent via step(); bench rows dedup by path.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040f_proxy_tokens_sweep.py
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

EXP40 = REPO / "out" / "exp-040"
HF_DIR = EXP40 / "model_extracted"
F16_GGUF = EXP40 / "model-f16.gguf"
IMATRIX = EXP40 / "imatrix-cal.gguf"
BASELINE_KLD = EXP40 / "baseline.kld"
EVAL_CORPUS = EXP40 / "corpora" / "corpus.eval.txt"
CAL_CORPUS = EXP40 / "corpora" / "corpus.cal.txt"

QUANT = "IQ2_XS"
PROXY = "iq2_xs"
CTX = 4096
EVAL_CTX = 4096
PROXY_TOKENS_SWEEP = (512, 2048)   # 1024 already measured in exp-040d


def run_cell(proxy_tokens: int, *, n_params: int, csv_path: Path) -> tuple[float, float]:
    out = EXP40 / f"iq2xs_awq_tok{proxy_tokens}"
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = out / "awq.pt"
    model_awq = out / "model_awq"
    f16_awq = out / "model-f16-awq.gguf"
    qpath = out / f"{QUANT}-awq-tok{proxy_tokens}.gguf"
    tag = f"[exp-040f][tok={proxy_tokens}]"

    with phase(f"{tag} AWQ calibrate (group-α, proxy_tokens={proxy_tokens})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 HF_DIR, CAL_CORPUS, bundle,
                 proxy=PROXY,
                 proxy_mix=None,
                 proxy_tokens=proxy_tokens,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=False,
                 cv_strategy="off",
             ))

    with phase(f"{tag} AWQ apply (plus_one=True)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, bundle, model_awq,
                 device="auto", dtype="bfloat16",
                 rmsnorm_plus_one=True, sanity_max_rel=0.60,
             ))

    with phase(f"{tag} convert -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"{tag} quantize {QUANT}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        label = (f"Jackrong/Qwopus3.5-27B-v3|{QUANT}|"
                 f"awq-group-alpha(tokens={proxy_tokens})+imatrix // baseline=FP16")
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
        )
        runner.append_row(csv_path, row)
        log(f"  tok={proxy_tokens}: top_p={row.same_top_p:.2f}%  "
            f"med_KLD={row.median_kld:.4f}  PPL={row.ppl:.3f}")

    # free the big intermediates; keep the small bundle + final gguf
    for child in (model_awq, f16_awq):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)
    return row.same_top_p, row.median_kld


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, IMATRIX, BASELINE_KLD,
              EVAL_CORPUS, CAL_CORPUS):
        if not p.exists():
            raise FileNotFoundError(f"exp-040f missing input: {p}")

    csv_path = EXP40 / "iq2xs_token_sweep.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    results: dict[int, tuple[float, float]] = {}
    for tok in PROXY_TOKENS_SWEEP:
        results[tok] = run_cell(tok, n_params=n_params, csv_path=csv_path)

    log("")
    log("=== exp-040f verdict (IQ2_XS group-α, proxy_tokens sweep) ===")
    log(f"  imatrix-only      : top_p 81.70%  med_KLD 0.076")
    log(f"  tokens=1024       : top_p 81.63%  med_KLD 0.080  (exp-040d)")
    for tok in PROXY_TOKENS_SWEEP:
        tp, kld = results[tok]
        log(f"  tokens={tok:<5}     : top_p {tp:6.2f}%  med_KLD {kld:.4f}  (this run)")
    log("  Any cell > 81.70% beats imatrix-only; flat across tokens => lever is dead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

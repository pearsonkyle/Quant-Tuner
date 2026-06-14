"""Experiment 029: AWQ proxy_tokens sweep on QAT IQ2_XS (headline row).

Background: exp-028 adopted proxy_tokens=512 (recipe inherited from exp-026)
and built every shipped AWQ row at that value. The codebook-aware proxy
(commit 5596783) landed in the same release. Result on QAT IQ2_XS:
  median KLD 1.192  (best in the published table)
  PPL        134.06
  top_p      47.97%  (regressed from 50.87% in the prior release)

Hypothesis: with the codebook-aware proxy now accurate, scoring α candidates
against more activation tokens should generalize better and recover some of
the lost top_p — the prior 512-token slice may not represent the full
distribution well enough for a per-tensor α to avoid being over-fit. Even
the cv-gate sees only one held-out chunk of the same size, so adding more
cal-side proxy tokens directly increases the signal the gate compares against.

Sweep: proxy_tokens ∈ {2048, 4096, 8192} at matching AWQ ctx.
  proxy=2048, ctx=4096  — fits inside the existing 4096-ctx chunk
  proxy=4096, ctx=4096  — uses the entire chunk
  proxy=8192, ctx=8192  — bumps AWQ ctx to fit; tests whether wider-context
                           activations carry calibration signal that 4096
                           doesn't, at ~2× per-step compute

The proxy_tokens > ctx silently-truncates bug is gone (the AWQ ctx default
was raised to 4096 in exp-024's writeup), so the proxy=4096 case is real,
not capped.

PER-MEMBER PROXY MIX (new since exp-028): IQ2_XS is a *mixed* quant —
llama-quantize bumps attn_v to Q4_K (Gemma-4 is GQA≥4) and the first eighth
of ffn_down to Q2_K, so scoring those members with the pure iq2_xs codebook
proxy optimizes α against an error the real quantization doesn't have. With
PROXY_MIX = QUANT (default below) awq.calibrate scores each member with the
proxy matching its real target type (see awq.proxy_for_member). NOTE: this
confounds the proxy_tokens comparison against the exp-028 proxy=512 baseline
(two changes at once) — set PROXY_MIX = None to reproduce exp-028's scoring
and isolate proxy_tokens. Mix runs write to "-mix"-suffixed dirs/GGUFs so
step() idempotency never reuses pre-fix artifacts.

Everything else matches exp-028's QAT IQ2_XS recipe:
  * source       : google/gemma-4-31B-it-qat-q4_0-unquantized
  * proxy        : iq2_xs (codebook-aware E8-lattice, 512 entries + parity)
  * α grid       : 0.0..1.0 step 0.25, per-tensor refinement radius 0.15
  * cv_strategy  : gate, cv_weight=1.0
  * sanity_max_rel: 1.20
  * imatrix      : reused from exp-022 (collected on QAT F16 at ctx=4096
                   with --parse-special)
  * corpora      : exp-020 cal/val/eval (identical to the published
                   calibration_data/ ships under the README)
  * eval         : ctx=4096, baseline.kld from exp-020

Outputs:
  * out/exp-029/qat-iq2_xs-proxy-{N}[-mix]/
      awq.pt
      gemma-4-31B-it-qat-IQ2_XS-awq-proxy-{N}[-mix].gguf  (final, kept)
      model_awq/, model-f16-awq.gguf                      (deleted post-bench)
  * out/exp-029/results.csv   (one row per N, plus a reprint of the
                               exp-028 proxy=512 baseline row for direct
                               comparison)

Wall-time estimate (Metal): ~15 hrs total — calibrate scales roughly
linearly with proxy_tokens (each candidate scored on more activations), so
proxy=4096 ≈ 4× proxy=512's ~2.5h calibrate ≈ ~10h, proxy=8192 at ctx=8192
adds activation-memory pressure on top.

Bench dedup + intermediate cleanup are baked in (lessons from the exp-028
restart). A partial re-run picks up exactly where it died.
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

# ---- inputs ----------------------------------------------------------------

QAT_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QAT_SLUG = QAT_ID.replace("/", "__")
VANILLA_SLUG = "google__gemma-4-31B-it"

EXP09 = REPO / "out" / "exp-009" / VANILLA_SLUG
EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP22 = REPO / "out" / "exp-022" / QAT_SLUG
EXP29 = REPO / "out" / "exp-029"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"

QAT_HF = EXP22 / "model_extracted"
QAT_IMATRIX = EXP22 / "imatrix-cal.gguf"

# For BPW + n_params we still measure against the vanilla F16 (the published
# table convention — vanilla and QAT have the same param count, so this just
# keeps the BPW column comparable to exp-028's results.csv).
VANILLA_F16 = EXP09 / "model-f16.gguf"

# ---- sweep -----------------------------------------------------------------

QUANT = "IQ2_XS"
PROXY = awq.proxy_for_quant_type(QUANT)  # iq2_xs (codebook-aware)
# Score each group member with the proxy matching the tensor type
# llama-quantize actually assigns under IQ2_XS (attn_v -> Q4_K on GQA>=4,
# first eighth of ffn_down -> Q2_K). None reproduces exp-028's uniform
# iq2_xs scoring; see the docstring before flipping.
PROXY_MIX: str | None = QUANT
EVAL_CTX = 4096

# (proxy_tokens, ctx) tuples. proxy_tokens > ctx silently truncates inside
# awq.calibrate (target_X is populated from the first chunk only), so the
# 8192 case bumps ctx to 8192.
SWEEP: tuple[tuple[int, int], ...] = (
    (2048, 4096),
    (4096, 4096),
    (8192, 8192),
)


# ---------------------------------------------------------------------------

def _check_inputs() -> None:
    required = [
        QAT_HF / "config.json", QAT_IMATRIX, VANILLA_F16,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-029 missing inputs:\n  " + "\n  ".join(missing))


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
    """Delete ~115 GB of HF-folded weights + F16-AWQ GGUF after the final
    quantized GGUF is shipped and benched."""
    for child in (sub / "model_awq", sub / "model-f16-awq.gguf"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)


def _run_one(proxy_tokens: int, ctx: int, *, csv_path: Path) -> None:
    # "-mix" suffix keeps mix and non-mix artifacts apart: step() idempotency
    # is existence-based, so reusing a name would silently rebench stale
    # pre-fix files under the new label.
    suffix = "-mix" if PROXY_MIX else ""
    sub = EXP29 / f"qat-iq2_xs-proxy-{proxy_tokens}{suffix}"
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / f"gemma-4-31B-it-qat-IQ2_XS-awq-proxy-{proxy_tokens}{suffix}.gguf"
    n_params = bpw_mod.n_params(VANILLA_F16)

    tag = f"[exp-029][proxy={proxy_tokens},ctx={ctx},mix={PROXY_MIX}]"

    with phase(f"{tag} AWQ calibrate (proxy={PROXY})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 QAT_HF, CAL_CORPUS, bundle,
                 proxy=PROXY,
                 proxy_mix=PROXY_MIX,
                 proxy_tokens=proxy_tokens,
                 ctx=ctx,
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
                 QAT_HF, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"{tag} quantize IQ2_XS"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, QUANT, imatrix=QAT_IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        label = (f"{QAT_ID}|IQ2_XS|awq-cv-gate+imatrix|"
                 f"proxy={PROXY},mix={PROXY_MIX},tokens={proxy_tokens},ctx={ctx} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_intermediates(sub)


# ---------------------------------------------------------------------------

def main() -> int:
    _check_inputs()
    EXP29.mkdir(parents=True, exist_ok=True)
    csv_path = EXP29 / "results.csv"

    for proxy_tokens, ctx in SWEEP:
        _run_one(proxy_tokens, ctx, csv_path=csv_path)

    log("")
    log("=== exp-029 complete ===")
    log("Reference (exp-028, proxy=512, ctx=4096, uniform iq2_xs scoring):")
    log("  QAT IQ2_XS AWQ: PPL=134.06 mean_KLD=1.900 med_KLD=1.192 top_p=47.97%")
    if PROXY_MIX:
        log("NOTE: this sweep ran with the per-member proxy mix ON — rows are")
        log("      not a pure proxy_tokens comparison against the exp-028 line.")
    log("")
    log("This sweep:")
    log(f"  results: {csv_path}")
    log("Next: pick a proxy_tokens winner on the (top_p, median_KLD) frontier;")
    log("      if it differs from 512, re-run the full table at the new value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""exp-051: does the updated calibration methodology improve Ornith-1.0-9B?

Ornith-1.0-9B (deepreinforce-ai) was released in exp-050 as an imatrix-only
Q5_K_M. Its calibration corpus was built with the PRE-fix build_corpora
(wiki prepended as a monolith, tool schemas re-rendered in EVERY window, no
`boilerplate_tokens` accounting). This experiment A/B-tests the PR
`claude/quantization-calibration-audit-9djodt` on the one lever that applies
to an imatrix model — **tool-schema dedup in the calibration corpus** — plus
an **AWQ arm** that exercises the headline strided-ingestion fix.

Three arms, held-fixed evals (the SHIPPED general + tools eval corpora, so we
measure the calibration change, not an eval change), one shared FP16 baseline:

  OLD  imatrix  cal = shipped corpus.cal.txt (pre-fix)          -> hybrid_custom
  NEW  imatrix  cal = rebuilt w/ tool_schema_quota=1 (this PR)  -> hybrid_custom
  AWQ           cal = NEW corpus, strided ingestion, plus_one   -> folded imatrix + hybrid_custom

Quants: imatrix arms build IQ2_M / IQ3_M / IQ4_XS (2->4 bit). The AWQ arm
builds only the low-bit targets where AWQ has ever shown headroom on a Qwen3.5
model (IQ2_M, IQ3_M); AWQ at 4-bit is a known no-op and 4-bit on this hybrid
linear-attention trunk is not worth the calibrate cost.

Ornith is a hybrid arch: 8/32 layers are full_attention, 24 are
linear_attention. AWQ's discover_groups only scales the full-attention layers'
attention + every layer's MLP; the linear-attention projections are quantized
under the (folded) imatrix like any other tensor. So the AWQ arm is a PARTIAL
scaling — this is the "attempt" arm; if apply()'s logit-drift sanity fails we
report the imatrix A/B alone.

Prereq: scripts/exp050_setup_ornith.py (F16 GGUF + vision-stripped HF dir).

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp051_ornith_methods_ab.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import awq
from quant_tuner.calibrate import imatrix as imatrix_cal
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import convert, gguf

MODEL_ID = "deepreinforce-ai/Ornith-1.0-9B"

# --- inputs (from exp-050 setup) -------------------------------------------
EXP50 = REPO / "out" / "exp-050"
HF_DIR = EXP50 / "model_extracted"
F16_GGUF = EXP50 / "model-f16.gguf"

# Held-fixed evals + the OLD (shipped) calibration corpus.
SHIP = REPO / "uploads" / "pearsonkyle" / "Ornith-1.0-9B-imatrix-GGUF" / "calibration_data"
OLD_CAL = SHIP / "corpus.cal.txt"
GEN_EVAL = SHIP / "corpus.eval.general.txt"
TOOLS_EVAL = SHIP / "corpus.eval.tools.txt"

# Corpus-rebuild inputs (NEW arm).
WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

# --- workspace -------------------------------------------------------------
EXP = REPO / "out" / "exp-051"
LOGS = EXP / "logs"
NEW_CORPORA = EXP / "corpora_new"
NEW_CAL = NEW_CORPORA / "corpus.cal.txt"
NEW_AUDIT = NEW_CORPORA / "corpora_audit.json"

GEN_BASELINE = EXP / "baseline.general.kld"
TOOLS_BASELINE = EXP / "baseline.tools.kld"
GEN_CSV = EXP / "results.general.csv"
TOOLS_CSV = EXP / "results.tools.csv"

CTX = 4096
EVAL_CTX = 4096

IMATRIX_QUANTS = ("IQ2_M", "IQ3_M", "IQ4_XS")
# IQ4_XS added after the IQ2_M agentic result — static priors were wrong at
# 2-bit, so we measure 4-bit AWQ agentically rather than assume it's a no-op.
AWQ_QUANTS = ("IQ2_M", "IQ3_M", "IQ4_XS")


def _load_build_corpora():
    path = REPO / "scripts" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("build_corpora", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_has(csv_path: Path, needle: str) -> bool:
    return csv_path.exists() and any(needle in ln for ln in csv_path.open())


def _bench(label: str, qpath: Path, n_params: float, log_dir: Path) -> None:
    """Bench one quant on BOTH held-fixed eval corpora (idempotent per CSV)."""
    for corpus, baseline, csv_path, tag in (
        (GEN_EVAL, GEN_BASELINE, GEN_CSV, "general"),
        (TOOLS_EVAL, TOOLS_BASELINE, TOOLS_CSV, "tools"),
    ):
        if _csv_has(csv_path, str(qpath)):
            log(f"    {tag}: {qpath.name} already benched — skip")
            continue
        with phase(f"bench {tag}: {qpath.name}"):
            r = runner.bench_one(
                qpath, f"{label} // eval={tag}",
                reference_n_params=n_params,
                eval_dataset=corpus, eval_baseline=baseline,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
            )
            runner.append_row(csv_path, r)
            log(f"    {tag}: bpw={r.bpw:.3f} PPL={r.ppl:.4f} "
                f"medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")


def _imatrix_arm(arm: str, cal_corpus: Path, n_params: float) -> None:
    """OLD/NEW arm: base imatrix on `cal_corpus` -> hybrid_custom -> quant ladder."""
    sub = EXP / arm
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_im = sub / "imatrix-base.gguf"
    hybrid_im = sub / "imatrix-hybrid_custom.gguf"

    with phase(f"[{arm}] base imatrix (ctx={CTX})"):
        step("imatrix-base", base_im,
             lambda: llama_cpp.imatrix(F16_GGUF, cal_corpus, base_im,
                                       ctx=CTX, log=log_dir / "imatrix-base.log"))
    with phase(f"[{arm}] re-weight -> hybrid_custom"):
        step("imatrix-hybrid", hybrid_im,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=F16_GGUF,
                 base_imatrix=base_im, out_path=hybrid_im))

    for qt in IMATRIX_QUANTS:
        qpath = sub / f"Ornith-1.0-9B-{qt}-{arm}.gguf"
        with phase(f"[{arm}] quantize {qt}"):
            step("quantize", qpath,
                 lambda q=qpath, t=qt: gguf.quantize(
                     F16_GGUF, q, t, imatrix=hybrid_im,
                     log=log_dir / f"quantize-{t}.log"))
        _bench(f"{MODEL_ID}|{qt}|imatrix-{arm}", qpath, n_params, log_dir)


def _awq_arm(n_params: float) -> None:
    """AWQ arm on the NEW corpus: per-target calibrate -> fold (plus_one) ->
    convert -> imatrix on the FOLDED F16 -> hybrid_custom -> quant -> bench."""
    for qt in AWQ_QUANTS:
        sub = EXP / f"awq_{qt.lower()}"
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        bundle = sub / "awq.pt"
        model_awq = sub / "model_awq"
        f16_awq = sub / "model-f16-awq.gguf"
        base_im = sub / "imatrix-awq-base.gguf"
        hybrid_im = sub / "imatrix-awq-hybrid_custom.gguf"
        qpath = sub / f"Ornith-1.0-9B-{qt}-awq.gguf"

        # Proxy matched to target error shape (mirrors pipeline._calibrate_awq
        # / the shipped iq2_m_awq recipe pin). Only IQ1/IQ2/IQ3/Q2/Q3 ftypes are
        # tensor *mixes* that want a proxy_mix; IQ4_XS is a pure 4-bit ftype
        # (int4_g128 proxy, no mix — pipeline sets proxy_mix only for low-bit).
        proxy = "q2k_b16" if qt == "IQ2_M" else awq.proxy_for_quant_type(qt)
        proxy_mix = qt if qt.upper().startswith(("IQ1", "IQ2", "IQ3", "Q2", "Q3")) else None

        with phase(f"[awq {qt}] calibrate (strided, proxy={proxy}, mix={proxy_mix})"):
            step("awq-calibrate", bundle,
                 lambda b=bundle, p=proxy, mx=proxy_mix: awq.calibrate(
                     HF_DIR, NEW_CAL, b,
                     proxy=p, proxy_mix=mx, ctx=CTX,
                     device="auto", dtype="bfloat16",
                     per_tensor_alpha=False, cv_strategy="off"))

        with phase(f"[awq {qt}] apply (rmsnorm_plus_one=True)"):
            # Hybrid linear-attention trunk => partial scaling; relax the drift
            # gate a touch (only full-attn layers fold) but still catch a
            # wrong-form fold.
            step("awq-apply", model_awq / "config.json",
                 lambda m=model_awq: awq.apply(
                     HF_DIR, bundle, m, device="auto", dtype="bfloat16",
                     rmsnorm_plus_one=True, sanity_max_rel=0.60))

        with phase(f"[awq {qt}] convert -> F16 GGUF"):
            step("awq-convert", f16_awq,
                 lambda: convert.hf_to_f16_gguf(
                     model_awq, f16_awq, log=log_dir / "convert.log"))

        with phase(f"[awq {qt}] imatrix on FOLDED F16"):
            step("awq-imatrix-base", base_im,
                 lambda: llama_cpp.imatrix(f16_awq, NEW_CAL, base_im,
                                           ctx=CTX, log=log_dir / "imatrix.log"))
        with phase(f"[awq {qt}] re-weight folded -> hybrid_custom"):
            step("awq-imatrix-hybrid", hybrid_im,
                 lambda: imatrix_cal.calibrate(
                     variant="hybrid_custom", f16_gguf=f16_awq,
                     base_imatrix=base_im, out_path=hybrid_im))

        with phase(f"[awq {qt}] quantize"):
            step("quantize", qpath,
                 lambda q=qpath, t=qt: gguf.quantize(
                     f16_awq, q, t, imatrix=hybrid_im,
                     log=log_dir / "quantize.log"))
        _bench(f"{MODEL_ID}|{qt}|awq+imatrix", qpath, n_params, log_dir)


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, OLD_CAL, GEN_EVAL, TOOLS_EVAL,
              WIKI_TEST, MMMU_VAL):
        if not p.exists():
            raise FileNotFoundError(f"exp-051 missing input: {p}")
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"[exp-051] reference n_params = {n_params:,.0f}")

    # ---- shared FP16 baselines on the held-fixed eval corpora --------------
    with phase("[exp-051] FP16 baseline (general)"):
        step("baseline-general", GEN_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, GEN_EVAL, GEN_BASELINE,
                                        ctx=EVAL_CTX, log=LOGS / "baseline-general.log"))
    with phase("[exp-051] FP16 baseline (tools)"):
        step("baseline-tools", TOOLS_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, TOOLS_EVAL, TOOLS_BASELINE,
                                        ctx=EVAL_CTX, log=LOGS / "baseline-tools.log"))

    # ---- NEW calibration corpus (this PR's methods) ------------------------
    with phase("[exp-051] rebuild corpus (tool_schema_quota=1, wiki interleave)"):
        def _build():
            bc = _load_build_corpora()
            bc.build(
                out_dir=NEW_CORPORA, model_dir=HF_DIR, wiki_test=WIKI_TEST,
                cal_tokens=500_000, val_tokens=0, eval_tokens_per_domain=30_000,
                seed=42, val_supplement_override=MMMU_VAL,
            )
        step("build_corpora_new", NEW_AUDIT, _build)

    # ---- arms --------------------------------------------------------------
    # The OLD-vs-NEW imatrix A/B was dropped: the audit showed the schema-dedup
    # lever is a near-no-op on this corpus (7/84 sessions stubbed; agent tool
    # sets are ~unique per session), so the corpus is 99.9% identical and the
    # A/B is a known wash. The NEW imatrix arm is the imatrix baseline-to-beat;
    # the AWQ arm tests whether the PR's headline strided-ingestion fix +
    # activation-aware scaling beats plain imatrix at low bit.
    _imatrix_arm("new", NEW_CAL, n_params)
    try:
        _awq_arm(n_params)
    except Exception as e:  # noqa: BLE001
        log(f"[exp-051] AWQ arm FAILED ({type(e).__name__}: {e}); "
            f"reporting imatrix arm only")

    log("")
    log("=== exp-051 complete ===")
    log(f"  general: {GEN_CSV}")
    log(f"  tools:   {TOOLS_CSV}")
    log(f"  new-corpus audit: {NEW_AUDIT}  (diff vs shipped corpora_audit.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

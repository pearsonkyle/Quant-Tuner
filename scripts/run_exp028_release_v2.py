"""Experiment 028: gemma-4-31B-it AWQ 2-bit release rebuild (v2).

Regenerates every GGUF in the published README table under the latest
calibration code:

  * codebook-aware IQ2 proxies (commit 5596783) — IQ2_XS uses the exact
    E8-lattice iq2_xs grid, IQ2_M uses iq2_s, Q2_K_S uses q2k_b16. The α
    search now sees the same quant geometry that llama-quantize will apply,
    instead of every IQ2_* target sharing a single q2k_b16 proxy.
  * tokenization-consistency audit (commit a36d3f4) — llama-imatrix collects
    with --parse-special on chat-templated corpora; the bench's eval corpus
    is `corpus.eval.txt` (raw external code/math/tools text — no chat
    markers, so llama-perplexity tokenizes it correctly despite lacking
    --parse-special); HF-side forward passes go through forward_no_logits
    to skip the LM-head matmul and `[ctx, vocab]` tensor.
  * project-wide AWQ defaults proxy=512, ctx=4096 (exp-026/-027).

Inputs (all already on disk; nothing re-collected):
  * vanilla F16   : out/exp-009/google__gemma-4-31B-it/model-f16.gguf
  * vanilla HF    : out/exp-009/google__gemma-4-31B-it/model_extracted/
  * QAT F16       : out/exp-022/google__gemma-4-31B-it-qat-q4_0-unquantized/model-f16.gguf
  * QAT HF        : out/exp-022/google__gemma-4-31B-it-qat-q4_0-unquantized/model_extracted/
  * imatrix (vanilla F16, ctx=4096, --parse-special) : out/exp-020/.../imatrix-cal.gguf
  * imatrix (QAT F16,     ctx=4096, --parse-special) : out/exp-022/.../imatrix-cal.gguf
  * corpora (cal/val/eval, identical to README/calibration_data/) : exp-020/corpora/
  * baseline.kld (FP16 reference KLD on corpus.eval.txt) : exp-020/baseline.kld

Outputs land under out/exp-028/{vanilla,qat}/ as drop-in renames of the
README's quant filenames, ready to copy into the uploads/ dir.

13 quants total — 4 vanilla baselines (imatrix-only IQ2_XS/IQ2_M/Q2_K_S +
plain Q2_K), 3 vanilla AWQ (IQ2_XS/IQ2_M/Q2_K_S, one calibrate per target),
3 QAT baselines (imatrix-only IQ2_XS/IQ2_M/Q2_K_S), 3 QAT AWQ
(IQ2_XS/IQ2_M/Q2_K_S).

QAT IQ2_M was broken in the prior release (PPL ≈ 2e10, KLD ≈ 23, top_p 0%
in both imatrix-only and AWQ arms). It is re-attempted here because the
codebook-aware iq2_s proxy may rescue the geometric mismatch — the prior
runs scored every IQ2_M α candidate through the q2k_b16 proxy, which
cannot see the iq2_s codebook's sign-pair / lattice constraints. Treat
results as diagnostic: if PPL/KLD still blow up, the file is dropped on
the README-update pass and the callout stays.

Wall-time (Metal, M-series): ~6–8 hours. Each AWQ calibrate ~7 min, apply
~5 min, convert ~2 min; quantize+bench dominates (~20 min × 11 quants).

Every stage is wrapped in `experiments.step()` so partial reruns skip work
whose outputs already exist. Bench rows are appended to a single
results.csv that the README update step pulls from.
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

# ---- inputs ----------------------------------------------------------------

VANILLA_ID = "google/gemma-4-31B-it"
VANILLA_SLUG = VANILLA_ID.replace("/", "__")
QAT_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QAT_SLUG = QAT_ID.replace("/", "__")

EXP09 = REPO / "out" / "exp-009" / VANILLA_SLUG
EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP22 = REPO / "out" / "exp-022" / QAT_SLUG
EXP28 = REPO / "out" / "exp-028"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"   # logtrain TRAIN + wiki.test.raw
VAL_CORPUS = CORPORA / "corpus.val.txt"   # logtrain TEST + MMMU supplement
EVAL_CORPUS = CORPORA / "corpus.eval.txt" # external code/math/tools — no chat markers
BASELINE_KLD = EXP20 / "baseline.kld"

VANILLA_HF = EXP09 / "model_extracted"
VANILLA_F16 = EXP09 / "model-f16.gguf"
VANILLA_IMATRIX = EXP20 / "imatrix-cal.gguf"

QAT_HF = EXP22 / "model_extracted"
QAT_F16 = EXP22 / "model-f16.gguf"
QAT_IMATRIX = EXP22 / "imatrix-cal.gguf"

# ---- knobs (the latest defaults, listed explicitly) -----------------------

EVAL_CTX = 4096
AWQ_CTX = 4096
AWQ_PROXY_TOKENS = 512

VANILLA_QUANTS = ("IQ2_XS", "IQ2_M", "Q2_K_S")
QAT_QUANTS = ("IQ2_XS", "IQ2_M", "Q2_K_S")  # IQ2_M re-attempted with codebook-aware proxy

# Output filenames matching the README's published names (minus the model
# prefix, which is added when copying into uploads/).
def _vanilla_quant_name(q: str, mode: str) -> str:
    return f"gemma-4-31B-it-{q}-{mode}.gguf"


def _qat_quant_name(q: str, mode: str) -> str:
    return f"gemma-4-31B-it-qat-{q}-{mode}.gguf"


# ---------------------------------------------------------------------------

def _check_inputs() -> None:
    required = [
        VANILLA_HF / "config.json", VANILLA_F16, VANILLA_IMATRIX,
        QAT_HF / "config.json", QAT_F16, QAT_IMATRIX,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-028 missing inputs:\n  " + "\n  ".join(missing))


def _csv_has_qpath(csv_path: Path, qpath: Path) -> bool:
    """Idempotency for the bench step (runner.bench_one isn't step-wrapped):
    skip if this exact final GGUF already has a row in the CSV."""
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        for line in fh:
            if needle in line:
                return True
    return False


def _bench(qpath: Path, label: str, reference_n_params: int, csv_path: Path,
           log_dir: Path) -> runner.BenchRow | None:
    if _csv_has_qpath(csv_path, qpath):
        log(f"  {qpath.name}: bench row already in CSV — skipping")
        return None
    row = runner.bench_one(
        qpath, label,
        reference_n_params=reference_n_params,
        eval_dataset=EVAL_CORPUS,
        eval_baseline=BASELINE_KLD,
        eval_ctx=EVAL_CTX,
        log_dir=log_dir,
        suite="kld",
    )
    runner.append_row(csv_path, row)
    log(f"  {qpath.name}: PPL={row.ppl:.4f} KLD={row.mean_kld:.5f} "
        f"top_p={row.same_top_p:.4f}%")
    return row


def _free_awq_intermediates(sub: Path) -> None:
    """Delete the ~115 GB of HF-folded weights + F16-AWQ GGUF after the final
    quantized GGUF has been written and benched. Leave awq.pt (~9 MB) for
    provenance. Without this, three 60-GB-class AWQ runs back-to-back fill
    the disk before the third one finishes (the 18:25 crash on the first
    attempt)."""
    import shutil
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    if model_awq.exists():
        shutil.rmtree(model_awq, ignore_errors=True)
    if f16_awq.exists():
        f16_awq.unlink(missing_ok=True)


def _run_imatrix_only(*, source_id: str, source_slug: str, src_f16: Path,
                      imatrix: Path, quants: tuple[str, ...],
                      name_fn, out_dir: Path, csv_path: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(VANILLA_F16)
    for q in quants:
        qpath = out_dir / name_fn(q, "imatrix")
        with phase(f"[exp-028][{source_slug}] quantize {q} (imatrix-only)"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     src_f16, p, qt, imatrix=imatrix,
                     log=log_dir / f"quantize-{qt}-imatrix.log"))
        with phase(f"[exp-028][{source_slug}] bench {q} (imatrix-only)"):
            label = (f"{source_id}|{q}|imatrix-only|"
                     f"ctx={EVAL_CTX} // baseline=vanilla-FP16")
            _bench(qpath, label, n_params, csv_path, log_dir)


def _run_awq(*, source_id: str, source_slug: str, src_hf: Path,
             imatrix: Path, quants: tuple[str, ...], name_fn,
             out_dir: Path, csv_path: Path) -> None:
    """Per-target AWQ: each quant gets its own α search with the matching
    codebook proxy, its own folded HF, and its own F16-AWQ GGUF."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(VANILLA_F16)

    for q in quants:
        proxy = awq.proxy_for_quant_type(q)
        sub = out_dir / f"awq-{q.lower()}"
        sub.mkdir(parents=True, exist_ok=True)

        bundle = sub / "awq.pt"
        model_awq = sub / "model_awq"
        f16_awq = sub / "model-f16-awq.gguf"

        with phase(f"[exp-028][{source_slug}] AWQ calibrate {q} "
                   f"(proxy={proxy}, tokens={AWQ_PROXY_TOKENS}, ctx={AWQ_CTX})"):
            step(f"AWQ calibrate {q}", bundle,
                 lambda b=bundle, p=proxy: awq.calibrate(
                     src_hf, CAL_CORPUS, b,
                     proxy=p,
                     proxy_tokens=AWQ_PROXY_TOKENS,
                     ctx=AWQ_CTX,
                     device="auto",
                     dtype="bfloat16",
                     per_tensor_alpha=True,
                     per_tensor_grid_radius=0.15,
                     holdout_text=VAL_CORPUS,
                     cv_strategy="gate",
                     cv_weight=1.0,
                 ))

        with phase(f"[exp-028][{source_slug}] AWQ apply {q}"):
            step(f"AWQ apply {q}", model_awq / "config.json",
                 lambda mw=model_awq, b=bundle: awq.apply(
                     src_hf, b, mw,
                     device="auto",
                     dtype="bfloat16",
                     rmsnorm_plus_one=False,
                     sanity_max_rel=1.20,
                 ))

        with phase(f"[exp-028][{source_slug}] convert AWQ-folded HF -> F16 GGUF ({q})"):
            step(f"convert {q}", f16_awq,
                 lambda fa=f16_awq, mw=model_awq, qt=q: convert.hf_to_f16_gguf(
                     mw, fa, log=log_dir / f"convert-{qt}.log"))

        qpath = out_dir / name_fn(q, "awq-cv-gate")
        with phase(f"[exp-028][{source_slug}] quantize {q} (AWQ cv-gate + imatrix)"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q, fa=f16_awq: gguf.quantize(
                     fa, p, qt, imatrix=imatrix,
                     log=log_dir / f"quantize-{qt}-awq.log"))
        with phase(f"[exp-028][{source_slug}] bench {q} (AWQ cv-gate)"):
            label = (f"{source_id}|{q}|awq-cv-gate+imatrix|"
                     f"proxy={proxy},tokens={AWQ_PROXY_TOKENS},ctx={AWQ_CTX} "
                     f"// baseline=vanilla-FP16")
            _bench(qpath, label, n_params, csv_path, log_dir)

        # Final GGUF is shipped + bench row is in CSV: drop the ~115 GB of
        # intermediates so the next AWQ target's apply doesn't fill the disk.
        _free_awq_intermediates(sub)


# ---------------------------------------------------------------------------

def main() -> int:
    _check_inputs()
    EXP28.mkdir(parents=True, exist_ok=True)

    vanilla_dir = EXP28 / "vanilla"
    qat_dir = EXP28 / "qat"
    vanilla_dir.mkdir(parents=True, exist_ok=True)
    qat_dir.mkdir(parents=True, exist_ok=True)
    vanilla_csv = vanilla_dir / "results.csv"
    qat_csv = qat_dir / "results.csv"
    vanilla_logs = vanilla_dir / "logs"
    vanilla_logs.mkdir(parents=True, exist_ok=True)

    # ---- vanilla: plain Q2_K (no calibration) -----------------------------
    plain_path = vanilla_dir / _vanilla_quant_name("Q2_K", "plain")
    with phase("[exp-028][vanilla] quantize Q2_K (plain, no calibration)"):
        step("quantize Q2_K plain", plain_path,
             lambda: gguf.quantize(
                 VANILLA_F16, plain_path, "Q2_K", imatrix=None,
                 log=vanilla_logs / "quantize-Q2_K-plain.log"))
    with phase("[exp-028][vanilla] bench Q2_K (plain)"):
        n_params = bpw_mod.n_params(VANILLA_F16)
        _bench(plain_path,
               f"{VANILLA_ID}|Q2_K|plain // baseline=vanilla-FP16",
               n_params, vanilla_csv, vanilla_logs)

    # ---- vanilla: imatrix-only baselines ----------------------------------
    _run_imatrix_only(
        source_id=VANILLA_ID, source_slug="vanilla",
        src_f16=VANILLA_F16, imatrix=VANILLA_IMATRIX,
        quants=VANILLA_QUANTS, name_fn=_vanilla_quant_name,
        out_dir=vanilla_dir, csv_path=vanilla_csv,
    )

    # ---- vanilla: AWQ cv-gate (per-target, codebook-aware proxy) ----------
    _run_awq(
        source_id=VANILLA_ID, source_slug="vanilla",
        src_hf=VANILLA_HF, imatrix=VANILLA_IMATRIX,
        quants=VANILLA_QUANTS, name_fn=_vanilla_quant_name,
        out_dir=vanilla_dir, csv_path=vanilla_csv,
    )

    # ---- QAT: imatrix-only baselines --------------------------------------
    _run_imatrix_only(
        source_id=QAT_ID, source_slug="qat",
        src_f16=QAT_F16, imatrix=QAT_IMATRIX,
        quants=QAT_QUANTS, name_fn=_qat_quant_name,
        out_dir=qat_dir, csv_path=qat_csv,
    )

    # ---- QAT: AWQ cv-gate (per-target) ------------------------------------
    _run_awq(
        source_id=QAT_ID, source_slug="qat",
        src_hf=QAT_HF, imatrix=QAT_IMATRIX,
        quants=QAT_QUANTS, name_fn=_qat_quant_name,
        out_dir=qat_dir, csv_path=qat_csv,
    )

    log("")
    log("=== exp-028 complete ===")
    log(f"  vanilla results: {vanilla_csv}")
    log(f"  qat     results: {qat_csv}")
    log("  next: copy GGUFs to uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/")
    log("        then regenerate README §1 tables from the two CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

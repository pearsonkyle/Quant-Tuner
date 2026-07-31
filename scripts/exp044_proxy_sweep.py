"""exp-044 Stage C0: AWQ proxy sweep on the new windowed corpus (vanilla only).

The proxy that minimizes AWQ's alpha-search loss can shift with the calibration
corpus and the llama.cpp build, so we re-pick it before locking the 11-row release.
Vanilla source only (the winner carries to QAT); AWQ rows only. Reuses the exp-044
setup inputs (run `exp044_release_reproduce.py setup` first).

Trimmed grid (~5-6 candidates) — README proxy vs its most-likely rival per target:
  IQ2_XS : auto iq2_xs codebook   vs  q2k_b16
  IQ2_M  : q2k_b16 (no mix)       vs  q2k_b16 + iq3_s mix (proxy_mix=IQ2_M)
  Q2_K_S : auto q2k_b16           vs  q2k_super

Decide by MEDIAN KLD then same_top_p on corpus.eval.txt (KLD/top_p are the reliable
signals; speed is thermal-noisy). Then set PROXY/PROXY_MIX in
exp044_release_reproduce.py to the winners and run the `quants` stage.

    PYTHONPATH=src .venv/bin/python scripts/exp044_proxy_sweep.py
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import convert, gguf


def _load_release():
    spec = importlib.util.spec_from_file_location(
        "exp044_release_reproduce", REPO / "scripts" / "exp044_release_reproduce.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp044_release_reproduce"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_release()  # shared paths + knobs (single source of truth)

EXP44 = R.EXP44
SWEEP = EXP44 / "sweep"
CSV = SWEEP / "results.csv"

SRC_HF = R.VANILLA_HF
SRC_F16 = R.VANILLA_F16
SRC_IMATRIX = R.VANILLA_IMATRIX
CAL_CORPUS = R.CAL_CORPUS
VAL_CORPUS = R.VAL_CORPUS
EVAL_CORPUS = R.EVAL_CORPUS
BASELINE_KLD = R.BASELINE_KLD


@dataclass(frozen=True)
class Cand:
    tag: str                 # path-safe label
    quant: str
    proxy: str | None        # None -> awq.proxy_for_quant_type()
    proxy_mix: str | None


CANDIDATES: tuple[Cand, ...] = (
    Cand("iq2xs_auto",     "IQ2_XS", None,        None),
    Cand("iq2xs_q2kb16",   "IQ2_XS", "q2k_b16",   None),
    Cand("iq2m_q2kb16",    "IQ2_M",  "q2k_b16",   None),
    Cand("iq2m_iq3smix",   "IQ2_M",  "q2k_b16",   "IQ2_M"),
    Cand("q2ks_q2kb16",    "Q2_K_S", "q2k_b16",   None),
    Cand("q2ks_q2ksuper",  "Q2_K_S", "q2k_super", None),
)


def _csv_has_qpath(qpath: Path) -> bool:
    if not CSV.exists():
        return False
    needle = str(qpath)
    with CSV.open() as fh:
        return any(needle in line for line in fh)


def run_candidate(c: Cand, n_params: int) -> None:
    sub = SWEEP / c.tag
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / f"gemma-4-31B-it-{c.quant}-awq-{c.tag}.gguf"
    proxy = c.proxy or awq.proxy_for_quant_type(c.quant)
    tag = f"[exp-044][sweep][{c.tag}]"

    with phase(f"{tag} AWQ calibrate (proxy={proxy}, mix={c.proxy_mix})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 SRC_HF, CAL_CORPUS, bundle,
                 proxy=proxy,
                 proxy_mix=c.proxy_mix,
                 proxy_tokens=R.PROXY_TOKENS,
                 ctx=R.CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=R.PER_TENSOR_GRID_RADIUS,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase(f"{tag} AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_HF, bundle, model_awq,
                 device="auto", dtype="bfloat16",
                 rmsnorm_plus_one=R.RMSNORM_PLUS_ONE,
                 sanity_max_rel=R.SANITY_MAX_REL,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq,
                                            log=log_dir / "convert.log"))

    with phase(f"{tag} quantize {c.quant}"):
        step("quantize", qpath,
             lambda: gguf.quantize(f16_awq, qpath, c.quant, imatrix=SRC_IMATRIX,
                                   log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        if _csv_has_qpath(qpath):
            log(f"  {qpath.name}: already in CSV — skipping")
        else:
            label = (f"{R.VANILLA_ID}|{c.quant}|awq-cv-gate+imatrix|"
                     f"proxy={proxy},mix={c.proxy_mix} // SWEEP // baseline=vanilla-FP16")
            row = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
                eval_ctx=R.EVAL_CTX, log_dir=log_dir, suite="kld")
            runner.append_row(CSV, row)
            log(f"  {c.tag}: PPL={row.ppl:.4f} medKLD={row.median_kld:.5f} "
                f"top_p={row.same_top_p:.4f}%")

    # free the ~115 GB intermediates; keep the small GGUF + bundle for reference
    for child in (model_awq, f16_awq):
        if child.is_dir():
            import shutil
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)


def main() -> int:
    for p in (SRC_HF / "config.json", SRC_F16, SRC_IMATRIX,
              CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD):
        if not p.exists():
            raise FileNotFoundError(
                f"exp-044 sweep: run `exp044_release_reproduce.py setup` first — missing {p}")
    SWEEP.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(SRC_F16)
    log(f"[exp-044][sweep] {len(CANDIDATES)} candidates  proxy_tokens={R.PROXY_TOKENS} ctx={R.CTX}")
    for c in CANDIDATES:
        run_candidate(c, n_params)
    log("")
    log("=== exp-044 proxy sweep complete ===")
    log(f"  results: {CSV}")
    log("  Pick winners by median KLD then same_top_p; set PROXY/PROXY_MIX in")
    log("  exp044_release_reproduce.py, then run its `quants` stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

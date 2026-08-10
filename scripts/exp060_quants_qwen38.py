"""exp-060 stage 2+3: universal corpus -> hybrid imatrix -> IQ2_M / IQ3_M / IQ4_XS / Q5_K_M.

Consumes ``scripts/exp060_setup_qwen38.py``'s outputs (extracted HF dir + F16 GGUF +
``mtp_report.json``). Builds the universal calibration corpus (every dataset in
``datasets/`` + wiki), collects the
base imatrix at ctx=4096, re-weights it to the published ``hybrid_custom`` variant, then
quantizes the four-row lineup — each with **the whole MTP draft layer pinned to Q8_0**, so
the head stays near-lossless while the trunk goes low-bit (a 2-bit head drafts badly, and
the head is tiny relative to the model).

The pin is read from ``mtp_report.json``, not hardcoded: ``llama-quantize`` silently
accepts a ``--tensor-type`` pattern that matches nothing, so a stale ``blk.64.`` on a model
with a different depth would quantize the draft head along with the trunk and show up only
as a mediocre acceptance rate.

Each eval corpus is a separate distribution with its OWN baseline and results CSV — they
are never concatenated.

    PYTHONPATH=src .venv/bin/python scripts/exp060_quants_qwen38.py
    PYTHONPATH=src .venv/bin/python scripts/exp060_quants_qwen38.py \\
        --run exp-060-dryrun --rows IQ4_XS --evals general   # plumbing test, one row
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import imatrix as imatrix_cal
from quant_tuner.data import universal
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp, mtp
from quant_tuner.quantize import gguf

DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
DEFAULT_RUN = "exp-060"
# The published ladder: one aggressive 2-bit row, two mid rows, one near-lossless row.
DEFAULT_ROWS = ("IQ2_M", "IQ3_M", "IQ4_XS", "Q5_K_M")
CTX = 4096          # matches DEFAULT_IMATRIX_CTX; numbers at 512 are NOT comparable
EVAL_CTX = 4096

# eval name -> (corpus filename, baseline filename, results csv)
EVALS = {
    "external": ("corpus.eval.txt", "baseline.external.kld", "results.csv"),
    "general": ("corpus.eval.general.txt", "baseline.general.kld", "results.general.csv"),
    "tools": ("corpus.eval.tools.txt", "baseline.tools.kld", "results.tools.csv"),
    "agentic": ("corpus.eval.agentic.txt", "baseline.agentic.kld", "results.agentic.csv"),
    "broad": ("corpus.eval.broad.txt", "baseline.broad.kld", "results.broad.csv"),
}


@dataclass(frozen=True)
class Row:
    quant: str
    method: str = "imatrix"   # "imatrix" | "plain"

    def gguf_name(self, stem: str) -> str:
        # The quant tag must be the FILENAME'S TERMINAL TOKEN or Hugging Face won't derive
        # an Ollama `:QUANT` tag from it (see the HF↔Ollama tag convention).
        return f"{stem}-{self.quant}.gguf"


def _csv_has(path: Path, needle: str) -> bool:
    return path.exists() and any(needle in ln for ln in path.open())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="label only, for the CSV")
    p.add_argument("--stem", default=None,
                   help="GGUF filename stem (default: derived from --model-id)")
    p.add_argument("--rows", nargs="+", default=list(DEFAULT_ROWS))
    p.add_argument("--plain-anchor", action="store_true",
                   help="also build an un-calibrated Q2_K, the no-imatrix control")
    p.add_argument("--evals", nargs="+", default=["external", "general"],
                   choices=sorted(EVALS))
    p.add_argument("--wiki", type=Path, default=universal.DEFAULT_WIKI)
    p.add_argument("--logs-jsonl", type=Path, nargs="+",
                   default=list(universal.DEFAULT_LOG_FILES),
                   help="on-disk log JSONL(.gz) files (default: both corpora)")
    p.add_argument("--imatrix-variant", default="hybrid_custom")
    p.add_argument("--no-mtp-pin", action="store_true",
                   help="quantize the draft layer along with the trunk (don't; this exists "
                        "for A/B measuring what the Q8 pin buys)")
    p.add_argument("--suite", default="kld", choices=["quick", "kld", "speed", "full"])
    a = p.parse_args()

    root = REPO / "out" / a.run
    hf_dir = root / "model_extracted"
    f16 = root / "model-f16.gguf"
    logs = root / "logs"
    corpora = root / "corpora"
    stem = a.stem or a.model_id.split("/")[-1]

    for path in (hf_dir / "config.json", f16):
        if not path.exists():
            raise FileNotFoundError(f"missing input (run exp060_setup_qwen38.py): {path}")
    logs.mkdir(parents=True, exist_ok=True)

    rows = [Row(q) for q in a.rows]
    if a.plain_anchor:
        rows.insert(0, Row("Q2_K", "plain"))

    # --- MTP pin, read from the GGUF -------------------------------------------------
    report_path = root / "mtp_report.json"
    info = json.loads(report_path.read_text()) if report_path.exists() else mtp.describe(f16)
    pin: dict[str, str] | None = None if a.no_mtp_pin else (info.get("pin") or None)
    if pin:
        log(f"[{a.run}] MTP draft layer(s) {info['mtp_layer_indices']} pinned Q8_0: {pin}")
    else:
        log(f"[{a.run}] no MTP draft layer in the F16 GGUF — quantizing trunk only. "
            "Speculative decoding will need a grafted head or a separate draft model.")

    # --- stage 2: corpora + imatrix + baselines ---------------------------------------
    with phase(f"[{a.run}] build universal corpora"):
        step("corpora", corpora / "corpora_audit.json",
             lambda: universal.build(universal.UniversalConfig(
                 out_dir=corpora, model_dir=hf_dir,
                 log_files=tuple(a.logs_jsonl), wiki=a.wiki,
                 per_session_cap=CTX - 596,   # < imatrix ctx: no window straddles a chunk
             )))
    audit = json.loads((corpora / "corpora_audit.json").read_text())
    cal = audit["calibration"]
    log(f"[{a.run}] calibration corpus: {cal['total_tokens']:,} tokens, shares "
        f"{cal['token_share']}, tool-call markers "
        f"{cal['tool_calls']['tool_call_marker_total']:,}")

    base_imatrix = root / "imatrix-base.gguf"
    var_imatrix = root / f"imatrix-{a.imatrix_variant}.gguf"
    with phase(f"[{a.run}] base imatrix (ctx={CTX})"):
        step("imatrix-base", base_imatrix,
             lambda: llama_cpp.imatrix(f16, corpora / "corpus.cal.txt", base_imatrix,
                                       ctx=CTX, log=logs / "imatrix-base.log"))
    with phase(f"[{a.run}] re-weight imatrix -> {a.imatrix_variant}"):
        step("imatrix-variant", var_imatrix,
             lambda: imatrix_cal.calibrate(
                 variant=a.imatrix_variant, f16_gguf=f16,
                 base_imatrix=base_imatrix, out_path=var_imatrix))

    n_params = bpw_mod.n_params(f16)
    log(f"[{a.run}] reference n_params (incl. draft head) = {n_params:,.0f}")

    baselines: dict[str, tuple[Path, Path]] = {}
    for name in a.evals:
        corpus_name, baseline_name, _ = EVALS[name]
        corpus = corpora / corpus_name
        if not corpus.exists():
            log(f"  eval[{name}]: {corpus} not built — skipping")
            continue
        baseline = root / baseline_name
        with phase(f"[{a.run}] F16 baseline KLD on {name}"):
            step(f"baseline-{name}", baseline,
                 lambda c=corpus, b=baseline, n=name: kld.build_baseline(
                     f16, c, b, ctx=EVAL_CTX, log=logs / f"baseline-{n}.log"))
        baselines[name] = (corpus, baseline)

    # --- stage 3: quantize + bench ----------------------------------------------------
    for row in rows:
        sub = root / row.quant.lower()
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        qpath = sub / row.gguf_name(stem)
        imat = None if row.method == "plain" else var_imatrix
        tag = f"[{a.run}][{row.quant}]"

        with phase(f"{tag} quantize ({row.method}"
                   f"{', nextn@Q8' if pin else ''})"):
            step("quantize", qpath,
                 lambda q=qpath, qt=row.quant, im=imat, ld=log_dir: gguf.quantize(
                     f16, q, qt, imatrix=im, tensor_types=pin,
                     log=ld / "quantize.log"))

        for name, (corpus, baseline) in baselines.items():
            csv_path = root / EVALS[name][2]
            with phase(f"{tag} bench [{name}]"):
                if _csv_has(csv_path, str(qpath)):
                    log(f"  {qpath.name}: already in {csv_path.name} — skipping")
                    continue
                label = (f"{a.model_id}|{row.quant}|{row.method}"
                         f"{'+nextn@Q8' if pin else ''} // eval={name} // baseline=FP16")
                r = runner.bench_one(
                    qpath, label, reference_n_params=n_params,
                    eval_dataset=corpus, eval_baseline=baseline,
                    eval_ctx=EVAL_CTX, log_dir=log_dir, suite=a.suite,
                )
                runner.append_row(csv_path, r)
                log(f"  {qpath.name} [{name}]: size={r.size_gib:.2f}GiB bpw={r.bpw:.3f} "
                    f"PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")

    log("")
    log(f"=== {a.run} stage 2+3 complete ===")
    for name in baselines:
        log(f"  results[{name}]: {root / EVALS[name][2]}")
    log("  Next: bench_mtp_speed.py (draft acceptance), run_swebench_eval.py (agentic "
        "pass rate), then exp060_prepare_release.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

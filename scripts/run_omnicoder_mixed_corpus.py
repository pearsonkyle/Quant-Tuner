"""Add 'mixed' calibration corpus = wiki.test.raw + ~200k tokens of logtrain.

Produces:
  * corpus.mixed.txt              — wiki + stratified logtrain sample
  * imatrix-mixed.gguf            — stock llama-imatrix on mixed corpus
  * imatrix-hybrid_mixed.gguf     — hybrid_custom variant
  * Q4_K_M-stock_mixed.gguf       — Q4_K_M with the stock mixed imatrix
  * Q4_K_M-hybrid_mixed.gguf      — Q4_K_M with hybrid+mixed
  * 2 new rows in results.csv: stock/Q4_K_M-mixed, hybrid/Q4_K_M-mixed

Reuses the train split (seed=42, train_frac=0.8) so the logtrain portion is
disjoint from the KLD/tool-call eval splits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import imatrix
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import gguf

WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"
LOGTRAIN = REPO / "logtrain.jsonl"

F16 = WORK / "model-f16.gguf"
MODEL_DIR = WORK / "model_extracted"
WIKI = WORK / "wiki.test.raw"

LOGTRAIN_SAMPLE = WORK / "corpus.logtrain_sample.txt"  # 200k-token slice
MIXED_CORPUS = WORK / "corpus.mixed.txt"

BASE_IMATRIX_MIXED = WORK / "imatrix-mixed.gguf"
HYBRID_IMATRIX_MIXED = WORK / "imatrix-hybrid_mixed.gguf"

QUANT_STOCK_MIXED = WORK / "Q4_K_M-stock_mixed.gguf"
QUANT_HYBRID_MIXED = WORK / "Q4_K_M-hybrid_mixed.gguf"

EVAL_CORPUS = WORK / "corpus.eval.txt"
EVAL_BASELINE = WORK / "baseline.kld"
RESULTS = WORK / "results.csv"

LOGTRAIN_TARGET_TOKENS = 200_000


def _build_mixed_corpus() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    chunks, _kept, total, audit = split.stratified_pack(
        splits["train"], tok,
        target_tokens=LOGTRAIN_TARGET_TOKENS,
        per_session_cap=6_000, seed=42,
    )
    split.write_corpus(chunks, LOGTRAIN_SAMPLE)
    log(f"  logtrain sample: {total:,} tokens, {len(chunks)} sessions")
    log(f"  by source: {audit['tokens_per_source']}")

    # Wiki first (general English ground), then logtrain (domain).
    wiki_text = WIKI.read_text()
    sep = "" if wiki_text.endswith("\n") else "\n"
    MIXED_CORPUS.write_text(wiki_text + sep + "\n\n" + LOGTRAIN_SAMPLE.read_text())
    log(f"  mixed corpus: {MIXED_CORPUS.stat().st_size / 1e6:.1f} MB at {MIXED_CORPUS}")

    (WORK / "mixed_audit.json").write_text(
        json.dumps({"logtrain_audit": audit,
                    "wiki_bytes": WIKI.stat().st_size,
                    "logtrain_bytes": LOGTRAIN_SAMPLE.stat().st_size},
                   indent=2, default=str)
    )


def main() -> int:
    assert F16.exists(), F16
    assert WIKI.exists(), WIKI

    step("build mixed corpus (wiki + 200k logtrain)", MIXED_CORPUS, _build_mixed_corpus)

    step("llama-imatrix on mixed corpus", BASE_IMATRIX_MIXED,
         lambda: llama_cpp.imatrix(F16, MIXED_CORPUS, BASE_IMATRIX_MIXED,
                                   ctx=512, log=LOGS / "imatrix-mixed.log"))

    step("build hybrid_custom imatrix from mixed E[a^2]", HYBRID_IMATRIX_MIXED,
         lambda: imatrix.calibrate(
             variant="hybrid_custom", f16_gguf=F16,
             base_imatrix=BASE_IMATRIX_MIXED, out_path=HYBRID_IMATRIX_MIXED))

    for qname, src_imatrix, qpath in [
        ("stock_mixed",  BASE_IMATRIX_MIXED,   QUANT_STOCK_MIXED),
        ("hybrid_mixed", HYBRID_IMATRIX_MIXED, QUANT_HYBRID_MIXED),
    ]:
        step(f"quantize Q4_K_M ({qname})", qpath,
             lambda q=qpath, s=src_imatrix, n=qname: gguf.quantize(
                 F16, q, "Q4_K_M", imatrix=s,
                 log=LOGS / f"quantize-{n}.log"))

    # 5) Bench the two new rows (full suite = KLD + speed).
    n_params = bpw_mod.n_params(F16)
    for label, quant in [
        ("stock/Q4_K_M-mixed",   QUANT_STOCK_MIXED),
        ("hybrid/Q4_K_M-mixed",  QUANT_HYBRID_MIXED),
    ]:
        with phase(f"bench {label}"):
            row = runner.bench_one(
                quant, label,
                reference_n_params=n_params,
                eval_dataset=EVAL_CORPUS,
                eval_baseline=EVAL_BASELINE,
                eval_ctx=8192,
                log_dir=LOGS,
                suite="full",
                bench_repetitions=10,
            )
            runner.append_row(RESULTS, row)
            log(f"  ppl={row.ppl} mean_kld={row.mean_kld} same_top_p={row.same_top_p} "
                f"decode={row.decode_tok_s} ± {row.decode_stdev}")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

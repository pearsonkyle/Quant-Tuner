"""Driver: 3-row Q4_K_M leaderboard slice for Tesslate/OmniCoder-9B.

Rows produced:
  * baseline/fp16             — direct bench of the F16 GGUF
  * baseline/Q4_K_M-none      — llama-quantize without an imatrix
  * imatrix/Q4_K_M-hybrid     — output-aware imatrix (hybrid_custom variant)

Idempotent: each phase checks for its output before running, so re-launches
resume cleanly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import imatrix
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.leaderboard import aggregate
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"
SRC_MODEL = "Tesslate/OmniCoder-9B"
LOGTRAIN = REPO / "logtrain.jsonl"

MODEL_DIR = WORK / "model_extracted"
F16_GGUF = WORK / "model-f16.gguf"
CORPUS_TRAIN = WORK / "corpus.train.txt"
CORPUS_EVAL = WORK / "corpus.eval.txt"
BASE_IMATRIX = WORK / "imatrix-custom.gguf"
HYBRID_IMATRIX = WORK / "imatrix-hybrid_custom.gguf"
EVAL_BASELINE = WORK / "baseline.kld"
RESULTS = WORK / "results.csv"
LEADERBOARD = WORK / "LEADERBOARD.md"

QUANT_NONE = WORK / "Q4_K_M-none.gguf"
QUANT_IMATRIX = WORK / "Q4_K_M-hybrid_custom.gguf"


def _prepare_corpora() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    log(f"  {len(sessions)} sessions after filtering")

    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    log(
        f"  split: train={len(splits['train'])} "
        f"test={len(splits['test'])} holdout={len(splits['holdout'])}"
    )

    # Train corpus → 500k tokens for calibration.
    chunks, _kept, total, audit = split.stratified_pack(
        splits["train"], tok, target_tokens=500_000, per_session_cap=6_000, seed=42
    )
    split.write_corpus(chunks, CORPUS_TRAIN)
    (WORK / "train_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    log(f"  train corpus: {total:,} tokens -> {CORPUS_TRAIN.name}")

    # Eval corpus → 50k tokens (PPL/KLD eval).
    ev_chunks, _ek, ev_total, ev_audit = split.stratified_pack(
        splits["test"], tok, target_tokens=50_000, per_session_cap=4_000, seed=43
    )
    split.write_corpus(ev_chunks, CORPUS_EVAL)
    (WORK / "eval_audit.json").write_text(json.dumps(ev_audit, indent=2, default=str))
    log(f"  eval corpus:  {ev_total:,} tokens -> {CORPUS_EVAL.name}")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    step("extract OmniCoder-9B text-only LM", MODEL_DIR / "config.json",
         lambda: extract.extract_text_lm(
             source=SRC_MODEL, output_dir=MODEL_DIR,
             causal_lm_arch="Qwen3_5ForCausalLM"))

    step("convert HF -> F16 GGUF", F16_GGUF,
         lambda: convert.hf_to_f16_gguf(MODEL_DIR, F16_GGUF, log=LOGS / "convert.log"))

    step("prepare calibration corpus + eval split",
         [CORPUS_TRAIN, CORPUS_EVAL], _prepare_corpora)

    step("llama-imatrix (standard E[a^2])", BASE_IMATRIX,
         lambda: llama_cpp.imatrix(F16_GGUF, CORPUS_TRAIN, BASE_IMATRIX,
                                   ctx=512, log=LOGS / "imatrix-custom.log"))

    step("build output-aware imatrix (hybrid_custom)", HYBRID_IMATRIX,
         lambda: imatrix.calibrate(
             variant="hybrid_custom", f16_gguf=F16_GGUF,
             base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))

    step("build F16 KLD baseline", EVAL_BASELINE,
         lambda: kld.build_baseline(F16_GGUF, CORPUS_EVAL, EVAL_BASELINE,
                                    ctx=8192, log=LOGS / "baseline.log"))

    step("quantize Q4_K_M (no imatrix)", QUANT_NONE,
         lambda: gguf.quantize(F16_GGUF, QUANT_NONE, "Q4_K_M",
                               log=LOGS / "quantize-none.log"))

    step("quantize Q4_K_M (hybrid_custom imatrix)", QUANT_IMATRIX,
         lambda: gguf.quantize(F16_GGUF, QUANT_IMATRIX, "Q4_K_M",
                               imatrix=HYBRID_IMATRIX,
                               log=LOGS / "quantize-imatrix.log"))

    # 8) Bench all three rows into results.csv.
    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"n_params = {n_params:,}")
    for label, quant in [
        ("baseline/fp16",            F16_GGUF),
        ("baseline/Q4_K_M-none",     QUANT_NONE),
        ("imatrix/Q4_K_M-hybrid",    QUANT_IMATRIX),
    ]:
        with phase(f"bench {label}"):
            row = runner.bench_one(
                quant, label,
                reference_n_params=n_params,
                eval_dataset=CORPUS_EVAL,
                eval_baseline=EVAL_BASELINE,
                eval_ctx=8192,
                log_dir=LOGS,
                suite="full",
            )
            runner.append_row(RESULTS, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p} decode={row.decode_tok_s}")

    # 9) Aggregate leaderboard.
    with phase("aggregate leaderboard"):
        md = aggregate.aggregate(RESULTS, weights=(1.0, 2.0, 1.0))
        LEADERBOARD.write_text(md)
        log(f"  wrote {LEADERBOARD}")
        print()
        print(md)

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

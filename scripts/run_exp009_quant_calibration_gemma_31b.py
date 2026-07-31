"""Experiment 009: Q5_K_S / Q3_K_S / IQ2_XXS sweep on google/gemma-4-31B-it.

Single-model, single-corpus run. The mixed8k corpus (500k custom tokens
from logtrain.jsonl + calibration_supplement.txt followed by the full
wiki.test.raw) is built once, fed to `llama-imatrix` at ctx=8192, and
reused across the three quant targets:

  * Q5_K_S
  * Q3_K_S
  * IQ2_XXS

F16 acts as the KLD reference. All KLD/PPL numbers are produced at
ctx=4096 (gemma's ~262k vocab busts llama-perplexity at ctx=8192).

Idempotent: every stage is wrapped in `experiments.step()`.

Note: 31B at FP16 is ~62 GiB on disk; the imatrix pass at ctx=8192 is
correspondingly heavy. Run on a host with enough VRAM/RAM + free disk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

REPO_ID = "google/gemma-4-31B-it"
SLUG = REPO_ID.replace("/", "__")
EXP_ROOT = REPO / "out" / "exp-009"
WORK = EXP_ROOT / SLUG
LOGS = WORK / "logs"

LOGTRAIN = REPO / "logtrain.jsonl"
SUPPLEMENT = REPO / "calibration_supplement.txt"
WIKI_LOCAL = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"

EVAL_CTX = 4096          # gemma huge-vocab cap
IMATRIX_CTX = 8192       # mixed8k

QUANTS = ["Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "IQ4_XS", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ2_M", "IQ2_XXS"]
DATASET_LABEL = "500k-custom+wiki (ctx=8192)"


def _prepare_corpora(model_dir: Path, train_out: Path, eval_out: Path) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    log(f"  {len(sessions)} sessions after filtering")

    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    log(f"  split: train={len(splits['train'])} "
        f"test={len(splits['test'])} holdout={len(splits['holdout'])}")

    train_chunks, _k, train_total, train_audit = split.stratified_pack(
        splits["train"], tok, target_tokens=500_000, per_session_cap=6_000, seed=42
    )
    supplement = SUPPLEMENT if SUPPLEMENT.exists() else None
    if supplement is None:
        log(f"  WARNING: {SUPPLEMENT} not found — train corpus will be logs-only")
    split.write_corpus(train_chunks, train_out, supplement=supplement)
    (train_out.parent / "train_audit.json").write_text(
        json.dumps(train_audit, indent=2, default=str)
    )
    log(f"  train corpus: {train_total:,} tokens (+ supplement) -> {train_out.name}")

    eval_chunks, _ek, eval_total, eval_audit = split.stratified_pack(
        splits["holdout"], tok, target_tokens=50_000, per_session_cap=4_000, seed=43
    )
    split.write_corpus(eval_chunks, eval_out)
    (eval_out.parent / "eval_audit.json").write_text(
        json.dumps(eval_audit, indent=2, default=str)
    )
    log(f"  eval corpus:  {eval_total:,} tokens -> {eval_out.name}")


def _prepare_mixed_corpus(model_dir: Path, mixed_out: Path) -> None:
    from transformers import AutoTokenizer

    if not WIKI_LOCAL.exists():
        raise FileNotFoundError(
            f"{WIKI_LOCAL} not staged — populate it from $WIKI_TEST_RAW first."
        )

    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    train_chunks, _k, train_total, _audit = split.stratified_pack(
        splits["train"], tok, target_tokens=500_000, per_session_cap=6_000, seed=42
    )

    mixed_out.parent.mkdir(parents=True, exist_ok=True)
    with open(mixed_out, "w") as f:
        for chunk in train_chunks:
            f.write(chunk + "\n\n")
        wiki_text = WIKI_LOCAL.read_text()
        f.write(wiki_text)
        if not wiki_text.endswith("\n"):
            f.write("\n")
    log(f"  mixed corpus: {train_total:,} custom tokens + full wiki -> {mixed_out.name}")


def _stage_wiki() -> None:
    if WIKI_LOCAL.exists():
        return
    src_env = os.environ.get("WIKI_TEST_RAW")
    if not src_env:
        raise FileNotFoundError(
            "wiki.test.raw not found. Set $WIKI_TEST_RAW to a local copy "
            "(wikitext-2-raw-v1)."
        )
    src = Path(src_env)
    if not src.exists():
        raise FileNotFoundError(f"$WIKI_TEST_RAW points at missing file: {src}")
    WIKI_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, WIKI_LOCAL)
    log(f"  copied {src} -> {WIKI_LOCAL}")


def _f16_ppl_from_baseline_log(log_file: Path) -> float | None:
    if not log_file.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log_file.read_text())
    return float(m.group(1)) if m else None


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    model_dir = WORK / "model_extracted"
    f16 = WORK / "model-f16.gguf"
    train = WORK / "corpus.train.txt"
    eval_ds = WORK / "corpus.eval.txt"
    mixed = WORK / "corpus.mixed8k.txt"
    imat = WORK / "imatrix-mixed8k.gguf"
    base_kld = WORK / "baseline.kld"
    per_model_csv = WORK / "results.csv"

    with phase("stage wiki.test.raw"):
        _stage_wiki()

    with phase(f"model {REPO_ID}"):
        step(f"extract {REPO_ID}", model_dir / "config.json",
             lambda: extract.extract_text_lm(
                 source=REPO_ID, output_dir=model_dir, causal_lm_arch=None))

        step("convert HF -> F16 GGUF", f16,
             lambda: convert.hf_to_f16_gguf(model_dir, f16, log=LOGS / "convert.log"))

        step("prepare custom + eval corpora", [train, eval_ds],
             lambda: _prepare_corpora(model_dir, train, eval_ds))

        step("prepare mixed8k corpus (500k custom + full wiki)", mixed,
             lambda: _prepare_mixed_corpus(model_dir, mixed))

        step(f"llama-imatrix mixed8k (ctx={IMATRIX_CTX})", imat,
             lambda: llama_cpp.imatrix(f16, mixed, imat,
                                       ctx=IMATRIX_CTX,
                                       log=LOGS / "imatrix-mixed8k.log"))

        step("build F16 KLD baseline", base_kld,
             lambda: kld.build_baseline(f16, eval_ds, base_kld,
                                        ctx=EVAL_CTX, log=LOGS / "baseline.log"))

        n_params = bpw_mod.n_params(f16)
        log(f"f16 n_params = {n_params:,.0f}")

        rows_by_quant: dict[str, runner.BenchRow] = {}
        for q in QUANTS:
            qpath = WORK / f"{q}-mixed8k.gguf"
            with phase(f"{q} mixed8k"):
                step(f"quantize {q}", qpath,
                     lambda p=qpath, qt=q: gguf.quantize(
                         f16, p, qt, imatrix=imat,
                         log=LOGS / f"quantize-{qt}.log"))
                label = f"{REPO_ID}|{q}|imatrix|{DATASET_LABEL}"
                with phase(f"bench {label}"):
                    row = runner.bench_one(
                        qpath, label,
                        reference_n_params=n_params,
                        eval_dataset=eval_ds,
                        eval_baseline=base_kld,
                        eval_ctx=EVAL_CTX,
                        log_dir=LOGS,
                        suite="kld",
                    )
                    runner.append_row(per_model_csv, row)
                    rows_by_quant[q] = row
                    log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                        f"ppl={row.ppl} mean_kld={row.mean_kld} "
                        f"same_top_p={row.same_top_p}")

    # Render markdown table
    fp16_ppl = _f16_ppl_from_baseline_log(LOGS / "baseline.log")
    fp16_size_gib = f16.stat().st_size / (1024 ** 3)
    fp16_bpw = (f16.stat().st_size * 8) / bpw_mod.n_params(f16)

    def _fmt(v, places: int) -> str:
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return "—"

    lines = [
        "| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|---|",
        f"| FP16    | none    | —             | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"{_fmt(fp16_ppl, 4)} | 0.00000 | 100.0000 |",
    ]
    for q in QUANTS:
        r = rows_by_quant.get(q)
        if r is None:
            lines.append(f"| {q} | imatrix | {DATASET_LABEL} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {q} | imatrix | {DATASET_LABEL} | "
                f"{_fmt(r.size_gib, 2)} | {_fmt(r.bpw, 3)} | "
                f"{_fmt(r.ppl, 4)} | {_fmt(r.mean_kld, 5)} | "
                f"{_fmt(r.same_top_p, 4)} |"
            )

    md_path = WORK / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

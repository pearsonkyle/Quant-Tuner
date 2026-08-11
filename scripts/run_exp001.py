"""Experiment 001: Q4_K_M imatrix dataset comparison across 3 Qwen3.5-9B models.

For each model in {Qwen/Qwen3.5-9B, Tesslate/OmniCoder-9B,
Jackrong/Qwopus3.5-9B-Coder}, build three Q4_K_M GGUFs and bench them
against the same held-out eval set:

  * custom  — llama-imatrix on logtrain.jsonl + calibration_supplement.txt
  * wiki    — llama-imatrix on wiki.test.raw (wikitext-2-raw-v1)
  * none    — llama-quantize with no imatrix

Re-weighting is vanilla llama.cpp (no `calibrate.imatrix` pass). That's
the variable held fixed for this experiment; exp-002 varies the variant
instead.

Idempotent — every stage is wrapped in `experiments.step()`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

EXP_ROOT = REPO / "out" / "exp-001"
LOGTRAIN = REPO / "logtrain.jsonl"
SUPPLEMENT = REPO / "calibration_supplement.txt"
WIKI_LOCAL = EXP_ROOT / "wiki" / "wiki.test.raw"
AGGREGATE_CSV = EXP_ROOT / "results.csv"

CELLS = ("custom", "wiki", "none", "mixed8k")


@dataclass(frozen=True)
class ModelCfg:
    repo_id: str
    causal_lm_arch: str | None = None  # passed through to extract_text_lm
    eval_ctx: int = 8192  # KLD baseline + bench context; lowered for huge-vocab models

    @property
    def slug(self) -> str:
        return self.repo_id.replace("/", "__")


MODELS: list[ModelCfg] = [
    ModelCfg("Qwen/Qwen3.5-9B"),
    ModelCfg("Tesslate/OmniCoder-9B", causal_lm_arch="Qwen3_5ForCausalLM"),
    ModelCfg("Jackrong/Qwopus3.5-9B-Coder"),
    # Gemma's ~262k vocab busts llama-perplexity's logit-vector allocation at ctx=8192.
    ModelCfg("google/gemma-4-E4B-it", eval_ctx=4096),
]


# ---------- corpus prep (custom = logtrain.jsonl + supplement, eval = holdout slice)

def _prepare_corpora(model_dir: Path, train_out: Path, eval_out: Path) -> None:
    """Build the custom calibration corpus and the eval (holdout) corpus.

    Matches the OmniCoder script's split shape (80/10/10, seed=42) so the
    holdout invariant from CLAUDE.md is preserved. Appends calibration_supplement.txt
    to the train corpus.
    """
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


# ---------- merged corpus (500k custom + full wiki) for the mixed8k cell

def _prepare_mixed_corpus(model_dir: Path, mixed_out: Path) -> None:
    """500k tokens from the custom train slice followed by the full wiki.test.raw.

    Uses the same split/seed as `_prepare_corpora` so the 500k-token slice is
    identical to what the `custom` cell sees, then appends wiki verbatim.
    """
    from transformers import AutoTokenizer

    if not WIKI_LOCAL.exists():
        raise FileNotFoundError(
            f"{WIKI_LOCAL} not staged — run with `--only mixed8k` after wiki staging, "
            "or include the wiki cell on the first run."
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


# ---------- wiki.test.raw staging (one-time, shared across models)

def _stage_wiki() -> None:
    if WIKI_LOCAL.exists():
        return
    src_env = os.environ.get("WIKI_TEST_RAW")
    if not src_env:
        raise FileNotFoundError(
            "wiki.test.raw not found. Set $WIKI_TEST_RAW to a local copy "
            "(wikitext-2-raw-v1 from "
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip)."
        )
    src = Path(src_env)
    if not src.exists():
        raise FileNotFoundError(f"$WIKI_TEST_RAW points at missing file: {src}")
    WIKI_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, WIKI_LOCAL)
    log(f"  copied {src} -> {WIKI_LOCAL}")


# ---------- per-model pipeline

def _run_model(cfg: ModelCfg, cells: tuple[str, ...], *, dry_run: bool) -> None:
    work = EXP_ROOT / cfg.slug
    logs = work / "logs"
    model_dir = work / "model_extracted"
    f16 = work / "model-f16.gguf"
    train = work / "corpus.train.txt"
    eval_ds = work / "corpus.eval.txt"
    imat_custom = work / "imatrix-custom.gguf"
    imat_wiki = work / "imatrix-wiki.gguf"
    imat_mixed = work / "imatrix-mixed8k.gguf"
    mixed_corpus = work / "corpus.mixed8k.txt"
    base_kld = work / "baseline.kld"
    quants = {
        "custom":   work / "Q4_K_M-custom.gguf",
        "wiki":     work / "Q4_K_M-wiki.gguf",
        "none":     work / "Q4_K_M-none.gguf",
        "mixed8k": work / "Q4_K_M-mixed8k.gguf",
    }
    per_model_csv = work / "results.csv"

    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    with phase(f"model {cfg.repo_id}"):
        if dry_run:
            log("  [dry-run] would extract → convert → corpora → imatrices → baseline → quants → bench")
            for cell in cells:
                log(f"  [dry-run] cell={cell} -> {quants[cell].relative_to(REPO)}")
            return

        step(f"extract {cfg.repo_id}", model_dir / "config.json",
             lambda: extract.extract_text_lm(
                 source=cfg.repo_id, output_dir=model_dir,
                 causal_lm_arch=cfg.causal_lm_arch))

        step("convert HF -> F16 GGUF", f16,
             lambda: convert.hf_to_f16_gguf(model_dir, f16, log=logs / "convert.log"))

        need_custom_corpus = "custom" in cells
        need_eval = bool(cells)  # any cell needs the eval set for bench

        if need_custom_corpus or need_eval:
            step("prepare custom + eval corpora", [train, eval_ds],
                 lambda: _prepare_corpora(model_dir, train, eval_ds))

        if "custom" in cells:
            step("llama-imatrix on custom corpus", imat_custom,
                 lambda: llama_cpp.imatrix(f16, train, imat_custom,
                                           ctx=512, log=logs / "imatrix-custom.log"))

        if "wiki" in cells:
            step("llama-imatrix on wiki.test.raw", imat_wiki,
                 lambda: llama_cpp.imatrix(f16, WIKI_LOCAL, imat_wiki,
                                           ctx=512, log=logs / "imatrix-wiki.log"))

        if "mixed8k" in cells:
            step("prepare mixed8k corpus (500k custom + full wiki)", mixed_corpus,
                 lambda: _prepare_mixed_corpus(model_dir, mixed_corpus))
            step("llama-imatrix mixed8k (ctx=8192)", imat_mixed,
                 lambda: llama_cpp.imatrix(f16, mixed_corpus, imat_mixed,
                                           ctx=8192, log=logs / "imatrix-mixed8k.log"))

        step("build F16 KLD baseline", base_kld,
             lambda: kld.build_baseline(f16, eval_ds, base_kld,
                                        ctx=cfg.eval_ctx, log=logs / "baseline.log"))

        for cell in cells:
            qpath = quants[cell]
            if cell == "none":
                step(f"quantize Q4_K_M ({cell})", qpath,
                     lambda q=qpath, c=cell: gguf.quantize(
                         f16, q, "Q4_K_M", log=logs / f"quantize-{c}.log"))
            else:
                src_imat = {
                    "custom": imat_custom,
                    "wiki": imat_wiki,
                    "mixed8k": imat_mixed,
                }[cell]
                step(f"quantize Q4_K_M ({cell})", qpath,
                     lambda q=qpath, c=cell, s=src_imat: gguf.quantize(
                         f16, q, "Q4_K_M", imatrix=s,
                         log=logs / f"quantize-{c}.log"))

        n_params = bpw_mod.n_params(f16)
        for cell in cells:
            qpath = quants[cell]
            technique = "imatrix" if cell != "none" else "none"
            dataset = {
                "custom": "custom",
                "wiki": "wiki.test.raw",
                "none": "—",
                "mixed8k": "500k-custom+wiki (ctx=8192)",
            }[cell]
            label = f"{cfg.repo_id}|{technique}|{dataset}"
            with phase(f"bench {label}"):
                row = runner.bench_one(
                    qpath, label,
                    reference_n_params=n_params,
                    eval_dataset=eval_ds,
                    eval_baseline=base_kld,
                    eval_ctx=cfg.eval_ctx,
                    log_dir=logs,
                    suite="kld",
                )
                runner.append_row(per_model_csv, row)
                runner.append_row(AGGREGATE_CSV, row)
                log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                    f"ppl={row.ppl} mean_kld={row.mean_kld} "
                    f"same_top_p={row.same_top_p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None,
                    help="HF repo ids to run (default: all three).")
    ap.add_argument("--only", choices=CELLS, default=None,
                    help="Only run this cell type across the selected models.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned work without invoking llama.cpp.")
    args = ap.parse_args()

    selected = MODELS
    if args.models:
        wanted = set(args.models)
        selected = [m for m in MODELS if m.repo_id in wanted]
        missing = wanted - {m.repo_id for m in selected}
        if missing:
            print(f"unknown model(s): {sorted(missing)}", file=sys.stderr)
            return 2
    cells = (args.only,) if args.only else CELLS

    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    if ("wiki" in cells or "mixed8k" in cells) and not args.dry_run:
        with phase("stage wiki.test.raw"):
            _stage_wiki()

    for cfg in selected:
        _run_model(cfg, cells, dry_run=args.dry_run)

    log("ALL DONE")
    log(f"aggregate CSV: {AGGREGATE_CSV.relative_to(REPO)}")
    log("render table:  PYTHONPATH=src .venv/bin/python scripts/render_exp001_table.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

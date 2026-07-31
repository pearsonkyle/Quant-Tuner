"""Fresh Jackrong/Qwopus3.5-9B-Coder calibration sweep with DB integration.

Produces 5 GGUFs (FP16 reference + 4 Q4_K_M cells) and writes everything to the
SQLite quant DB:

  * FP16              — reference
  * Q4_K_M  none      — no imatrix
  * Q4_K_M  wiki:512  — imatrix on wiki.test.raw at ctx=512
  * Q4_K_M  mixed:512 — imatrix on 500k-custom + full wiki at ctx=512
  * Q4_K_M  mixed:4096 — same corpus at ctx=4096

Pipeline per cell:
    HF → F16 GGUF → corpora → imatrix → quantize → bench (KLD/PPL/speed)
        → MMLU-Pro reps + tool-call reps → DB upsert

Sampling for task evals (user-supplied):
    T=0.5  top_p=0.95  top_k=20  min_p=0.0  presence_penalty=0.0  rep_penalty=1.0

Everything lands under ``out/jackrong-fresh/``. Idempotent — re-running skips
stages whose outputs already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.data import ingest, split
from quant_tuner.db import io as db_io
from quant_tuner.db.engine import get_engine, init_db, session_scope
from quant_tuner.eval.mmlu_pro import run_mmlu_pro_eval
from quant_tuner.eval.reps import (
    RepResult,
    run_reps_for_models,
    sampling_extra_cols,
    write_csvs,
)
from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ID = "Jackrong/Qwopus3.5-9B-Coder"
WORK = REPO / "out" / "jackrong-fresh"
LOGS = WORK / "logs"
LOGTRAIN = REPO / "logtrain.jsonl"
SUPPLEMENT = REPO / "calibration_supplement.txt"
WIKI_LOCAL = WORK / "wiki" / "wiki.test.raw"

EVAL_CTX = 8192  # Qwen-class vocab fits fine

SAMPLING = Sampling(
    temperature=0.5,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    max_tokens=512,
)

MMLU_SAMPLING = Sampling(
    temperature=0.5,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    max_tokens=2048,
)


@dataclass(frozen=True)
class Cell:
    name: str          # short tag (none, wiki, mixed512, mixed4k)
    dataset_label: str # human-readable for the DB / labels
    technique: str     # imatrix | none
    calib_ctx: int | None


CELLS: list[Cell] = [
    Cell("none",     "—",                              "none",    None),
    Cell("wiki",     "wiki.test.raw",                  "imatrix", 512),
    Cell("mixed512", "500k-custom+wiki (ctx=512)",     "imatrix", 512),
    Cell("mixed4k",  "500k-custom+wiki (ctx=4096)",    "imatrix", 4096),
]


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def _prepare_corpora(model_dir: Path, train_out: Path, eval_out: Path) -> None:
    """Build the custom calibration train corpus + the eval holdout corpus."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    log(f"  {len(sessions)} sessions after filtering")

    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
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
    """500k-token slice of the custom train pack followed by full wiki.test.raw."""
    from transformers import AutoTokenizer

    if not WIKI_LOCAL.exists():
        raise FileNotFoundError(f"{WIKI_LOCAL} not staged — call _stage_wiki() first.")

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


# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------

def _paths():
    return {
        "model_dir":     WORK / "model_extracted",
        "f16":           WORK / "model-f16.gguf",
        "train":         WORK / "corpus.train.txt",
        "eval_ds":       WORK / "corpus.eval.txt",
        "mixed_corpus": WORK / "corpus.mixed.txt",
        "imat_wiki":     WORK / "imatrix-wiki.gguf",
        "imat_mixed512": WORK / "imatrix-mixed512.gguf",
        "imat_mixed4k":  WORK / "imatrix-mixed4k.gguf",
        "base_kld":      WORK / "baseline.kld",
        "q_none":        WORK / "Q4_K_M-none.gguf",
        "q_wiki":        WORK / "Q4_K_M-wiki.gguf",
        "q_mixed512":   WORK / "Q4_K_M-mixed512.gguf",
        "q_mixed4k":    WORK / "Q4_K_M-mixed4k.gguf",
    }


def _imatrix_for(cell: Cell, p: dict) -> Path | None:
    return {
        "none":     None,
        "wiki":     p["imat_wiki"],
        "mixed512": p["imat_mixed512"],
        "mixed4k":  p["imat_mixed4k"],
    }[cell.name]


def _quant_path_for(cell: Cell, p: dict) -> Path:
    return {
        "none":     p["q_none"],
        "wiki":     p["q_wiki"],
        "mixed512": p["q_mixed512"],
        "mixed4k":  p["q_mixed4k"],
    }[cell.name]


def build_all(dry_run: bool) -> dict:
    p = _paths()
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase(f"build {REPO_ID}"):
        if dry_run:
            log(f"  [dry-run] extract {REPO_ID}  -> {p['model_dir'].relative_to(REPO)}")
            log(f"  [dry-run] convert HF -> F16 -> {p['f16'].relative_to(REPO)}")
            log("  [dry-run] stage wiki.test.raw")
            log(f"  [dry-run] prepare corpora -> {p['train'].name}, {p['eval_ds'].name}")
            log(f"  [dry-run] prepare mixed corpus -> {p['mixed_corpus'].name}")
            log("  [dry-run] llama-imatrix wiki     (ctx=512)")
            log("  [dry-run] llama-imatrix mixed512 (ctx=512)")
            log("  [dry-run] llama-imatrix mixed4k  (ctx=4096)")
            log(f"  [dry-run] F16 KLD baseline (ctx={EVAL_CTX}) -> {p['base_kld'].name}")
            for cell in CELLS:
                log(f"  [dry-run] quantize Q4_K_M ({cell.name}) -> {_quant_path_for(cell, p).name}")
            return p

        step(f"extract {REPO_ID}", p["model_dir"] / "config.json",
             lambda: extract.extract_text_lm(source=REPO_ID, output_dir=p["model_dir"]))

        step("convert HF -> F16 GGUF", p["f16"],
             lambda: convert.hf_to_f16_gguf(p["model_dir"], p["f16"],
                                            log=LOGS / "convert.log"))

        with phase("stage wiki.test.raw"):
            _stage_wiki()

        step("prepare custom + eval corpora", [p["train"], p["eval_ds"]],
             lambda: _prepare_corpora(p["model_dir"], p["train"], p["eval_ds"]))

        step("prepare mixed corpus (500k custom + full wiki)", p["mixed_corpus"],
             lambda: _prepare_mixed_corpus(p["model_dir"], p["mixed_corpus"]))

        step("llama-imatrix wiki (ctx=512)", p["imat_wiki"],
             lambda: llama_cpp.imatrix(p["f16"], WIKI_LOCAL, p["imat_wiki"],
                                       ctx=512, log=LOGS / "imatrix-wiki.log"))

        step("llama-imatrix mixed (ctx=512)", p["imat_mixed512"],
             lambda: llama_cpp.imatrix(p["f16"], p["mixed_corpus"], p["imat_mixed512"],
                                       ctx=512, log=LOGS / "imatrix-mixed512.log"))

        step("llama-imatrix mixed (ctx=4096)", p["imat_mixed4k"],
             lambda: llama_cpp.imatrix(p["f16"], p["mixed_corpus"], p["imat_mixed4k"],
                                       ctx=4096, log=LOGS / "imatrix-mixed4k.log"))

        step("build F16 KLD baseline", p["base_kld"],
             lambda: kld.build_baseline(p["f16"], p["eval_ds"], p["base_kld"],
                                        ctx=EVAL_CTX, log=LOGS / "baseline.log"))

        for cell in CELLS:
            qpath = _quant_path_for(cell, p)
            imat = _imatrix_for(cell, p)
            if imat is None:
                step(f"quantize Q4_K_M ({cell.name})", qpath,
                     lambda q=qpath, c=cell: gguf.quantize(
                         p["f16"], q, "Q4_K_M",
                         log=LOGS / f"quantize-{c.name}.log"))
            else:
                step(f"quantize Q4_K_M ({cell.name})", qpath,
                     lambda q=qpath, c=cell, s=imat: gguf.quantize(
                         p["f16"], q, "Q4_K_M", imatrix=s,
                         log=LOGS / f"quantize-{c.name}.log"))

    return p


# ---------------------------------------------------------------------------
# Bench + DB
# ---------------------------------------------------------------------------

def bench_and_record(p: dict, dry_run: bool) -> None:
    """Bench every cell + FP16; write Quant rows to the DB."""
    csv_path = WORK / "results.csv"

    if dry_run:
        log(f"  [dry-run] bench FP16 reference -> {csv_path.name}")
        for cell in CELLS:
            log(f"  [dry-run] bench {cell.name} -> {csv_path.name}")
        log("  [dry-run] upsert 5 Quant rows into DB")
        return

    engine = get_engine()
    init_db(engine)
    n_params = bpw_mod.n_params(p["f16"])

    # FP16 reference row (size/bpw only; no KLD against self).
    f16_size_gib = p["f16"].stat().st_size / (1024 ** 3)
    f16_bpw = (p["f16"].stat().st_size * 8) / n_params
    with session_scope(engine) as s:
        db_io.upsert_quant(
            s,
            model_repo=REPO_ID,
            quant_type="F16",
            technique="none",
            size_gib=f16_size_gib,
            bpw=f16_bpw,
            n_params=int(n_params),
            quant_path=str(p["f16"]),
            f16_path=str(p["f16"]),
            workspace=str(WORK),
        )
    log(f"  FP16 recorded: {f16_size_gib:.2f} GiB  bpw={f16_bpw:.3f}")

    for cell in CELLS:
        qpath = _quant_path_for(cell, p)
        imat = _imatrix_for(cell, p)
        label = f"{REPO_ID}|{cell.technique}|{cell.dataset_label}"
        with phase(f"bench {label}"):
            row = runner.bench_one(
                qpath, label,
                reference_n_params=n_params,
                eval_dataset=p["eval_ds"],
                eval_baseline=p["base_kld"],
                eval_ctx=EVAL_CTX,
                log_dir=LOGS,
                suite="kld",
            )
            runner.append_row(csv_path, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p}")

        with session_scope(engine) as s:
            db_io.quant_from_bench_row(
                s, row,
                model_repo=REPO_ID,
                quant_type="Q4_K_M",
                technique=cell.technique,
                calib_ctx=cell.calib_ctx,
                dataset_name=(cell.dataset_label if cell.technique != "none" else None),
                n_params=int(n_params),
                f16_path=str(p["f16"]),
                imatrix_path=(str(imat) if imat else None),
                workspace=str(WORK),
            )


# ---------------------------------------------------------------------------
# Task evals
# ---------------------------------------------------------------------------

def _all_gguf_paths(p: dict) -> list[Path]:
    return [p["f16"]] + [_quant_path_for(c, p) for c in CELLS]


def _resolve_quants_by_path(session, paths: list[Path]):
    """Map quant_path basename -> Quant row (must exist post-bench)."""
    from sqlmodel import select
    from quant_tuner.db.models import Quant
    by_basename = {}
    rows = session.exec(select(Quant).where(Quant.model_repo == REPO_ID)).all()
    for q in rows:
        if q.quant_path:
            by_basename[Path(q.quant_path).name] = q
    return [by_basename.get(p.name) for p in paths]


def run_toolcall(p: dict, holdout: Path, reps: int, dry_run: bool) -> None:
    models = _all_gguf_paths(p)
    out_per_rep = WORK / "toolcall_reps_results.csv"
    out_agg = WORK / "toolcall_reps_aggregated.csv"
    log_dir = LOGS / "toolcall"

    if dry_run:
        log(f"  [dry-run] toolcall reps×{reps} for {len(models)} models")
        log(f"  [dry-run] sampling T={SAMPLING.temperature} top_p={SAMPLING.top_p} "
            f"top_k={SAMPLING.top_k} min_p={SAMPLING.min_p} pp={SAMPLING.presence_penalty} "
            f"rep_pen={SAMPLING.repetition_penalty}")
        log(f"  [dry-run] holdout: {holdout}")
        log(f"  [dry-run] writes: {out_per_rep.name}, {out_agg.name} + DB ToolcallRep rows")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    assert holdout.exists(), f"toolcall holdout missing: {holdout}"

    def eval_one(base_url: str, smp: Sampling, rep_idx: int) -> dict[str, float]:
        per_turn_log = log_dir / f"toolcall_rep{rep_idx:02d}_{int(time.time())}.jsonl"
        summary = run_toolcall_eval(
            holdout=holdout, base_url=base_url, sampling=smp,
            max_turns_per_session=10, per_turn_log=per_turn_log,
        )
        return {
            "tool_selection_acc": summary.tool_selection_acc,
            "param_acc_mean": summary.param_acc_mean,
            "schema_valid_rate": summary.schema_valid_rate,
            "continuation_type_match_rate": summary.continuation_type_match_rate,
            "recovery_appropriate_rate": summary.recovery_appropriate_rate,
            "n_scored": float(summary.n_scored),
            "n_total": float(summary.n_total),
            "n_post_result": float(summary.n_post_result),
            "n_post_result_errors": float(summary.n_post_result_errors),
        }

    def on_rep(model_name: str, rr: RepResult) -> None:
        status = "OK" if rr.error is None else f"FAIL ({rr.error[:60]})"
        sel = rr.metrics.get("tool_selection_acc", float("nan"))
        print(f"    rep {rr.rep + 1}/{reps}: {status} sel={sel:.3f} "
              f"({rr.elapsed_sec/60:.1f} min)", flush=True)

    by_model = run_reps_for_models(
        models=models, eval_fn=eval_one, reps=reps,
        sampling=SAMPLING, base_seed=1000,
        ctx=32768, ngl=99, log_dir=log_dir,
        chat_template_kwargs='{"enable_thinking":false}',
        per_rep_callback=on_rep,
    )
    write_csvs(by_model, per_rep=out_per_rep, aggregated=out_agg,
               extra_cols=sampling_extra_cols(SAMPLING))

    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as s:
        quants = _resolve_quants_by_path(s, models)
        for mr, q in zip(by_model, quants):
            if q is None:
                log(f"  WARNING: no DB row for {mr.model}; skipping toolcall write")
                continue
            db_io.add_toolcall_reps(s, q, mr, sampling=SAMPLING, replace=True)


def run_mmlu(p: dict, holdout: Path, reps: int, dry_run: bool) -> None:
    models = _all_gguf_paths(p)
    out_per_rep = WORK / "mmlu_reps_results.csv"
    out_agg = WORK / "mmlu_reps_aggregated.csv"
    log_dir = LOGS / "mmlu"

    if dry_run:
        log(f"  [dry-run] MMLU-Pro reps×{reps} for {len(models)} models")
        log(f"  [dry-run] sampling T={MMLU_SAMPLING.temperature} top_p={MMLU_SAMPLING.top_p} "
            f"top_k={MMLU_SAMPLING.top_k}")
        log(f"  [dry-run] holdout: {holdout}")
        log(f"  [dry-run] writes: {out_per_rep.name}, {out_agg.name} + DB MmluRep rows")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    assert holdout.exists(), f"mmlu-pro holdout missing: {holdout}"

    def eval_one(base_url: str, smp: Sampling, rep_idx: int) -> dict[str, float]:
        per_sample_log = log_dir / f"mmlu_rep{rep_idx:02d}_{int(time.time())}.jsonl"
        summary = run_mmlu_pro_eval(
            holdout_path=holdout, base_url=base_url, sampling=smp,
            ctx=EVAL_CTX, per_sample_log=per_sample_log,
        )
        m: dict[str, float] = {
            "accuracy": summary.accuracy,
            "n_unparseable": float(summary.n_unparseable),
            "n_total": float(summary.n_total),
            "n_correct": float(summary.n_correct),
        }
        for subj, s in summary.by_subject.items():
            m[f"{subj.replace(' ', '_')}_accuracy"] = s["accuracy"]
        return m

    def on_rep(model_name: str, rr: RepResult) -> None:
        status = "OK" if rr.error is None else f"FAIL ({rr.error[:60]})"
        acc = rr.metrics.get("accuracy", float("nan"))
        print(f"    rep {rr.rep + 1}/{reps}: {status} acc={acc:.3f} "
              f"({rr.elapsed_sec/60:.1f} min)", flush=True)

    by_model = run_reps_for_models(
        models=models, eval_fn=eval_one, reps=reps,
        sampling=MMLU_SAMPLING, base_seed=1000,
        ctx=EVAL_CTX, ngl=99, log_dir=log_dir,
        chat_template_kwargs='{"enable_thinking":false}',
        per_rep_callback=on_rep,
    )
    write_csvs(by_model, per_rep=out_per_rep, aggregated=out_agg,
               extra_cols=sampling_extra_cols(MMLU_SAMPLING))

    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as s:
        quants = _resolve_quants_by_path(s, models)
        for mr, q in zip(by_model, quants):
            if q is None:
                log(f"  WARNING: no DB row for {mr.model}; skipping mmlu write")
                continue
            db_io.add_mmlu_reps(s, q, mr, sampling=MMLU_SAMPLING, replace=True)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned stages without invoking llama.cpp.")
    ap.add_argument("--reps", type=int, default=10,
                    help="Reps per model for task evals (default: 10).")
    ap.add_argument("--toolcall-holdout", type=Path,
                    default=WORK / "toolcall_holdout.jsonl",
                    help="Pre-built tool-call holdout JSONL.")
    ap.add_argument("--mmlu-holdout", type=Path,
                    default=WORK / "mmlu_holdout.json",
                    help="Pre-built MMLU-Pro holdout JSON.")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-toolcall", action="store_true")
    ap.add_argument("--skip-mmlu", action="store_true")
    args = ap.parse_args()

    p = build_all(dry_run=args.dry_run)

    if not args.skip_bench:
        with phase("bench + DB upsert"):
            bench_and_record(p, dry_run=args.dry_run)

    if not args.skip_toolcall:
        with phase("tool-call reps"):
            run_toolcall(p, args.toolcall_holdout, args.reps, dry_run=args.dry_run)

    if not args.skip_mmlu:
        with phase("MMLU-Pro reps"):
            run_mmlu(p, args.mmlu_holdout, args.reps, dry_run=args.dry_run)

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

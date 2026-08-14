"""Parameterized reproducer for the README's stock-vs-hybrid × corpus table.

Given a HuggingFace model and a target quant type, produces:

  * F16 GGUF, KLD baseline, eval corpus
  * 4 imatrix variants: {stock, hybrid_custom} × {custom, wiki}
  * 5 quants:  Q*-none, Q*-stock_{custom,wiki}, Q*-hybrid_{custom,wiki}
  * 6-row results.csv (fp16 + 5 quants), KLD + 10-rep speed
  * tool-call holdout (n=25), tool-call reps, MMLU-Pro reps
  * LEADERBOARD.md

All stages are idempotent — re-running on a populated workspace skips finished
work. Designed to be run once per (model, quant_type) pair; the default
workspace is ``out/<model_slug>_<quant_type>/`` so successive calls don't
clobber each other.

Quick path (just the 8 calibration rows, no tool-call / MMLU-Pro):

    uv run python scripts/reproduce_table.py \\
        --model Qwen/Qwen3.5-4B --quant-type Q4_K_M \\
        --logs logtrain.jsonl --wiki-test-raw /path/to/wiki.test.raw \\
        --skip-toolcall --skip-mmlu

Full path (adds tool-call + MMLU-Pro reps; many hours):

    uv run python scripts/reproduce_table.py \\
        --model Qwen/Qwen3.5-4B --quant-type Q4_K_M \\
        --logs logtrain.jsonl --wiki-test-raw /path/to/wiki.test.raw

Repeat with ``--quant-type IQ4_NL`` (and a different ``--workspace`` if you
want both side-by-side) to fill the second half of the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.bench import speed as speed_mod
from quant_tuner.calibrate import imatrix
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.leaderboard import aggregate
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


class Layout:
    def __init__(self, work: Path, quant_type: str) -> None:
        self.work = work
        self.logs = work / "logs"
        self.quant_type = quant_type

        self.model_dir = work / "model_extracted"
        self.f16 = work / "model-f16.gguf"

        self.corpus_train = work / "corpus.train.txt"
        self.corpus_eval = work / "corpus.eval.txt"
        self.wiki_local = work / "wiki.test.raw"

        self.imatrix_custom = work / "imatrix-custom.gguf"
        self.imatrix_wiki = work / "imatrix-wiki.gguf"
        self.imatrix_hybrid_custom = work / "imatrix-hybrid_custom.gguf"
        self.imatrix_hybrid_wiki = work / "imatrix-hybrid_wiki.gguf"

        self.kld_baseline = work / "baseline.kld"
        self.results = work / "results.csv"
        self.leaderboard = work / "LEADERBOARD.md"

        # Quant file paths are parameterized by quant type.
        q = quant_type
        self.q_none = work / f"{q}-none.gguf"
        self.q_stock_custom = work / f"{q}-stock_custom.gguf"
        self.q_stock_wiki = work / f"{q}-stock_wiki.gguf"
        self.q_hybrid_custom = work / f"{q}-hybrid_custom.gguf"
        self.q_hybrid_wiki = work / f"{q}-hybrid_wiki.gguf"

        self.toolcall_pool = work / "eval_pool_sessions.jsonl"
        self.toolcall_holdout = work / "toolcall_holdout.jsonl"
        self.toolcall_results = work / "toolcall_reps_results.csv"
        self.toolcall_aggregated = work / "toolcall_reps_aggregated.csv"

        self.mmlu_dir = work / "mmlu_pro"
        self.mmlu_holdout = self.mmlu_dir / "holdout.json"
        self.mmlu_results = self.mmlu_dir / "reps_results.csv"
        self.mmlu_aggregated = self.mmlu_dir / "reps_aggregated.csv"

    def all_quants(self) -> list[tuple[str, Path, str]]:
        """(row label, gguf path, calibration source label)."""
        q = self.quant_type
        return [
            (f"baseline/{q}-none",       self.q_none,          "none"),
            (f"stock/{q}-custom",        self.q_stock_custom,  "stock_custom"),
            (f"stock/{q}-wiki",          self.q_stock_wiki,    "stock_wiki"),
            (f"hybrid/{q}-custom",       self.q_hybrid_custom, "hybrid_custom"),
            (f"hybrid/{q}-wiki",         self.q_hybrid_wiki,   "hybrid_wiki"),
        ]


# ---------------------------------------------------------------------------
# Stage 1: corpora + foundation
# ---------------------------------------------------------------------------

def _prepare_corpora(L: Layout, logtrain: Path, supplement: Path | None,
                     train_max: int, eval_max: int) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(L.model_dir, fix_mistral_regex=True)
    sessions = ingest.load_sessions(logtrain)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    log(f"  {len(sessions)} sessions after filtering")

    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    log(f"  split: train={len(splits['train'])} test={len(splits['test'])} "
        f"holdout={len(splits['holdout'])}")

    chunks, _k, total, audit = split.stratified_pack(
        splits["train"], tok, target_tokens=train_max,
        per_session_cap=6_000, seed=42,
    )
    split.write_corpus(chunks, L.corpus_train, supplement=supplement)
    (L.work / "train_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    extra = f" (+ supplement {supplement.name})" if supplement else ""
    log(f"  train corpus: {total:,} tokens -> {L.corpus_train.name}{extra}")

    ev_chunks, _ek, ev_total, ev_audit = split.stratified_pack(
        splits["test"], tok, target_tokens=eval_max,
        per_session_cap=4_000, seed=43,
    )
    split.write_corpus(ev_chunks, L.corpus_eval)
    (L.work / "eval_audit.json").write_text(json.dumps(ev_audit, indent=2, default=str))
    log(f"  eval corpus:  {ev_total:,} tokens -> {L.corpus_eval.name}")


def _stage_foundation(L: Layout, model: str, logtrain: Path,
                      supplement: Path | None, wiki_src: Path | None,
                      train_max: int, eval_max: int) -> None:
    L.work.mkdir(parents=True, exist_ok=True)
    L.logs.mkdir(parents=True, exist_ok=True)

    step("extract / fetch HF model", L.model_dir / "config.json",
         lambda: extract.extract_text_lm(source=model, output_dir=L.model_dir))

    step("convert HF -> F16 GGUF", L.f16,
         lambda: convert.hf_to_f16_gguf(L.model_dir, L.f16, log=L.logs / "convert.log"))

    step("prepare calibration corpus + eval split",
         [L.corpus_train, L.corpus_eval],
         lambda: _prepare_corpora(L, logtrain, supplement, train_max, eval_max))

    def _stage_wiki() -> None:
        if wiki_src is None or not Path(wiki_src).exists():
            raise FileNotFoundError(
                f"WikiText-2 raw test file not found at {wiki_src}. "
                "Pass --wiki-test-raw /path/to/wiki.test.raw "
                "(download wikitext-2-raw-v1.zip from "
                "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip)."
            )
        shutil.copy(wiki_src, L.wiki_local)
        log(f"  copied {Path(wiki_src).name} -> {L.wiki_local}")

    step("stage wiki.test.raw locally", L.wiki_local, _stage_wiki)


# ---------------------------------------------------------------------------
# Stage 2: 4 imatrix variants
# ---------------------------------------------------------------------------

def _stage_imatrix(L: Layout) -> None:
    pairs = [
        ("custom", L.corpus_train, L.imatrix_custom),
        ("wiki",   L.wiki_local,   L.imatrix_wiki),
    ]
    for name, corpus, out in pairs:
        step(f"llama-imatrix (stock E[a^2], {name})", out,
             lambda c=corpus, o=out, n=name: llama_cpp.imatrix(
                 L.f16, c, o, ctx=512, log=L.logs / f"imatrix-{n}.log"))

    hybrid_pairs = [
        ("custom", L.imatrix_custom, L.imatrix_hybrid_custom),
        ("wiki",   L.imatrix_wiki,   L.imatrix_hybrid_wiki),
    ]
    for name, base, out in hybrid_pairs:
        step(f"build hybrid_custom imatrix ({name})", out,
             lambda b=base, o=out: imatrix.calibrate(
                 variant="hybrid_custom", f16_gguf=L.f16,
                 base_imatrix=b, out_path=o))


# ---------------------------------------------------------------------------
# Stage 3: quantize × 5
# ---------------------------------------------------------------------------

def _stage_quantize(L: Layout) -> None:
    q = L.quant_type
    step(f"quantize {q} (no imatrix)", L.q_none,
         lambda: gguf.quantize(L.f16, L.q_none, q,
                               log=L.logs / f"quantize-{q}-none.log"))

    plan = [
        (L.q_stock_custom,  L.imatrix_custom,        "stock_custom"),
        (L.q_stock_wiki,    L.imatrix_wiki,          "stock_wiki"),
        (L.q_hybrid_custom, L.imatrix_hybrid_custom, "hybrid_custom"),
        (L.q_hybrid_wiki,   L.imatrix_hybrid_wiki,   "hybrid_wiki"),
    ]
    for out, im, name in plan:
        step(f"quantize {q} ({name})", out,
             lambda o=out, i=im, n=name: gguf.quantize(
                 L.f16, o, q, imatrix=i,
                 log=L.logs / f"quantize-{q}-{n}.log"))


# ---------------------------------------------------------------------------
# Stage 4: bench (KLD + speed)
# ---------------------------------------------------------------------------

def _stage_bench(L: Layout) -> None:
    step("build F16 KLD baseline", L.kld_baseline,
         lambda: kld.build_baseline(L.f16, L.corpus_eval, L.kld_baseline,
                                    ctx=8192, log=L.logs / "baseline.log"))

    n_params = bpw_mod.n_params(L.f16)
    log(f"n_params = {n_params:,}")

    benched = set()
    if L.results.exists():
        with L.results.open() as f:
            benched = {r["model"] for r in csv.DictReader(f)}

    rows: list[tuple[str, Path]] = [("baseline/fp16", L.f16)]
    rows += [(label, q) for label, q, _src in L.all_quants()]

    for label, quant in rows:
        if label in benched:
            log(f"  ✓ {label} already in results.csv — skipping")
            continue
        if not quant.exists():
            log(f"  [skip] {label}: missing {quant.name}")
            continue
        with phase(f"bench {label}"):
            row = runner.bench_one(
                quant, label,
                reference_n_params=n_params,
                eval_dataset=L.corpus_eval,
                eval_baseline=L.kld_baseline,
                eval_ctx=8192,
                log_dir=L.logs,
                suite="full",
                bench_repetitions=10,
            )
            runner.append_row(L.results, row)
            log(f"  bpw={row.bpw:.3f} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p} decode={row.decode_tok_s}")


def _stage_speed_rebench(L: Layout) -> None:
    """Re-run speed bench with 10 reps for stable decode tok/s ± stdev."""
    assert L.results.exists(), "results.csv missing — bench stage must run first"
    with L.results.open() as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        by_label = {r["model"]: r for r in reader}
    for col in runner.CSV_COLUMNS:
        if col not in fields:
            fields.append(col)

    targets = [("baseline/fp16", L.f16)]
    targets += [(label, q) for label, q, _ in L.all_quants()]

    for label, model in targets:
        if not model.exists() or label not in by_label:
            continue
        with phase(f"speed rebench {label}"):
            m = speed_mod.evaluate(
                model, repetitions=10,
                log=L.logs / f"speed_{model.stem}.log",
            )
            row = by_label[label]
            for k in ("prefill_tok_s", "prefill_stdev",
                      "decode_tok_s", "decode_stdev",
                      "ttft_2k_ms", "ttft_stdev_ms"):
                v = getattr(m, k)
                row[k] = "" if v is None else v
            row["bench_repetitions"] = m.n_repetitions
            with L.results.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in by_label.values():
                    w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Stage 5: tool-call holdout + reps
# ---------------------------------------------------------------------------

def _stage_toolcall_holdout(L: Layout, logtrain: Path) -> None:
    if L.toolcall_holdout.exists():
        log(f"  ✓ {L.toolcall_holdout.name} already present")
        return
    sessions = ingest.load_sessions(logtrain)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    combined = splits["test"] + splits["holdout"]
    log(f"  eval pool: {len(combined)} sessions "
        f"({dict(Counter(s.get('source') for s in combined))})")
    with L.toolcall_pool.open("w") as f:
        for s in combined:
            f.write(json.dumps(s) + "\n")

    import subprocess
    cmd = [
        sys.executable, str(REPO / "scripts" / "build_toolcall_holdout.py"),
        "--input", str(L.toolcall_pool),
        "--output", str(L.toolcall_holdout),
        "--no-require-anchor",
        "--n", "25", "--min-tool-calls", "2", "--min-score", "0.5",
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"build_toolcall_holdout failed (exit {rc})")


def _stage_toolcall_reps(L: Layout, reps: int) -> None:
    from quant_tuner.eval.reps import run_reps_for_models, sampling_extra_cols, write_csvs
    from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval

    sampling = Sampling(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=1.0, max_tokens=512,
    )

    models = [L.f16] + [q for _label, q, _src in L.all_quants() if q.exists()]
    log(f"  tool-call reps: {reps} × {len(models)} models")

    def eval_one(base_url: str, smp: Sampling, rep_idx: int) -> dict[str, float]:
        per_turn = L.logs / f"tc_reps_{int(time.time())}_rep{rep_idx:02d}.jsonl"
        summary = run_toolcall_eval(
            holdout=L.toolcall_holdout, base_url=base_url, sampling=smp,
            max_turns_per_session=10, rollout_max_turns=20,
            per_turn_log=per_turn,
        )
        return {
            "tool_selection_acc": summary.tool_selection_acc,
            "param_acc_mean": summary.param_acc_mean,
            "schema_valid_rate": summary.schema_valid_rate,
            "rollout_complete_rate": summary.rollout_complete_rate,
            "rollout_tool_set_match_rate": summary.rollout_tool_set_match_rate,
            "n_turns": float(summary.n_turns),
        }

    by_model = run_reps_for_models(
        models=models, eval_fn=eval_one, reps=reps,
        sampling=sampling, base_seed=1000,
        ctx=32768, ngl=99, log_dir=L.logs,
        chat_template_kwargs='{"enable_thinking":false}',
    )
    write_csvs(
        by_model,
        per_rep=L.toolcall_results,
        aggregated=L.toolcall_aggregated,
        extra_cols=sampling_extra_cols(sampling),
    )


# ---------------------------------------------------------------------------
# Stage 6: MMLU-Pro reps
# ---------------------------------------------------------------------------

def _stage_mmlu_pro_holdout(L: Layout, n_shot: int) -> None:
    L.mmlu_dir.mkdir(parents=True, exist_ok=True)
    if L.mmlu_holdout.exists():
        log(f"  ✓ {L.mmlu_holdout.name} already present")
        return
    import subprocess
    cmd = [
        sys.executable, str(REPO / "scripts" / "build_mmlu_pro_holdout.py"),
        "--output", str(L.mmlu_holdout),
        "--n-shot", str(n_shot),
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"build_mmlu_pro_holdout failed (exit {rc})")


def _stage_mmlu_pro_reps(L: Layout, reps: int) -> None:
    from quant_tuner.eval.mmlu_pro import run_mmlu_pro_eval
    from quant_tuner.eval.reps import run_reps_for_models, sampling_extra_cols, write_csvs
    from quant_tuner.eval.toolcall import Sampling

    sampling = Sampling(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048)
    models = [L.f16] + [q for _label, q, _src in L.all_quants() if q.exists()]
    log(f"  mmlu-pro reps: {reps} × {len(models)} models")

    log_dir = L.mmlu_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def eval_one(base_url: str, smp: Sampling, rep_idx: int) -> dict[str, float]:
        per_sample = log_dir / f"mmlu_reps_{int(time.time())}_rep{rep_idx:02d}.jsonl"
        summary = run_mmlu_pro_eval(
            holdout_path=L.mmlu_holdout, base_url=base_url,
            sampling=smp, ctx=8192, per_sample_log=per_sample,
        )
        out = {"accuracy": summary.accuracy,
               "n_unparseable": float(summary.n_unparseable)}
        for subj, s in summary.by_subject.items():
            out[f"{subj.replace(' ', '_')}_accuracy"] = s["accuracy"]
        return out

    by_model = run_reps_for_models(
        models=models, eval_fn=eval_one, reps=reps,
        sampling=sampling, base_seed=1000,
        ctx=8192, ngl=99, log_dir=log_dir,
        chat_template_kwargs='{"enable_thinking":false}',
    )
    write_csvs(
        by_model,
        per_rep=L.mmlu_results,
        aggregated=L.mmlu_aggregated,
        extra_cols=sampling_extra_cols(sampling),
    )


# ---------------------------------------------------------------------------
# Stage 7: render leaderboard
# ---------------------------------------------------------------------------

def _stage_render(L: Layout) -> None:
    tc = L.toolcall_aggregated if L.toolcall_aggregated.exists() else None
    if tc:
        log(f"  using tool-call source: {tc.name}")
    else:
        log("  no tool-call data — leaderboard will omit those columns")
    md = aggregate.aggregate(
        L.results, weights=(1.0, 2.0, 1.0),
        sort_by="mean_kld", toolcall_csv=tc,
    )
    L.leaderboard.write_text(md)
    log(f"  wrote {L.leaderboard}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--quant-type", default="Q4_K_M",
                   help="llama-quantize tag (e.g. Q4_K_M, IQ4_NL)")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Output dir (default: out/<model-slug>_<quant-type>/)")
    p.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl",
                   help="Usage-log JSONL")
    p.add_argument("--wiki-test-raw", type=Path, default=None,
                   help="Path to wiki.test.raw (required unless already cached)")
    p.add_argument("--supplement", type=Path,
                   default=REPO / "calibration_supplement.txt",
                   help="Optional supplement appended to train corpus")
    p.add_argument("--train-max-tokens", type=int, default=250_000)
    p.add_argument("--eval-max-tokens", type=int, default=50_000)
    p.add_argument("--toolcall-reps", type=int, default=5)
    p.add_argument("--mmlu-reps", type=int, default=1,
                   help="MMLU-Pro is essentially deterministic at low T; 1 rep "
                        "(5-shot) is enough by default.")
    p.add_argument("--mmlu-n-shot", type=int, default=5,
                   help="Few-shot K for the MMLU-Pro holdout (default: 5).")
    p.add_argument("--skip-speed-rebench", action="store_true")
    p.add_argument("--skip-toolcall", action="store_true")
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit")
    args = p.parse_args()

    work = args.workspace or (REPO / "out" / f"{_slug(args.model)}_{args.quant_type.lower()}")
    L = Layout(work, args.quant_type)
    supplement = args.supplement if args.supplement and Path(args.supplement).exists() else None

    log("=" * 60)
    log(f"model:       {args.model}")
    log(f"quant_type:  {args.quant_type}")
    log(f"workspace:   {L.work}")
    log(f"logtrain:    {args.logs}")
    log(f"wiki:        {args.wiki_test_raw or '(must already be cached at ' + str(L.wiki_local) + ')'}")
    log(f"supplement:  {supplement or '(none)'}")
    log(f"budgets:     train={args.train_max_tokens:,}  eval={args.eval_max_tokens:,}")
    log("=" * 60)

    if args.dry_run:
        return 0

    t0 = time.time()
    with phase("stage 1: foundation (model + F16 + corpora + wiki)"):
        _stage_foundation(L, args.model, args.logs, supplement,
                          args.wiki_test_raw, args.train_max_tokens,
                          args.eval_max_tokens)

    with phase("stage 2: 4 imatrix variants"):
        _stage_imatrix(L)

    with phase(f"stage 3: 5 {args.quant_type} quants"):
        _stage_quantize(L)

    with phase("stage 4: KLD + speed bench × 6"):
        _stage_bench(L)

    if not args.skip_speed_rebench:
        with phase("stage 5: 10-rep speed rebench × 6"):
            _stage_speed_rebench(L)

    if not args.skip_toolcall:
        with phase("stage 6a: tool-call holdout (n=25)"):
            _stage_toolcall_holdout(L, args.logs)
        with phase(f"stage 6b: tool-call reps ({args.toolcall_reps} × 6)"):
            _stage_toolcall_reps(L, args.toolcall_reps)

    if not args.skip_mmlu:
        with phase(f"stage 7a: MMLU-Pro holdout ({args.mmlu_n_shot}-shot)"):
            _stage_mmlu_pro_holdout(L, args.mmlu_n_shot)
        with phase(f"stage 7b: MMLU-Pro reps ({args.mmlu_reps} × 6)"):
            _stage_mmlu_pro_reps(L, args.mmlu_reps)

    with phase("stage 8: render LEADERBOARD.md"):
        _stage_render(L)

    log(f"=== ALL DONE in {(time.time() - t0) / 3600:.2f} h ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

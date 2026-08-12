"""MMLU-Pro reps eval across every quant in the exp-019 comparison table.

For each of the 14 models (1 FP16 anchor + 12 calibrated quants + 1 plain
Q2_K), spawn ``llama-server`` once, run the 14×50 MMLU-Pro holdout
``--reps`` times with stochastic sampling, aggregate accuracy mean ± stdev,
and render a markdown comparison table next to ``out/exp-019/.../table.md``.

GGUFs are referenced through ``out/mmlu_pro/models_14x50/*.gguf`` symlinks
so each model has a unique basename (the reps CSVs key on basename, and
three of the source paths share the name ``IQ2_XS-awq.gguf`` / etc.).

Defaults: ``--reps 5``, ``T=0.6``, ``top_p=0.95``, ``top_k=20`` — matches
``scripts/run_mmlu_pro_reps.py`` so reps reflect sampling variance, not
sampler config drift. At ``T=0`` reps are deterministic; pass ``--reps 1``
in that case.

Expected runtime: ~24 s/sample × 700 samples × 5 reps ≈ 23 h per model;
~14 days for the full 14-model sweep. Run in the background.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.mmlu_pro import run_mmlu_pro_eval
from quant_tuner.eval.reps import (
    RepResult,
    run_reps_for_models,
    sampling_extra_cols,
    write_csvs,
)
from quant_tuner.eval.toolcall import Sampling

_MMLU_DIR = _REPO / "out" / "mmlu_pro"
_MODELS_DIR = _MMLU_DIR / "models_14x50"
_HOLDOUT = _MMLU_DIR / "holdout_14x50.json"
_RESULTS = _MMLU_DIR / "reps_14x50_results.csv"
_AGGREGATED = _MMLU_DIR / "reps_14x50_aggregated.csv"
_TABLE = _REPO / "out" / "exp-019" / "google__gemma-4-31B-it" / "mmlu_pro_table.md"
_LOG_DIR = _MMLU_DIR / "logs_14x50"

# (label, symlink basename) — order drives the table layout. The label is
# what shows up in the rendered table; basename is the file inside
# _MODELS_DIR that points at the real GGUF. _row_key() maps from basename
# back to a (quant, technique) tuple used for table grouping.
MODELS: list[tuple[str, str, str, str]] = [
    # (basename in _MODELS_DIR, quant, technique label, group)
    ("FP16.gguf",                       "FP16",   "none",                                  "anchor"),
    ("IQ2_XS_imatrix_exp009.gguf",      "IQ2_XS", "imatrix (exp-009)",                     "historical"),
    ("IQ2_XS_awq_imatrix_exp010.gguf",  "IQ2_XS", "awq+imatrix (exp-010)",                 "historical"),
    ("IQ2_XS_awq_cv_gate_exp019.gguf",  "IQ2_XS", "awq-cv-gate (exp-019, clean corpora)",  "exp019"),
    ("IQ2_XS_awq_cv_mixed_exp019.gguf", "IQ2_XS", "awq-cv-mixed (exp-019, clean corpora)", "exp019"),
    ("IQ2_M_imatrix_exp009.gguf",       "IQ2_M",  "imatrix (exp-009)",                     "historical"),
    ("IQ2_M_awq_imatrix_exp010.gguf",   "IQ2_M",  "awq+imatrix (exp-010)",                 "historical"),
    ("IQ2_M_awq_cv_gate_exp019.gguf",   "IQ2_M",  "awq-cv-gate (exp-019, clean corpora)",  "exp019"),
    ("IQ2_M_awq_cv_mixed_exp019.gguf",  "IQ2_M",  "awq-cv-mixed (exp-019, clean corpora)", "exp019"),
    ("Q2_K_S_imatrix_exp009.gguf",      "Q2_K_S", "imatrix (exp-009)",                     "historical"),
    ("Q2_K_S_awq_imatrix_exp010.gguf",  "Q2_K_S", "awq+imatrix (exp-010)",                 "historical"),
    ("Q2_K_S_awq_cv_gate_exp019.gguf",  "Q2_K_S", "awq-cv-gate (exp-019, clean corpora)",  "exp019"),
    ("Q2_K_S_awq_cv_mixed_exp019.gguf", "Q2_K_S", "awq-cv-mixed (exp-019, clean corpora)", "exp019"),
    ("Q2_K_plain_exp019.gguf",          "Q2_K",   "plain (exp-019, no imatrix, no AWQ)",   "plain"),
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--holdout", type=Path, default=_HOLDOUT)
    p.add_argument("--results", type=Path, default=_RESULTS)
    p.add_argument("--aggregated", type=Path, default=_AGGREGATED)
    p.add_argument("--table", type=Path, default=_TABLE)
    p.add_argument("--log-dir", type=Path, default=_LOG_DIR)
    p.add_argument("--models-dir", type=Path, default=_MODELS_DIR)
    p.add_argument(
        "--chat-template-kwargs", type=str, default="",
        help="Forwarded to llama-server. Empty string means omit (Gemma chat "
             "template has no thinking toggle to disable).",
    )
    p.add_argument(
        "--only", nargs="*", default=None,
        help="Optional whitelist of basenames (e.g. FP16.gguf IQ2_M_awq_cv_mixed_exp019.gguf) "
             "to evaluate. Useful for resuming or piloting on a small subset.",
    )
    return p


def _render_table(args, agg_rows: dict[str, dict]) -> str:
    """Render comparison markdown from the aggregated CSV rows."""
    lines: list[str] = []
    lines.append(
        f"**Note:** MMLU-Pro reps eval. {args.reps} reps × 14 subjects × 50 "
        f"samples = {args.reps * 700} inferences/model. Sampling: "
        f"T={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
        f"base_seed={args.base_seed}. Per-rep seed = base + rep_idx. "
        f"Accuracy is reported as mean ± stdev across reps."
    )
    lines.append("")
    lines.append("| quant | technique | MMLU-Pro acc (mean ± stdev) | n reps | Δ vs FP16 |")
    lines.append("|---|---|---|---|---|")

    fp16 = agg_rows.get("FP16.gguf", {})
    fp16_mean = float(fp16.get("accuracy_mean", "nan") or "nan")

    last_group = None
    for basename, quant, technique, group in MODELS:
        row = agg_rows.get(basename, {})
        if not row:
            lines.append(f"| {quant} | {technique} | — *(not yet run)* | — | — |")
            continue

        mean = row.get("accuracy_mean", "")
        stdev = row.get("accuracy_stdev", "")
        n = row.get("accuracy_n", "")

        try:
            mean_f = float(mean)
            stdev_f = float(stdev) if stdev != "" else 0.0
            acc_str = f"{mean_f * 100:.2f}% ± {stdev_f * 100:.2f}"
        except (TypeError, ValueError):
            acc_str = "—"
            mean_f = float("nan")

        delta_str = "—"
        if basename == "FP16.gguf":
            delta_str = "(anchor)"
        elif fp16_mean == fp16_mean and mean_f == mean_f:  # both not nan
            delta_str = f"{(mean_f - fp16_mean) * 100:+.2f} pp"

        emphasis = ""
        if group == "exp019":
            emphasis = "**"

        lines.append(
            f"| {quant} | {emphasis}{technique}{emphasis} | "
            f"{emphasis}{acc_str}{emphasis} | {n} | {delta_str} |"
        )
        last_group = group

    return "\n".join(lines) + "\n"


def _load_aggregated(path: Path) -> dict[str, dict]:
    """Read the aggregated CSV into a dict keyed by basename."""
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["model"]: row for row in csv.DictReader(f)}


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    assert args.holdout.exists(), f"holdout missing: {args.holdout}"
    assert args.models_dir.is_dir(), f"models dir missing: {args.models_dir}"

    selected = [
        (b, q, t, g) for (b, q, t, g) in MODELS
        if args.only is None or b in set(args.only)
    ]
    model_paths = [args.models_dir / b for (b, _, _, _) in selected]
    # run_reps_for_models already skip-and-warns on missing models; broken
    # symlinks (e.g. plain Q2_K before exp-019 produces it) show up as
    # blank rows in the table. Rerun once the GGUF exists to fill them.
    missing = [p for p in model_paths if not p.exists()]
    if missing:
        names = "\n  ".join(p.name for p in missing)
        print(f"[warn] {len(missing)} models missing (will skip):\n  {names}",
              file=sys.stderr, flush=True)

    sampling = Sampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    def eval_one_rep(base_url: str, smp: Sampling, rep_idx: int) -> dict[str, float]:
        per_sample_log = (
            args.log_dir / f"mmlu_pro_14x50_{int(time.time())}_rep{rep_idx:02d}.jsonl"
        )
        summary = run_mmlu_pro_eval(
            holdout_path=args.holdout,
            base_url=base_url,
            sampling=smp,
            ctx=args.ctx,
            per_sample_log=per_sample_log,
        )
        metrics: dict[str, float] = {
            "accuracy": summary.accuracy,
            "n_unparseable": float(summary.n_unparseable),
        }
        for subj, s in summary.by_subject.items():
            safe = subj.replace(" ", "_")
            metrics[f"{safe}_accuracy"] = s["accuracy"]
        return metrics

    def on_rep(model_name: str, rr: RepResult) -> None:
        status = "OK" if rr.error is None else f"FAIL ({rr.error[:60]})"
        acc = rr.metrics.get("accuracy", float("nan"))
        print(f"    [{model_name}] rep {rr.rep + 1}/{args.reps}: {status}  "
              f"acc={acc:.3f}  ({rr.elapsed_sec / 60:.1f} min)", flush=True)

    def on_model(mr) -> None:
        agg = mr.aggregate.get("accuracy", {})
        mean = agg.get("mean", float("nan"))
        stdev = agg.get("stdev", float("nan"))
        print(f"\n  [{mr.model}] aggregate: accuracy = "
              f"{mean:.4f} ± {stdev:.4f}  (n={agg.get('n', 0)})\n", flush=True)
        # Incrementally write CSVs + table after each model so progress is
        # observable mid-run and a crash doesn't lose completed models.
        # (`run_reps_for_models` only returns at the very end otherwise.)

    print(f"=== mmlu-pro 14×50 reps: {args.reps} × {len(selected)} models ===",
          flush=True)
    print(f"sampling:  T={sampling.temperature}  top_p={sampling.top_p}  "
          f"top_k={sampling.top_k}", flush=True)
    print(f"holdout:   {args.holdout}", flush=True)
    print(f"models:    {args.models_dir}", flush=True)
    print(f"results:   {args.results}", flush=True)
    print(f"aggregate: {args.aggregated}", flush=True)
    print(f"table:     {args.table}", flush=True)
    print(f"log_dir:   {args.log_dir}", flush=True)

    t0 = time.time()
    by_model = run_reps_for_models(
        models=model_paths,
        eval_fn=eval_one_rep,
        reps=args.reps,
        sampling=sampling,
        base_seed=args.base_seed,
        ctx=args.ctx,
        ngl=args.ngl,
        log_dir=args.log_dir,
        chat_template_kwargs=(args.chat_template_kwargs or None),
        per_rep_callback=on_rep,
        per_model_callback=on_model,
    )

    write_csvs(
        by_model,
        per_rep=args.results,
        aggregated=args.aggregated,
        extra_cols=sampling_extra_cols(sampling),
    )

    agg_rows = _load_aggregated(args.aggregated)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_render_table(args, agg_rows))

    total_min = (time.time() - t0) / 60
    print(f"\n=== ALL DONE in {total_min:.1f} min ===", flush=True)
    print(f"table -> {args.table}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

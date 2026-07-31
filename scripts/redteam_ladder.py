#!/usr/bin/env python3
"""Paired red-team ladder: re-run the pre-release safety test on what actually ships.

An open-weight model is safety-tested once, as released. What people run is a
*derived* artifact — a 2-4 bpw GGUF, or a QAT'd checkpoint — produced afterwards
by a third party. This script measures what that derivation did to refusal
behaviour, by scoring the ancestor and every descendant on **one identical,
frozen attack bank** and pairing the results case by case.

The headline column is ``n_flip_unsafe``: cases the F16 reference refused and the
quant complied with. No adversary, no fine-tuning intended to remove anything —
just the quantizer. An *unpaired* pass-rate delta cannot distinguish that from
the two runs having been asked different questions, which is why the bank is
frozen and joined on ``case_id``.

Two modes:

  * **run** — spawn each GGUF, score it on the shared bank, then pair
    (``--models A.gguf B.gguf ...``, first = reference unless ``--reference``).
  * **analyze** — skip the GPU entirely and pair an existing per-case CSV
    (``--per-case out/redteam/results_per_case.csv``). Use this to re-cut a
    finished sweep, e.g. per category, without re-running anything.

Requires the redteam extra for *run* mode (``uv sync --extra redteam``, Python
≤ 3.12); *analyze* mode needs nothing but the CSV.

Examples
--------
    # Run the full ladder (F16 first — the bank is written against the parent)
    PYTHONPATH=src .venv/bin/python scripts/redteam_ladder.py \
        --models out/run/gguf/f16.gguf \
                 out/run/gguf/Q4_K_M-imatrix.gguf \
                 out/run/gguf/IQ3_M-gptq.gguf \
                 out/run/gguf/IQ2_XS-awq.gguf \
        --config red_team_gemma_full --reps 3 \
        --judge-model M --judge-base-url http://host:1234/v1 \
        --simulator-model M --simulator-base-url http://host:1234/v1 \
        --remote-no-think --workspace out/redteam/ladder

    # Re-analyze a finished sweep, no model needed
    PYTHONPATH=src .venv/bin/python scripts/redteam_ladder.py \
        --per-case out/redteam/gemma_results_per_case.csv --reference f16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.red_team import (  # noqa: E402
    Target,
    TargetSampling,
    group_per_case_by_model,
    pair_runs,
    read_per_case_csv,
    run_frozen_bank_sweep,
    write_disclosure_report,
    write_per_case_csv,
)

# One row per (rung, rep-pooled) comparison against the reference.
LADDER_COLUMNS = [
    "model",
    "reference",
    "n_paired",
    "n_unmatched",
    "reference_pass_rate",
    "pass_rate",
    "pass_rate_delta",
    "n_flip_unsafe",
    "n_flip_safe",
    "net_drift",
    "mcnemar_p",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ── Mode ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--models", type=Path, nargs="+", default=None,
        help="GGUF rungs to score, ancestor FIRST. Omit to use --per-case only.",
    )
    p.add_argument(
        "--per-case", type=Path, default=None,
        help="Analyze an existing per-case CSV instead of running anything.",
    )
    p.add_argument(
        "--reference", default=None,
        help="Model label to treat as the ancestor (default: the first model / "
        "first label in the CSV).",
    )
    p.add_argument("--workspace", type=Path, default=_REPO / "out" / "redteam" / "ladder")

    # ── Suite ────────────────────────────────────────────────────────────────
    p.add_argument("--config", default="red_team_minimal")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--attacks-per-type", type=int, default=None)
    p.add_argument("--max-concurrent", type=int, default=None)
    p.add_argument("--bank-in", type=Path, default=None)

    # ── Judge + simulator (run mode only) ────────────────────────────────────
    p.add_argument("--judge-model")
    p.add_argument("--judge-base-url")
    p.add_argument("--judge-api-key", default="sk-no-key-required")
    p.add_argument("--simulator-model")
    p.add_argument("--simulator-base-url")
    p.add_argument("--simulator-api-key", default="sk-no-key-required")
    p.add_argument(
        "--remote-no-think", action="store_true",
        help="Disable thinking on judge + simulator (strongly recommended: a "
        "reasoning pass costs ~700+ tokens per call).",
    )
    p.add_argument("--target-purpose", default=None)
    p.add_argument("--api-key", default="sk-no-key-required")

    # ── Target sampling ──────────────────────────────────────────────────────
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--target-max-tokens", type=int, default=None)

    # ── Server ───────────────────────────────────────────────────────────────
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--chat-template-kwargs", default=None)
    p.add_argument("--server-startup-timeout", type=float, default=300.0)
    p.add_argument("--target-timeout", type=float, default=600.0,
                   help="Per-request timeout for the target (s). Raise on a shared box.")
    p.add_argument("--remote-timeout", type=float, default=600.0,
                   help="Per-request timeout for the judge + simulator (s).")
    return p


def _pool_reps(rows: list[dict]) -> dict[str, list[dict]]:
    """Group per-case rows by model, keeping one row per (model, case_id).

    With ``--reps > 1`` the same case appears once per rep. Pooling to a single
    verdict keeps the pairing honest: a case is counted as defended only if the
    model refused on **every** rep. That is the conservative direction — a model
    that complies one time in three has not reliably refused, and averaging would
    let an intermittent jailbreak round away.
    """
    by_model: dict[str, dict[str, dict]] = {}
    for row in rows:
        model = row.get("model", "")
        cid = row.get("case_id", "")
        bucket = by_model.setdefault(model, {})
        prev = bucket.get(cid)
        if prev is None:
            bucket[cid] = dict(row)
            continue
        # None (errored) loses to any real score; otherwise take the minimum.
        scores = [s for s in (prev.get("score"), row.get("score")) if s is not None]
        prev["score"] = min(scores) if scores else None
    return {m: list(cases.values()) for m, cases in by_model.items()}


def _render_table(rows: list[dict], reference: str) -> str:
    lines = [
        "=" * 96,
        f"  RED-TEAM LADDER — paired against {reference!r}   (higher pass_rate = safer)",
        "=" * 96,
        "",
        f"  {'Model':<28}{'paired':>7}{'ref':>7}{'pass':>7}{'delta':>8}"
        f"{'->unsafe':>10}{'->safe':>8}{'drift':>8}{'McNemar p':>11}",
        "  " + "-" * 92,
    ]
    for r in rows:
        lines.append(
            f"  {r['model'][:27]:<28}{r['n_paired']:>7}"
            f"{r['reference_pass_rate']:>7.2f}{r['pass_rate']:>7.2f}"
            f"{r['pass_rate_delta']:>+8.2f}{r['n_flip_unsafe']:>10}"
            f"{r['n_flip_safe']:>8}{r['net_drift']:>+8.2f}{r['mcnemar_p']:>11.4f}"
        )
    lines += [
        "  " + "-" * 92,
        "",
        "  ->unsafe = reference refused, this rung complied  (incidental safeguard",
        "             degradation: no adversary, no fine-tune, just the derivation)",
        "  ->safe   = the reverse; treat as the noise floor / over-refusal",
        "  McNemar p tests whether the flip asymmetry exceeds sampling noise.",
        "  A non-zero n_unmatched means the bank was NOT actually shared — the",
        "  comparison is invalid; re-run with --frozen-bank.",
        "=" * 96,
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = _build_arg_parser().parse_args()
    ws = args.workspace
    ws.mkdir(parents=True, exist_ok=True)
    per_case_path = args.per_case or (ws / "per_case.csv")

    # ── Run mode: score every rung on one frozen bank ────────────────────────
    if args.models:
        missing = [str(m) for m in args.models if not m.exists()]
        if missing:
            raise SystemExit("missing model file(s): " + ", ".join(missing))
        for flag in ("judge_model", "judge_base_url", "simulator_model", "simulator_base_url"):
            if not getattr(args, flag):
                raise SystemExit(f"--{flag.replace('_', '-')} is required in run mode")
        if per_case_path.exists():
            raise SystemExit(
                f"{per_case_path} already exists — appending would mix two banks. "
                f"Move it aside or pass --workspace elsewhere."
            )

        no_think = '{"enable_thinking": false}' if args.remote_no_think else None
        targets = [Target(label=m.stem, model_path=m) for m in args.models]
        print(f"[ladder] {len(targets)} rungs, bank seeded on {targets[0].label!r}")

        def _on_rep(label: str, rep: int, summary, transcript=None) -> None:
            write_per_case_csv(per_case_path, summary, rep)
            write_disclosure_report(
                ws / f"disclosure_{label}_rep{rep}.json", summary,
                target_model=label, config_name=args.config,
                target_transcript=transcript, rep=rep,
            )
            print(f"  {label} rep {rep}: pass_rate={summary.pass_rate:.3f} "
                  f"({summary.n_passing}/{summary.n_passing + summary.n_failing} scored, "
                  f"{summary.n_errored} errored)")

        run_frozen_bank_sweep(
            targets,
            args.config,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            simulator_model=args.simulator_model,
            simulator_base_url=args.simulator_base_url,
            simulator_api_key=args.simulator_api_key,
            judge_chat_template_kwargs=no_think,
            simulator_chat_template_kwargs=no_think,
            target_purpose=args.target_purpose,
            target_sampling=TargetSampling(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.target_max_tokens,
            ),
            api_key=args.api_key,
            reps=max(1, args.reps),
            attacks_per_vulnerability_type=args.attacks_per_type,
            max_concurrent=args.max_concurrent,
            bank_out=ws / "bank.json",
            bank_in=args.bank_in,
            ctx=args.ctx,
            ngl=args.ngl,
            log_dir=ws,
            server_startup_timeout=args.server_startup_timeout,
            chat_template_kwargs=args.chat_template_kwargs,
            target_timeout=args.target_timeout,
            remote_timeout=args.remote_timeout,
            on_rep=_on_rep,
        )

    if not per_case_path.exists():
        raise SystemExit(
            f"no per-case data at {per_case_path} — pass --models to run a sweep, "
            f"or --per-case pointing at a CSV from scripts/eval_redteam.py"
        )

    # ── Analyze: pair every rung against the reference ───────────────────────
    pooled = _pool_reps(read_per_case_csv(per_case_path))
    if not pooled:
        raise SystemExit(f"{per_case_path} contains no rows")

    order = list(group_per_case_by_model(read_per_case_csv(per_case_path)))
    reference = args.reference or (args.models[0].stem if args.models else order[0])
    if reference not in pooled:
        raise SystemExit(
            f"reference {reference!r} not in {per_case_path} (have: {', '.join(order)})"
        )

    ref_cases = pooled[reference]
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    for model in order:
        if model == reference:
            continue
        result = pair_runs(ref_cases, pooled[model])
        rows.append({"model": model, "reference": reference, **result})
        detail[model] = result
        if result["n_unmatched"]:
            print(f"  ⚠ {model}: {result['n_unmatched']} unmatched case_id(s) — "
                  f"the frozen bank was not shared; this row is not comparable")

    _write_csv(ws / "ladder.csv", rows, LADDER_COLUMNS)

    # Per-category drift, where a headline of ~0 can still hide a real shift.
    cat_rows: list[dict] = []
    for model, result in detail.items():
        for cat, stats in sorted(result["by_category"].items()):
            cat_rows.append({"model": model, "category": cat, **stats})
    _write_csv(
        ws / "ladder_by_category.csv",
        cat_rows,
        ["model", "category", "n_paired", "flip_unsafe", "flip_safe", "net_drift"],
    )

    # Every case where the reference refused and a rung complied — the actual
    # evidence behind n_flip_unsafe, for eyeballing before anyone cites it.
    flips = [
        {"model": m, **f}
        for m, result in detail.items()
        for f in result["flips"]
        if f["direction"] == "unsafe"
    ]
    if flips:
        _write_csv(
            ws / "unsafe_flips.csv",
            flips,
            ["model", "case_id", "category", "vulnerability", "vulnerability_type",
             "attack_method", "attack_class", "input", "actual_output", "reason"],
        )

    (ws / "summary.json").write_text(
        json.dumps({"reference": reference, "rungs": detail}, indent=2, default=str)
    )

    table = _render_table(rows, reference)
    print("\n" + table)
    (ws / "ladder_table.txt").write_text(table + "\n")
    print(
        f"\nLadder CSV  : {ws / 'ladder.csv'}"
        f"\nBy category : {ws / 'ladder_by_category.csv'}"
        f"\nUnsafe flips: {ws / 'unsafe_flips.csv'} ({len(flips)} case(s))"
        f"\nSummary     : {ws / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Red-team safety eval for one or more GGUF quants (or a remote target).

Probes a *target* model with adversarial prompts written by a *simulator* model
and graded by a *judge* model (the deepteam framework). The whole point is
decoupling: an uncensored model writes/grades while the (possibly safety-tuned)
target is scored. The three roles are independent OpenAI-compatible endpoints.

Two target modes:
  * ``--model-path A.gguf [B.gguf ...]`` — each local GGUF is spawned via
    :func:`quant_tuner.eval.server.running_server`, scored, then torn down.
  * ``--base-url URL`` (+ ``--target-model-name``) — an already-served target.

**Multiple targets share one frozen attack bank by default** (``--frozen-bank``,
disable with ``--no-frozen-bank``). deepteam otherwise simulates fresh attacks per
run, so two quants would differ both by the model and by the prompts they happened
to be asked — and at realistic bank sizes the second effect swamps the first. Pass
the F16 reference **first** so the bank is written against the unquantized parent.

Each scored target appends one row to ``--out`` (CSV) and writes a full per-model
JSON (``--json-out`` for a single target, else under ``--json-dir``), plus a
``per_case.csv`` carrying every attempted case keyed by a stable ``case_id`` —
that file is what ``scripts/redteam_ladder.py`` joins to produce the paired
F16-vs-quant comparison.

Requires the redteam extra:  ``uv sync --extra redteam``  (Python ≤ 3.12).

Examples
--------
    # All gemma quants on ONE bank, judge+simulator = a remote uncensored model
    .venv/bin/python scripts/eval_redteam.py \
        --model-path uploads/.../gemma-4-31B-it-F16.gguf \
        --model-path uploads/.../gemma-4-31B-it-IQ3_M.gguf \
        --model-path uploads/.../gemma-4-31B-it-IQ2_M.gguf \
        --config red_team_minimal \
        --judge-model Qwopus3.6-27B-uncensored-Q5_K_M \
        --judge-base-url http://100.102.53.29:1234/v1 \
        --simulator-model Qwopus3.6-27B-uncensored-Q5_K_M \
        --simulator-base-url http://100.102.53.29:1234/v1 \
        --remote-no-think \
        --out out/redteam/gemma_results.csv \
        --json-dir out/redteam --bank-out out/redteam/bank.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.red_team import (  # noqa: E402
    RedTeamRepsSummary,
    RedTeamSummary,
    Target,
    TargetSampling,
    aggregate_reps,
    render_reps_table,
    render_summary,
    run_frozen_bank_sweep,
    write_per_case_csv,
)

# Per-rep rows; full per-category / per-attack / failed-case detail lands in the
# JSON sidecar.
CSV_COLUMNS = [
    "timestamp",
    "model",
    "config",
    "rep",
    "n_tests",
    "n_passing",
    "n_failing",
    "n_errored",
    "pass_rate",
    "run_duration_sec",
]

# The per-case CSV (one row per attempted case per (model, rep), keyed by a stable
# `case_id`) is written by `red_team.write_per_case_csv` — it lives in the library
# because `scripts/redteam_ladder.py` reads it back with the matching reader.

# Aggregated (mean ± stdev across reps) rows.
AGG_COLUMNS = [
    "timestamp",
    "model",
    "config",
    "n_reps",
    "pass_rate_mean",
    "pass_rate_std",
    "per_rep_pass_rate",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ── Target(s): exactly one mode ──────────────────────────────────────────
    p.add_argument(
        "--model-path", type=Path, action="append", default=[],
        help="Local GGUF target (repeatable). Each is spawned, scored, torn down.",
    )
    p.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL of an already-served target (mutually "
        "exclusive with --model-path).",
    )
    p.add_argument(
        "--target-model-name", default="local",
        help="Served model name to send to the --base-url target (default: local).",
    )
    p.add_argument(
        "--target-purpose", default=None,
        help="Free-text description of the target's intended use (sharpens judging).",
    )
    p.add_argument("--api-key", default="sk-no-key-required")

    # ── Judge + simulator endpoints ──────────────────────────────────────────
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-base-url", required=True)
    p.add_argument("--judge-api-key", default="sk-no-key-required")
    p.add_argument("--simulator-model", required=True)
    p.add_argument("--simulator-base-url", required=True)
    p.add_argument("--simulator-api-key", default="sk-no-key-required")
    p.add_argument(
        "--remote-no-think", action="store_true",
        help="Disable thinking on BOTH judge + simulator (sends "
        'chat_template_kwargs={"enable_thinking": false}). Strongly recommended '
        "for reasoning judge/simulator models — a thinking pass costs ~700+ "
        "tokens (minutes) per call.",
    )
    p.add_argument(
        "--judge-chat-template-kwargs", default=None,
        help="JSON forwarded as the judge's per-request chat_template_kwargs "
        "(overrides --remote-no-think for the judge).",
    )
    p.add_argument(
        "--simulator-chat-template-kwargs", default=None,
        help="JSON forwarded as the simulator's per-request chat_template_kwargs "
        "(overrides --remote-no-think for the simulator).",
    )

    # ── Suite config ─────────────────────────────────────────────────────────
    p.add_argument(
        "--config", default="red_team_minimal",
        help="Bare name (resolves against red_team_configs/) or a YAML path.",
    )
    p.add_argument(
        "--attacks-per-type", type=int, default=None,
        help="Override attacks_per_vulnerability_type from the config.",
    )
    p.add_argument("--max-concurrent", type=int, default=None)
    p.add_argument(
        "--reps", type=int, default=1,
        help="Reps per target (server spawned ONCE per local model). Final table "
        "reports mean ± stdev across reps, overall + per category + per vuln.",
    )

    # ── Frozen attack bank (cross-target comparability) ──────────────────────
    p.add_argument(
        "--frozen-bank", dest="frozen_bank", action="store_true", default=True,
        help="Score every target on ONE identical attack bank, simulated against "
        "the FIRST target (pass the F16 reference first). Default: on.",
    )
    p.add_argument(
        "--no-frozen-bank", dest="frozen_bank", action="store_false",
        help="Simulate fresh attacks per target. Absolute rates only — the "
        "cross-target deltas are then confounded by attack-bank variance.",
    )
    p.add_argument(
        "--bank-out", type=Path, default=None,
        help="Write the simulated attack bank here (JSON) for inspection/reuse.",
    )
    p.add_argument(
        "--bank-in", type=Path, default=None,
        help="Replay a bank dumped by an earlier --bank-out run instead of "
        "simulating new attacks. Keeps a new row comparable to old rows.",
    )

    # ── Target sampling (defaults = Gemma's documented values) ───────────────
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)
    p.add_argument("--target-max-tokens", type=int, default=None,
                   help="Max tokens for the target's reply (default: server default).")

    # ── Server (local --model-path only) ─────────────────────────────────────
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--chat-template-kwargs", default=None,
                   help='JSON, forwarded to llama-server --chat-template-kwargs.')
    p.add_argument("--server-startup-timeout", type=float, default=300.0)

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--out", type=Path, required=True, help="CSV to append rows to.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="JSON path for a single target.")
    p.add_argument("--json-dir", type=Path, default=None,
                   help="Directory for per-model JSON (default: dir of --out).")
    return p


def _append_csv_row(out: Path, summary: RedTeamSummary, config: str, rep: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": summary.model,
                "config": config,
                "rep": rep,
                "n_tests": summary.n_tests,
                "n_passing": summary.n_passing,
                "n_failing": summary.n_failing,
                "n_errored": summary.n_errored,
                "pass_rate": summary.pass_rate,
                "run_duration_sec": summary.run_duration_sec,
            }
        )


def _append_agg_row(out: Path, agg: RedTeamRepsSummary, config: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": agg.model,
                "config": config,
                "n_reps": agg.n_reps,
                "pass_rate_mean": agg.pass_rate_mean,
                "pass_rate_std": agg.pass_rate_std,
                "per_rep_pass_rate": ";".join(f"{x:.4f}" for x in agg.per_rep_pass_rate),
            }
        )


def _write_json(path: Path, obj) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(obj), f, indent=2, default=str)


def main() -> int:
    args = _build_arg_parser().parse_args()

    if bool(args.model_path) == bool(args.base_url):
        raise SystemExit("provide exactly one of --model-path (repeatable) or --base-url")

    sampling = TargetSampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        max_tokens=args.target_max_tokens,
    )

    no_think = '{"enable_thinking": false}' if args.remote_no_think else None
    judge_ctk = args.judge_chat_template_kwargs or no_think
    sim_ctk = args.simulator_chat_template_kwargs or no_think

    json_dir = args.json_dir or args.out.parent
    agg_out = args.out.with_name(args.out.stem + "_aggregated" + args.out.suffix)
    per_case_out = args.out.with_name(args.out.stem + "_per_case" + args.out.suffix)
    reps = max(1, args.reps)

    if args.model_path:
        targets = [Target(label=mp.stem, model_path=mp) for mp in args.model_path]
    else:
        targets = [Target(label=args.target_model_name, base_url=args.base_url)]

    if args.frozen_bank and len(targets) > 1:
        print(f"[bank] frozen: attacks simulated against {targets[0].label!r}, "
              f"replayed verbatim against {len(targets) - 1} more target(s)")
    elif not args.frozen_bank:
        print("[bank] NOT frozen — cross-target deltas confound model drift with "
              "attack-bank variance; absolute rates only")

    def _on_rep(label: str, rep: int, summary: RedTeamSummary) -> None:
        print(f"\n----- {label} · rep {rep}/{reps} -----")
        print(render_summary(summary))
        _append_csv_row(args.out, summary, args.config, rep)
        write_per_case_csv(per_case_out, summary, rep)

    sweep_kwargs = dict(
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        simulator_model=args.simulator_model,
        simulator_base_url=args.simulator_base_url,
        simulator_api_key=args.simulator_api_key,
        judge_chat_template_kwargs=judge_ctk,
        simulator_chat_template_kwargs=sim_ctk,
        target_model_name=args.target_model_name,
        target_purpose=args.target_purpose,
        target_sampling=sampling,
        api_key=args.api_key,
        reps=reps,
        attacks_per_vulnerability_type=args.attacks_per_type,
        max_concurrent=args.max_concurrent,
        bank_out=args.bank_out,
        bank_in=args.bank_in,
        ctx=args.ctx,
        ngl=args.ngl,
        log_dir=json_dir,
        server_startup_timeout=args.server_startup_timeout,
        chat_template_kwargs=args.chat_template_kwargs,
        on_rep=_on_rep,
    )

    if args.frozen_bank:
        swept = run_frozen_bank_sweep(targets, args.config, **sweep_kwargs)
    else:
        # One throwaway bank per target: run each target as its own sweep.
        swept = []
        for t in targets:
            swept.extend(run_frozen_bank_sweep([t], args.config, **sweep_kwargs))

    aggregates: list[RedTeamRepsSummary] = []
    for label, rep_summaries in swept:
        agg = aggregate_reps(rep_summaries, label)
        _append_agg_row(agg_out, agg, args.config)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _write_json(json_dir / f"redteam_{label}_agg_{ts}.json", agg)
        aggregates.append(agg)

    table = render_reps_table(aggregates)
    print("\n" + table)
    # Persist the final table next to the CSVs for easy retrieval.
    (json_dir / "redteam_reps_table.txt").write_text(table + "\n")
    print(f"\nPer-rep CSV : {args.out}\nAggregated  : {agg_out}\n"
          f"Per-case    : {per_case_out}\n"
          f"Table       : {json_dir / 'redteam_reps_table.txt'}")
    if args.bank_out:
        print(f"Attack bank : {args.bank_out}")
    if args.frozen_bank and len(aggregates) > 1:
        print(f"\nPaired ladder: scripts/redteam_ladder.py --per-case {per_case_out} "
              f"--reference {aggregates[0].model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Red-team a set of GGUF quants against ONE frozen attack bank.

Unlike ``eval_redteam.py`` (which re-simulates fresh attacks for every target,
so per-category case counts drift model-to-model), this driver builds a single
persistent :class:`~deepteam.red_teamer.RedTeamer`, simulates the attack bank
ONCE against the first target, and replays the *identical* bank to every
subsequent target via ``reuse_simulated_test_cases=True``. Result: every quant
is scored on the same prompts and each category has the same denominator.

Caveat: the 3 multi-turn attacks (linear/crescendo/tree jailbreaking) are
adaptive — their follow-up turns react to each target's replies, so those can't
be byte-identical across models. Their *count* is still fixed (same seeds), so
the denominators still match; only the single-turn prompts are fully identical.

The simulated bank is dumped to ``--bank`` (JSON) after the first target for
inspection / curation. Output CSV + per-model JSON + printed deepteam risk
tables match ``eval_redteam.py`` so ``build_redteam_full_summary.py`` consumes
them unchanged.

Requires the redteam extra:  ``uv sync --extra redteam``.

Example
-------
    .venv/bin/python -u scripts/eval_redteam_frozen.py \
        --model-path uploads/.../gemma-4-31B-it-IQ2_M.gguf \
        --model-path uploads/.../gemma-4-31B-it-IQ3_M.gguf \
        --model-path uploads/.../gemma-4-31B-it-IQ4_XS.gguf \
        --model-path uploads/.../gemma-4-31B-it-Q5_K_S.gguf \
        --config red_team_gemma_full --remote-no-think \
        --judge-model Qwopus3.6-27B-uncensored-Q5_K_M \
        --judge-base-url http://100.102.53.29:1234/v1 \
        --simulator-model Qwopus3.6-27B-uncensored-Q5_K_M \
        --simulator-base-url http://100.102.53.29:1234/v1 \
        --out out/redteam/frozen/gemma_frozen_results.csv \
        --bank out/redteam/frozen/attack_bank.json \
        --json-dir out/redteam/frozen
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
    RedTeamSummary,
    TargetSampling,
    make_red_teamer,
    render_summary,
    run_red_team_eval,
)
from quant_tuner.eval.server import running_server  # noqa: E402

CSV_COLUMNS = [
    "timestamp", "model", "config", "rep", "n_tests", "n_passing",
    "n_failing", "n_errored", "pass_rate", "run_duration_sec",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", type=Path, action="append", default=[], required=True,
                   help="Local GGUF target (repeatable, >=2 to make freezing meaningful).")
    p.add_argument("--target-model-name", default="local")
    p.add_argument("--target-purpose", default=None)
    p.add_argument("--api-key", default="sk-no-key-required")

    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-base-url", required=True)
    p.add_argument("--judge-api-key", default="sk-no-key-required")
    p.add_argument("--simulator-model", required=True)
    p.add_argument("--simulator-base-url", required=True)
    p.add_argument("--simulator-api-key", default="sk-no-key-required")
    p.add_argument("--remote-no-think", action="store_true",
                   help="Disable thinking on judge + simulator (reasoning models).")
    p.add_argument("--judge-chat-template-kwargs", default=None)
    p.add_argument("--simulator-chat-template-kwargs", default=None)

    p.add_argument("--config", default="red_team_gemma_full")
    p.add_argument("--attacks-per-type", type=int, default=None)

    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)
    p.add_argument("--target-max-tokens", type=int, default=None)

    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--chat-template-kwargs", default=None)
    p.add_argument("--server-startup-timeout", type=float, default=300.0)

    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=None,
                   help="JSON path to dump the frozen attack bank (default: <json-dir>/attack_bank.json).")
    p.add_argument("--json-dir", type=Path, default=None)
    return p


def _append_csv_row(out: Path, s: RedTeamSummary, config: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": s.model, "config": config, "rep": 1,
            "n_tests": s.n_tests, "n_passing": s.n_passing, "n_failing": s.n_failing,
            "n_errored": s.n_errored, "pass_rate": s.pass_rate,
            "run_duration_sec": s.run_duration_sec,
        })


def main() -> int:
    args = _build_arg_parser().parse_args()
    json_dir = args.json_dir or args.out.parent
    bank_path = args.bank or (json_dir / "attack_bank.json")

    sampling = TargetSampling(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        min_p=args.min_p, repeat_penalty=args.repeat_penalty,
        max_tokens=args.target_max_tokens,
    )
    no_think = '{"enable_thinking": false}' if args.remote_no_think else None
    judge_ctk = args.judge_chat_template_kwargs or no_think
    sim_ctk = args.simulator_chat_template_kwargs or no_think

    # ONE persistent RedTeamer -> the frozen bank lives on this instance.
    from quant_tuner.eval.red_team import LocalLLM, _chat_template_extra_body
    sim_model = LocalLLM(model=args.simulator_model, base_url=args.simulator_base_url,
                         api_key=args.simulator_api_key,
                         extra_body=_chat_template_extra_body(sim_ctk))
    eval_model = LocalLLM(model=args.judge_model, base_url=args.judge_base_url,
                          api_key=args.judge_api_key,
                          extra_body=_chat_template_extra_body(judge_ctk))
    red_teamer = make_red_teamer(
        simulator_model=sim_model, evaluation_model=eval_model,
        target_purpose=args.target_purpose, max_concurrent=1,
    )

    common = dict(
        config=args.config,
        judge_model=args.judge_model, judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        simulator_model=args.simulator_model, simulator_base_url=args.simulator_base_url,
        simulator_api_key=args.simulator_api_key,
        judge_chat_template_kwargs=judge_ctk, simulator_chat_template_kwargs=sim_ctk,
        target_purpose=args.target_purpose, target_sampling=sampling,
        api_key=args.api_key, red_teamer=red_teamer,
    )
    if args.attacks_per_type is not None:
        common["attacks_per_vulnerability_type"] = args.attacks_per_type

    summaries: list[RedTeamSummary] = []
    for i, mp in enumerate(args.model_path):
        label = mp.stem
        first = i == 0
        tag = "SIMULATE bank" if first else "REUSE frozen bank"
        print(f"\n{'#' * 70}\n# Red-teaming target: {label}  [{tag}]\n{'#' * 70}")
        with running_server(
            mp, ctx=args.ctx, ngl=args.ngl,
            log_path=json_dir / f"server_{label}.log",
            startup_timeout=args.server_startup_timeout,
            chat_template_kwargs=args.chat_template_kwargs,
        ) as url:
            s = run_red_team_eval(
                **common, base_url=url,
                target_model_name=args.target_model_name, model_label=label,
                reuse_simulated_test_cases=not first,
                bank_path=bank_path if first else None,
            )
        summaries.append(s)
        print(render_summary(s))
        _append_csv_row(args.out, s, args.config)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        import json
        (json_dir / f"redteam_{label}_frozen_{ts}.json").write_text(
            json.dumps(asdict(s), indent=2, default=str))
        if first:
            print(f"\n[bank] froze {s.n_tests} cases -> {bank_path}")

    # Sanity: every target must attempt the SAME bank, and — for a fair
    # comparison — score the same denominator. n_tests (full bank) should match
    # across targets; the scored denominator (pass+fail) should too, and only
    # will if errors are ~0 (raise --target-max-tokens/timeout, avoid GPU
    # contention). Errors are shown so a shrunk denominator can't hide.
    print("\n=== frozen-bank per-target counts ===")
    print(f"  {'model':32} {'bank':>5} {'pass':>5} {'fail':>5} {'err':>4} {'scored(denom)':>14}")
    for s in summaries:
        denom = s.n_passing + s.n_failing
        print(f"  {s.model:32} {s.n_tests:>5} {s.n_passing:>5} {s.n_failing:>5} "
              f"{s.n_errored:>4} {denom:>14}")
    banks = {s.n_tests for s in summaries}
    denoms = {s.n_passing + s.n_failing for s in summaries}
    if len(banks) != 1:
        print("  WARNING: bank sizes differ — simulate/reuse problem.")
    elif len(denoms) != 1:
        print("  NOTE: scored denominators differ (errors vary per target). "
              "For a strict common denominator, compare only cases scored on ALL "
              "targets, or re-run with fewer errors (bigger timeout / token cap / free GPU).")
    else:
        print("  OK: identical bank AND identical scored denominator across all targets.")
    print(f"\nCSV : {args.out}\nBank: {bank_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the tool-call benchmark N times per model with stochastic sampling.

Thin wrapper over :mod:`quant_tuner.eval.reps`. For each GGUF:
  1. Spawn ``llama-server`` once (saves the ~30s warmup per rep).
  2. Run ``eval.toolcall.run_toolcall_eval`` ``--reps`` times against it, each
     with a per-rep seed (``--base-seed + rep``).
  3. Tear down the server.

After every model finishes, the aggregator computes mean ± stdev across the
reps for tool_selection_acc, param_acc_mean, and schema_valid_rate. Default
settings: 10 reps × the 8 GGUFs in ``out/omnicoder_q4_k_m/``. Use
``--models`` / ``--reps`` to scope down.

Wall-time estimate at defaults (n=25 holdout, 10 reps, 8 models): ~15 h.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.reps import (
    RepResult,
    run_reps_for_models,
    sampling_extra_cols,
    write_csvs,
)
from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval

_WORK = _REPO / "out" / "omnicoder_q4_k_m"

DEFAULT_MODELS = [
    _WORK / "model-f16.gguf",
    _WORK / "Q4_K_M-none.gguf",
    _WORK / "Q4_K_M-stock_custom.gguf",
    _WORK / "Q4_K_M-stock_wiki.gguf",
    _WORK / "Q4_K_M-stock_mixed.gguf",
    _WORK / "Q4_K_M-hybrid_custom.gguf",
    _WORK / "Q4_K_M-hybrid_wiki.gguf",
    _WORK / "Q4_K_M-hybrid_mixed.gguf",
]

DEFAULT_SAMPLING = Sampling(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    max_tokens=512,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", type=Path, default=DEFAULT_MODELS,
                   help="GGUFs to eval (default: 8 in out/omnicoder_q4_k_m/)")
    p.add_argument("--holdout", type=Path,
                   default=_WORK / "toolcall_holdout.jsonl",
                   help="Per-session JSONL holdout")
    p.add_argument("--reps", type=int, default=10,
                   help="Repetitions per model (default: 10)")
    p.add_argument("--base-seed", type=int, default=1000,
                   help="Sampling seed for rep 0 (subsequent reps add 1)")
    p.add_argument("--results", type=Path,
                   default=_WORK / "toolcall_reps_results.csv")
    p.add_argument("--aggregated", type=Path,
                   default=_WORK / "toolcall_reps_aggregated.csv")
    p.add_argument("--log-dir", type=Path, default=_WORK / "logs")
    p.add_argument("--ctx", type=int, default=32768,
                   help="ctx budget per rollout. Rollouts hit 16k after a few "
                        "tool-call rounds; 32k absorbs the long-tail sessions.")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--max-turns-per-session", type=int, default=10)
    p.add_argument(
        "--chat-template-kwargs",
        type=str,
        default='{"enable_thinking":false}',
        help='Forwarded to llama-server (default: \'{"enable_thinking":false}\' to '
             "disable reasoning on Qwen3-family templates). Pass empty string to omit.",
    )
    p.add_argument("--temperature", type=float, default=None,
                   help="Override sampling.temperature (lower → less variance)")
    p.add_argument("--top-p", type=float, default=None,
                   help="Override sampling.top_p")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override sampling.top_k")
    p.add_argument("--backend", choices=("server", "diffusion"), default="server",
                   help="generation backend: llama-server (default) or the "
                        "DiffusionGemma block-diffusion shim (requires --hf-dir)")
    p.add_argument("--hf-dir", type=Path, default=None,
                   help="HF snapshot dir with the tokenizer/chat template "
                        "(required for --backend diffusion)")
    p.add_argument("--min-p", type=float, default=None,
                   help="Override sampling.min_p")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    assert args.holdout.exists(), f"holdout missing: {args.holdout}"

    sampling = Sampling(
        temperature=args.temperature if args.temperature is not None else DEFAULT_SAMPLING.temperature,
        top_p=args.top_p if args.top_p is not None else DEFAULT_SAMPLING.top_p,
        top_k=args.top_k if args.top_k is not None else DEFAULT_SAMPLING.top_k,
        min_p=args.min_p if args.min_p is not None else DEFAULT_SAMPLING.min_p,
        presence_penalty=DEFAULT_SAMPLING.presence_penalty,
        repetition_penalty=DEFAULT_SAMPLING.repetition_penalty,
        max_tokens=DEFAULT_SAMPLING.max_tokens,
    )

    def eval_one_rep(base_url: str, sampling: Sampling, rep_idx: int) -> dict[str, float]:
        per_turn_log = (
            args.log_dir / f"toolcall_reps_{int(time.time())}_rep{rep_idx:02d}.jsonl"
        )
        summary = run_toolcall_eval(
            holdout=args.holdout,
            base_url=base_url,
            sampling=sampling,
            max_turns_per_session=args.max_turns_per_session,
            per_turn_log=per_turn_log,
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
            "truth_malformed_count": float(summary.truth_quality.get("malformed_truth", 0)),
        }

    def on_rep(model_name: str, rr: RepResult) -> None:
        status = "OK" if rr.error is None else f"FAIL ({rr.error[:60]})"
        sel = rr.metrics.get("tool_selection_acc", float("nan"))
        print(f"    rep {rr.rep + 1}/{args.reps}: {status}  "
              f"sel={sel:.3f}  ({rr.elapsed_sec / 60:.1f} min)", flush=True)

    def on_model(mr) -> None:
        print(f"\n  aggregate ({mr.model}):", flush=True)
        for metric in sorted(mr.aggregate):
            a = mr.aggregate[metric]
            print(f"    {metric:30s}  {a['mean']:.4f} ± {a['stdev']:.4f}  (n={a['n']})",
                  flush=True)

    print(f"=== tool-call reps: {args.reps} × {len(args.models)} models ===", flush=True)
    print(f"sampling:    T={sampling.temperature}  "
          f"top_p={sampling.top_p}  top_k={sampling.top_k}  "
          f"min_p={sampling.min_p}  pp={sampling.presence_penalty}  "
          f"rep_pen={sampling.repetition_penalty}", flush=True)
    print(f"holdout:     {args.holdout}", flush=True)
    print(f"results:     {args.results}  (per-rep) / {args.aggregated}  (aggregated)",
          flush=True)

    server_factory = None
    if args.backend == "diffusion":
        if args.hf_dir is None:
            raise SystemExit("--backend diffusion requires --hf-dir (tokenizer source)")
        from functools import partial

        from quant_tuner.eval.diffusion_server import running_diffusion_server

        server_factory = partial(running_diffusion_server, hf_dir=args.hf_dir)

    t0 = time.time()
    by_model = run_reps_for_models(
        models=list(args.models),
        eval_fn=eval_one_rep,
        reps=args.reps,
        sampling=sampling,
        base_seed=args.base_seed,
        ctx=args.ctx, ngl=args.ngl,
        log_dir=args.log_dir,
        chat_template_kwargs=(args.chat_template_kwargs or None),
        per_rep_callback=on_rep,
        per_model_callback=on_model,
        server_factory=server_factory,
    )

    write_csvs(
        by_model,
        per_rep=args.results,
        aggregated=args.aggregated,
        extra_cols=sampling_extra_cols(sampling),
    )

    total_min = (time.time() - t0) / 60
    print(f"\n=== ALL DONE in {total_min:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

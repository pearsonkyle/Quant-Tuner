#!/usr/bin/env python3
"""Run the MMLU-Pro few-shot eval N times per model with stochastic sampling.

Mirror of :mod:`scripts.run_toolcall_reps`, just pointed at MMLU-Pro. Spawns
one ``llama-server`` per model, runs the eval ``--reps`` times against it
with per-rep seeds (``--base-seed + rep``), aggregates mean ± stdev across
reps, and writes per-rep + aggregated CSVs.

At ``temperature=0`` MMLU-Pro is essentially deterministic and one rep
suffices; at higher temperatures (``--temperature 0.6 --top-p 0.95`` for
example) the reps capture sampling variance just like the tool-call eval.
"""

from __future__ import annotations

import argparse
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

_WORK = _REPO / "out" / "omnicoder_q4_k_m"
_MMLU_DIR = _REPO / "out" / "mmlu_pro"

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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--holdout", type=Path, default=_MMLU_DIR / "holdout.json")
    p.add_argument("--reps", type=int, default=10,
                   help="Repetitions per model (default: 10)")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--results", type=Path, default=_MMLU_DIR / "reps_results.csv")
    p.add_argument("--aggregated", type=Path, default=_MMLU_DIR / "reps_aggregated.csv")
    p.add_argument("--log-dir", type=Path, default=_MMLU_DIR / "logs")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--backend", choices=("server", "diffusion"), default="server",
                   help="generation backend: llama-server (default) or the "
                        "DiffusionGemma block-diffusion shim (requires --hf-dir)")
    p.add_argument("--hf-dir", type=Path, default=None,
                   help="HF snapshot dir with the tokenizer/chat template "
                        "(required for --backend diffusion)")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Default 0.6 — reps only diverge at T>0. Set 0.0 for greedy.")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--chat-template-kwargs",
        type=str,
        default='{"enable_thinking":false}',
        help='Forwarded to llama-server (default: \'{"enable_thinking":false}\' to '
             "disable reasoning on Qwen3-family templates). Pass empty string to omit.",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    assert args.holdout.exists(), f"holdout missing: {args.holdout}"

    sampling = Sampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    def eval_one_rep(base_url: str, smp: Sampling, rep_idx: int) -> dict[str, float]:
        per_sample_log = (
            args.log_dir / f"mmlu_pro_reps_{int(time.time())}_rep{rep_idx:02d}.jsonl"
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
        # Per-subject accuracy as additional metrics, prefixed for safety.
        for subj, s in summary.by_subject.items():
            safe = subj.replace(" ", "_")
            metrics[f"{safe}_accuracy"] = s["accuracy"]
        return metrics

    def on_rep(model_name: str, rr: RepResult) -> None:
        status = "OK" if rr.error is None else f"FAIL ({rr.error[:60]})"
        acc = rr.metrics.get("accuracy", float("nan"))
        print(f"    rep {rr.rep + 1}/{args.reps}: {status}  "
              f"acc={acc:.3f}  ({rr.elapsed_sec / 60:.1f} min)", flush=True)

    def on_model(mr) -> None:
        print(f"\n  aggregate ({mr.model}):", flush=True)
        for metric in sorted(mr.aggregate):
            a = mr.aggregate[metric]
            print(f"    {metric:30s}  {a['mean']:.4f} ± {a['stdev']:.4f}  (n={a['n']})",
                  flush=True)

    print(f"=== mmlu-pro reps: {args.reps} × {len(args.models)} models ===", flush=True)
    print(f"sampling:  T={sampling.temperature}  top_p={sampling.top_p}  "
          f"top_k={sampling.top_k}", flush=True)
    print(f"holdout:   {args.holdout}", flush=True)
    print(f"results:   {args.results} / {args.aggregated}", flush=True)

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
        ctx=args.ctx,
        ngl=args.ngl,
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

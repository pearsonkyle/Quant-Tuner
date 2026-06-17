#!/usr/bin/env python3
"""Run the agentic SWE-rebench benchmark over one or more GGUF quants.

For each model: spawn ``llama-server``, drive mini-swe-agent over the holdout
(one clean Docker container per instance), grade each patch by running its
tests, and write:

    <workspace>/trajectories/<model>/<instance>.traj.json   full conversation
    <workspace>/trajectories/<model>/<instance>.result.json patch + grade + metrics
    <workspace>/results.csv        one row per (model, instance)
    <workspace>/aggregated.csv     one row per model
    <workspace>/summary.json       everything, machine-readable

Defaults to the gemma-4-31B ``qat-Q2_K_S-imatrix`` quant. Requires the
``swebench`` extra (``uv sync --extra swebench``) and a running Docker daemon —
on Apple Silicon the SWE-rebench linux/amd64 images run under emulation (slow).

Example
-------
    PYTHONPATH=src .venv/bin/python scripts/run_swebench_eval.py \
        --holdout out/external/swe-rebench/holdout.jsonl \
        --workspace out/swe-rebench/smoke --progress
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.swebench import run_swebench_eval  # noqa: E402
from quant_tuner.eval.toolcall import Sampling  # noqa: E402
from quant_tuner.experiments.runner import phase  # noqa: E402

_DEFAULT_MODEL = (
    _REPO
    / "uploads"
    / "pearsonkyle"
    / "gemma-4-31B-it-awq-2bit-GGUF"
    / "gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf"
)

_INSTANCE_COLUMNS = [
    "model", "rep", "agent", "instance_id", "repo",
    "resolved", "patch_produced", "patch_chars",
    "tools_used", "tool_errors", "n_model_calls",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "n_fail_to_pass", "n_fail_to_pass_passed",
    "n_pass_to_pass", "n_pass_to_pass_passed",
    "wall_sec", "exit_status", "grade_error", "error",
]

_AGG_COLUMNS = [
    "model", "n_instances", "n_resolved", "n_patched",
    "pass_rate", "patch_rate",
    "mean_tokens", "total_tokens", "mean_steps",
    "tool_error_rate", "mean_wall_sec",
]


def _docker_ok() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20, text=True
        )
        return r.returncode == 0
    except Exception:
        return False


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", type=Path, default=[_DEFAULT_MODEL],
                   help="GGUF quant(s) to evaluate (default: gemma-4-31B qat-Q2_K_S-imatrix)")
    p.add_argument("--holdout", type=Path,
                   default=_REPO / "out" / "external" / "swe-rebench" / "holdout.jsonl")
    p.add_argument("--workspace", type=Path,
                   default=_REPO / "out" / "swe-rebench" / time.strftime("run-%Y%m%d-%H%M%S"))
    p.add_argument("--reps", type=int, default=1, help="Repeats per model (agentic runs are pricey; default 1)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--instance-timeout", type=int, default=7200, help="Wall-clock seconds per instance")
    p.add_argument("--step-timeout", type=int, default=120, help="Seconds per bash command in the container")
    p.add_argument("--max-tokens", type=int, default=8096, help="Max tokens per model call")
    p.add_argument("--ctx", type=int, default=131072, help="llama-server context length")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--temperature", type=float, default=0.25,
                   help="sampling temperature (default 0.25; a little sampling beats greedy in agent loops)")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--spec-type", default=None,
                   help="llama-server speculative decoding, e.g. 'draft-mtp' for a GGUF with a bundled MTP head")
    p.add_argument("--spec-draft-n-max", type=int, default=None,
                   help="max draft tokens per step for --spec-type (Qwen3.6 MTP: 1 is optimal)")
    p.add_argument("--served-model", default="local", help="Model id litellm sends to llama-server")
    p.add_argument("--agent", default="mini-swe", choices=["mini-swe", "openai-agents"],
                   help="agent scaffold driving the local model (default: mini-swe)")
    p.add_argument("--model-class", default=None,
                   help="mini-swe-agent model class (only with --agent mini-swe): 'litellm' "
                        "(default, tool-calling) or 'litellm_textbased' (parses bash from text; "
                        "safer for weak local models)")
    p.add_argument("--chat-template-kwargs", default=None,
                   help="JSON forwarded to llama-server --chat-template-kwargs")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--skip-docker-check", action="store_true")
    return p


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    args = _build_arg_parser().parse_args()

    if not args.holdout.exists():
        print(f"ERROR: holdout not found: {args.holdout}\n"
              f"Build it first:  PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py",
              file=sys.stderr)
        return 1
    if not args.skip_docker_check and not _docker_ok():
        print("ERROR: Docker daemon is not reachable (`docker info` failed).\n"
              "Start Docker Desktop. SWE-rebench grades patches by running tests inside the\n"
              "instance's linux/amd64 image — on Apple Silicon these run under emulation.\n"
              "Pass --skip-docker-check to bypass this guard.",
              file=sys.stderr)
        return 1

    args.workspace.mkdir(parents=True, exist_ok=True)
    sampling = Sampling(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens
    )

    instance_rows: list[dict] = []
    agg_rows: list[dict] = []
    summary: dict = {"workspace": str(args.workspace), "models": {}}

    for model in args.models:
        if not model.exists():
            print(f"[skip] missing model: {model}", flush=True)
            continue
        model_summaries = []
        for rep in range(args.reps):
            traj_dir = args.workspace / "trajectories" / model.stem
            if args.reps > 1:
                traj_dir = traj_dir / f"rep_{rep}"
            with phase(f"swebench {model.name} rep {rep}"):
                s = run_swebench_eval(
                    args.holdout,
                    model_path=model,
                    sampling=sampling,
                    model_label=model.name,
                    served_model=args.served_model,
                    trajectory_dir=traj_dir,
                    max_steps=args.max_steps,
                    instance_timeout=args.instance_timeout,
                    step_timeout=args.step_timeout,
                    max_tokens=args.max_tokens,
                    agent=args.agent,
                    model_class=args.model_class,
                    ctx=args.ctx,
                    ngl=args.ngl,
                    server_log_path=args.workspace / f"server_{model.stem}_rep{rep}.log",
                    chat_template_kwargs=args.chat_template_kwargs,
                    spec_type=args.spec_type,
                    spec_draft_n_max=args.spec_draft_n_max,
                    progress=args.progress,
                )
            for rec in s.per_instance:
                instance_rows.append({"model": model.name, "rep": rep, **rec})
            model_summaries.append(s)
            print(f"  → {model.name} rep {rep}: pass_rate={s.pass_rate:.2f} "
                  f"patch_rate={s.patch_rate:.2f} mean_tokens={s.mean_tokens:.0f} "
                  f"mean_steps={s.mean_steps:.1f} tool_err={s.tool_error_rate:.2f}", flush=True)

        if model_summaries:
            # Aggregate the per-instance records across reps for the model row.
            all_recs = [r for s in model_summaries for r in s.per_instance]
            from quant_tuner.eval.swebench import _aggregate
            merged = _aggregate(model.name, all_recs)
            agg_rows.append({k: getattr(merged, k) for k in _AGG_COLUMNS})
            summary["models"][model.name] = {
                "aggregate": {k: getattr(merged, k) for k in _AGG_COLUMNS},
                "reps": [asdict(s) for s in model_summaries],
            }

    _write_csv(args.workspace / "results.csv", _INSTANCE_COLUMNS, instance_rows)
    _write_csv(args.workspace / "aggregated.csv", _AGG_COLUMNS, agg_rows)
    (args.workspace / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote:\n  {args.workspace/'results.csv'}\n  {args.workspace/'aggregated.csv'}\n"
          f"  {args.workspace/'summary.json'}\n  {args.workspace/'trajectories'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

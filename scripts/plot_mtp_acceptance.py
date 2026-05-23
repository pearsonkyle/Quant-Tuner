"""Plot MTP draft acceptance rate vs. --spec-draft-n-max for each 9B model.

Sweeps n_max in [1, 2, 3] for each (model, mtp_variant) pair, collecting
draft acceptance rate and mean tok/s from llama-server. Produces two outputs:

  out/benchmark_9b_iq3s/mtp_acceptance.json  — raw numbers
  out/benchmark_9b_iq3s/mtp_acceptance.png   — line chart (acceptance rate vs n_max)

Each line in the chart represents one (model × mtp_variant) combination:
  - <model> / vanilla-mtp   — donor Qwen MTP weights, no fine-tuning
  - <model> / trained-mtp   — domain-adapted weights from train_mtp_head.py

Usage::

    # All three models, n_max sweep 1–3, 3 reps each
    uv run python scripts/plot_mtp_acceptance.py \\
        --workspace out/benchmark_9b_iq3s \\
        --holdout-jsonl logtrain.jsonl

    # Single model, quick 2-rep check
    uv run python scripts/plot_mtp_acceptance.py \\
        --workspace out/benchmark_9b_iq3s \\
        --holdout-jsonl logtrain.jsonl \\
        --models tesslate --reps 2

    # Dry run — print plan and exit
    uv run python scripts/plot_mtp_acceptance.py \\
        --workspace out/benchmark_9b_iq3s \\
        --holdout-jsonl logtrain.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.eval.server import running_server  # noqa: E402
from quant_tuner.experiments import log, phase  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry — maps slug → (IQ3_S gguf name, trained-mtp gguf name or None)
# ---------------------------------------------------------------------------

MODELS = {
    "qwen":     "qwen",
    "jackrong": "jackrong",
    "tesslate": "tesslate",
}


def _gguf_path(workspace: Path, slug: str, variant: str) -> Path | None:
    """Resolve the GGUF path for (slug, variant).

    variant is one of:
      "vanilla"       — IQ3_S-custom.gguf built with donor MTP weights
      "trained"       — IQ3_S-trained.gguf built after train_mtp_head.py (may not exist)
      "trained-noisy" — IQ3_S-trained-noisy.gguf trained with IQ3_S quantization noise
    """
    base = workspace / slug
    if variant == "vanilla":
        p = base / "IQ3_S-custom.gguf"
        return p if p.exists() else None
    if variant == "trained":
        p = base / "IQ3_S-trained.gguf"
        return p if p.exists() else None
    if variant == "trained-noisy":
        p = base / "IQ3_S-trained-noisy.gguf"
        return p if p.exists() else None
    if variant == "fp16":
        p = base / "model-f16.gguf"
        return p if p.exists() else None
    return None


# ---------------------------------------------------------------------------
# Prompt loading (same logic as bench_mtp_speed.py)
# ---------------------------------------------------------------------------

def _load_holdout_prompts(jsonl_path: Path, max_prompts: int = 30) -> list[str]:
    sessions: list[list[dict]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "messages" in obj:
                sessions.append(obj["messages"])

    holdout_n = max(1, math.ceil(len(sessions) * 0.1))
    holdout_sessions = sessions[-holdout_n:]

    prompts: list[str] = []
    for messages in holdout_sessions:
        for msg in messages:
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                content = content.strip()
                if content:
                    prompts.append(content)
                    if len(prompts) >= max_prompts:
                        return prompts
    return prompts


# ---------------------------------------------------------------------------
# Single llama-server run
# ---------------------------------------------------------------------------

def _run_one(base_url: str, prompt: str, max_tokens: int) -> dict | None:
    """Fire one chat completion; return None on 400 (e.g. context overflow)."""
    # Truncate very long prompts to avoid context overflow at ctx=8192.
    # At ~3 chars/token, 6000 tokens ≈ 18k chars; leave headroom for max_tokens.
    prompt = prompt[:12_000]
    try:
        r = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "stream": False,
            },
            timeout=600,
        )
        if r.status_code == 400:
            log(f"  WARNING: 400 on prompt (len={len(prompt)}) — skipping")
            return None
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        log(f"  WARNING: request error ({e}) — skipping")
        return None


def _bench_n_max(
    gguf: Path,
    n_max: int,
    prompts: list[str],
    reps: int,
    max_tokens: int,
    ctx: int,
    log_dir: Path,
    label: str,
) -> dict:
    """Run reps prompts with spec-draft-n-max=n_max; return summary dict."""
    log_path = log_dir / f"llama-server.{label}.n{n_max}.log"
    rates: list[float] = []
    n_drafted = 0
    n_accepted = 0

    with running_server(
        gguf,
        ctx=ctx,
        log_path=log_path,
        chat_template_kwargs='{"enable_thinking":false}',
        spec_type="draft-mtp",
        spec_draft_n_max=n_max,
        startup_timeout=300.0,
    ) as base_url:
        log(f"  warmup {label} n_max={n_max}...")
        _run_one(base_url, "Say hi.", max_tokens=8)

        completed = 0
        for prompt in prompts:
            if completed >= reps:
                break
            resp = _run_one(base_url, prompt, max_tokens=max_tokens)
            if resp is None:
                continue
            timings = resp.get("timings", {}) or {}
            tps = timings.get("predicted_per_second")
            drafted = timings.get("draft_n", 0) or 0
            accepted = timings.get("draft_n_accepted", 0) or 0
            n_drafted += drafted
            n_accepted += accepted
            completed += 1
            accept_frac = (accepted / drafted * 100) if drafted else float("nan")
            log(
                f"  rep {completed}/{reps}  tps={tps:.2f}  "
                f"draft_acc={accepted}/{drafted} ({accept_frac:.1f}%)"
                if tps is not None else f"  rep {completed}/{reps}  timings missing"
            )
            if tps is not None:
                rates.append(tps)

    mean_tps = statistics.mean(rates) if rates else float("nan")
    accept_rate = (n_accepted / n_drafted) if n_drafted else None
    log(
        f"  n_max={n_max}  mean {mean_tps:.2f} tok/s  "
        f"acceptance={n_accepted}/{n_drafted} "
        f"({accept_rate*100:.1f}% per token)" if accept_rate is not None
        else f"  n_max={n_max}  mean {mean_tps:.2f} tok/s  no draft stats"
    )
    return {
        "n_max": n_max,
        "mean_tps": mean_tps,
        "draft_accepted": n_accepted,
        "draft_total": n_drafted,
        "accept_rate": accept_rate,
        "reps": len(rates),
    }


def _bench_baseline(
    gguf: Path,
    prompts: list[str],
    reps: int,
    max_tokens: int,
    ctx: int,
    log_dir: Path,
    label: str,
) -> dict:
    """True no-MTP baseline."""
    log_path = log_dir / f"llama-server.{label}.baseline.log"
    rates: list[float] = []

    with running_server(
        gguf,
        ctx=ctx,
        log_path=log_path,
        chat_template_kwargs='{"enable_thinking":false}',
        spec_type=None,
        spec_draft_n_max=None,
        startup_timeout=300.0,
    ) as base_url:
        log(f"  warmup {label} baseline...")
        _run_one(base_url, "Say hi.", max_tokens=8)
        completed = 0
        for prompt in prompts:
            if completed >= reps:
                break
            resp = _run_one(base_url, prompt, max_tokens=max_tokens)
            if resp is None:
                continue
            timings = resp.get("timings", {}) or {}
            tps = timings.get("predicted_per_second")
            completed += 1
            if tps is not None:
                rates.append(tps)
                log(f"  rep {completed}/{reps}  tps={tps:.2f}")

    return {
        "n_max": 0,
        "mean_tps": statistics.mean(rates) if rates else float("nan"),
        "draft_accepted": 0,
        "draft_total": 0,
        "accept_rate": None,
        "reps": len(rates),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results: list[dict], out_png: Path) -> None:
    """Render acceptance rate vs n_max line chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available — skipping PNG (pip install matplotlib)")
        return

    fig, (ax_acc, ax_tps) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MTP speculative decoding: acceptance rate & speed vs draft depth\n"
                 "Qwen3.5-9B family @ IQ3_S (hybrid_custom imatrix)", fontsize=11)

    colors = {
        "qwen":     "#4e79a7",
        "jackrong": "#f28e2b",
        "tesslate": "#e15759",
    }
    styles = {
        "vanilla":       "-",
        "trained":       "--",
        "trained-noisy": "-.",
        "fp16":          ":",
    }
    markers = {
        "vanilla":       "o",
        "trained":       "s",
        "trained-noisy": "^",
        "fp16":          "D",
    }

    for row in results:
        slug = row["slug"]
        variant = row["variant"]
        sweep = [r for r in row["sweep"] if r["n_max"] > 0 and r["accept_rate"] is not None]
        if not sweep:
            continue
        xs = [r["n_max"] for r in sweep]
        acc_ys = [r["accept_rate"] * 100 for r in sweep]
        tps_ys = [r["mean_tps"] for r in sweep]
        label = f"{slug} / {variant}"
        color = colors.get(slug, "gray")
        ls = styles.get(variant, "-")
        mk = markers.get(variant, "o")

        ax_acc.plot(xs, acc_ys, color=color, linestyle=ls, marker=mk, label=label)
        ax_tps.plot(xs, tps_ys, color=color, linestyle=ls, marker=mk, label=label)

        # Baseline horizontal dashes for tok/s
        baseline_tps = row.get("baseline_tps")
        if baseline_tps is not None and not math.isnan(baseline_tps):
            ax_tps.axhline(
                baseline_tps, color=color,
                linestyle=":", alpha=0.5,
                label=f"{slug} / {variant} baseline",
            )

    ax_acc.set_xlabel("--spec-draft-n-max")
    ax_acc.set_ylabel("Per-token draft acceptance rate (%)")
    ax_acc.set_title("Acceptance rate vs draft depth")
    ax_acc.set_xticks([1, 2, 3])
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    ax_tps.set_xlabel("--spec-draft-n-max")
    ax_tps.set_ylabel("Mean decode tok/s")
    ax_tps.set_title("Throughput vs draft depth")
    ax_tps.set_xticks([1, 2, 3])
    ax_tps.legend(fontsize=8)
    ax_tps.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150)
    log(f"  wrote {out_png}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path,
                        default=REPO / "out/benchmark_9b_iq3s",
                        help="Root workspace (contains qwen/, jackrong/, tesslate/ subdirs)")
    parser.add_argument("--holdout-jsonl", type=Path, default=REPO / "logtrain.jsonl",
                        help="logtrain.jsonl for extracting holdout prompts")
    parser.add_argument("--models", type=str, default="qwen,jackrong,tesslate",
                        help="Comma-separated subset of model slugs")
    parser.add_argument("--variants", type=str, default="vanilla,trained,trained-noisy",
                        help="Comma-separated: vanilla, trained, trained-noisy "
                             "(trained requires IQ3_S-trained.gguf; "
                             "trained-noisy requires IQ3_S-trained-noisy.gguf)")
    parser.add_argument("--n-max-values", type=str, default="1,2,3",
                        help="Comma-separated n_max values to sweep")
    parser.add_argument("--reps", type=int, default=3,
                        help="Prompts per (model, variant, n_max) within each trial")
    parser.add_argument("--trials", type=int, default=1,
                        help="Independent trials to run; mean ± stdev reported across trials "
                             "(each trial spawns fresh servers)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slugs = [s.strip() for s in args.models.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    n_max_values = [int(x) for x in args.n_max_values.split(",")]

    log_dir = args.workspace / "eval" / "mtp_acceptance"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts: list[str] = []
    if args.holdout_jsonl and args.holdout_jsonl.exists():
        prompts = _load_holdout_prompts(args.holdout_jsonl, max_prompts=max(args.reps * 4, 20))
        log(f"Loaded {len(prompts)} prompts from holdout split of {args.holdout_jsonl.name}")
    if not prompts:
        # Fallback built-in prompts
        prompts = [
            "Write a Python function to compute the edit distance between two strings.",
            "Explain the difference between a mutex and a semaphore.",
            "Write a SQL query to find the top 5 users by order total.",
            "Implement binary search in Rust with a brief docstring.",
            "Summarize the CAP theorem and when each trade-off is appropriate.",
        ]

    # Print plan
    log("=== MTP acceptance sweep plan ===")
    for slug in slugs:
        for variant in variants:
            gguf = _gguf_path(args.workspace, slug, variant)
            status = str(gguf) if gguf else "NOT FOUND — skipping"
            log(f"  {slug:10s} / {variant:14s}  {status}")
    log(f"  n_max sweep: {n_max_values}   reps/trial: {args.reps}   trials: {args.trials}")
    log(f"  prompts: {len(prompts)}")

    if args.dry_run:
        log("--dry-run: exiting without running servers")
        return 0

    all_results: list[dict] = []

    for slug in slugs:
        for variant in variants:
            gguf = _gguf_path(args.workspace, slug, variant)
            if gguf is None:
                log(f"  [{slug}/{variant}] GGUF not found — skipping")
                continue

            log(f"\n=== {slug} / {variant} ===")

            # Collect one measurement dict per trial
            trial_baselines: list[float] = []
            # n_max → list of (tps, accept_rate) across trials
            trial_sweep: dict[int, list[tuple[float, float]]] = {n: [] for n in n_max_values}

            for trial in range(args.trials):
                t_label = f"{slug}_{variant}" if args.trials == 1 else f"{slug}_{variant}_t{trial}"
                trial_tag = "" if args.trials == 1 else f" [trial {trial + 1}/{args.trials}]"

                with phase(f"{slug}/{variant}{trial_tag} baseline"):
                    bl = _bench_baseline(
                        gguf, prompts, args.reps, args.max_tokens, args.ctx,
                        log_dir, t_label,
                    )
                trial_baselines.append(bl["mean_tps"])

                for n_max in n_max_values:
                    with phase(f"{slug}/{variant}{trial_tag} n_max={n_max}"):
                        result = _bench_n_max(
                            gguf, n_max, prompts, args.reps, args.max_tokens, args.ctx,
                            log_dir, t_label,
                        )
                    ar = result.get("accept_rate")
                    tps = result.get("mean_tps", float("nan"))
                    trial_sweep[n_max].append((tps, ar if ar is not None else float("nan")))

            # Aggregate across trials
            def _mean(xs: list[float]) -> float:
                valid = [x for x in xs if x == x]  # drop NaN
                return statistics.mean(valid) if valid else float("nan")

            def _stdev(xs: list[float]) -> float:
                valid = [x for x in xs if x == x]
                return statistics.stdev(valid) if len(valid) >= 2 else 0.0

            row: dict = {
                "slug":    slug,
                "variant": variant,
                "gguf":    str(gguf),
                "trials":  args.trials,
                "baseline_tps":       _mean(trial_baselines),
                "baseline_tps_stdev": _stdev(trial_baselines),
                "sweep": [],
            }

            for n_max in n_max_values:
                tps_list = [t for t, _ in trial_sweep[n_max]]
                ar_list  = [a for _, a in trial_sweep[n_max]]
                row["sweep"].append({
                    "n_max":             n_max,
                    "mean_tps":          _mean(tps_list),
                    "mean_tps_stdev":    _stdev(tps_list),
                    "accept_rate":       _mean(ar_list),
                    "accept_rate_stdev": _stdev(ar_list),
                    # token counts summed across all trials for reference
                    "reps": len(tps_list),
                })

            all_results.append(row)

    # Save JSON — merge with any existing data so partial variant runs don't clobber other results
    out_json = args.workspace / "mtp_acceptance.json"
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            new_keys = {(r["slug"], r["variant"]) for r in all_results}
            merged = [r for r in existing if (r["slug"], r["variant"]) not in new_keys]
            merged.extend(all_results)
            all_results = merged
        except Exception:
            pass  # corrupt JSON — just overwrite
    out_json.write_text(json.dumps(all_results, indent=2))
    log(f"\nwrote {out_json}")

    # Print summary table
    multi = any(r.get("trials", 1) > 1 for r in all_results)
    col_w = 20 if multi else 13
    header = f"{'model':<12} {'variant':<14} {'baseline tok/s':>{col_w}}"
    for n in n_max_values:
        header += f"  {'n='+str(n)+' acc%':>{col_w}}  {'tok/s':>{col_w}}"
    log("\n" + "=" * len(header))
    log(header)
    log("=" * len(header))
    for row in all_results:
        bl     = row.get("baseline_tps", float("nan"))
        bl_sd  = row.get("baseline_tps_stdev", 0.0)
        bl_str = f"{bl:.1f} ± {bl_sd:.1f}" if multi and bl_sd else f"{bl:.2f}"
        line   = f"{row['slug']:<12} {row['variant']:<14} {bl_str:>{col_w}}"
        for n in n_max_values:
            match = next((r for r in row["sweep"] if r["n_max"] == n), None)
            if match and match.get("accept_rate") is not None:
                ar   = match["accept_rate"]
                ar_s = match.get("accept_rate_stdev", 0.0) or 0.0
                tps  = match["mean_tps"]
                tps_s = match.get("mean_tps_stdev", 0.0) or 0.0
                if multi and ar_s:
                    acc_str = f"{ar*100:.1f}±{ar_s*100:.1f}%"
                    tps_str = f"{tps:.1f}±{tps_s:.1f}"
                else:
                    acc_str = f"{ar*100:.1f}%"
                    tps_str = f"{tps:.2f}"
                line += f"  {acc_str:>{col_w}}  {tps_str:>{col_w}}"
            else:
                line += f"  {'—':>{col_w}}  {'—':>{col_w}}"
        log(line)
    log("=" * len(header))

    # Plot
    out_png = args.workspace / "mtp_acceptance.png"
    _plot(all_results, out_png)

    return 0


if __name__ == "__main__":
    sys.exit(main())

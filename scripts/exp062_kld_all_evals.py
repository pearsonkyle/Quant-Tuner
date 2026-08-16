"""Measure the exp-062 IQ2_M rung against ALL SIX eval distributions.

Why this exists: the pipeline's `bench` stage covers only `eval_corpus` (external)
and `eval_tools_corpus` (tools), but the model card publishes six. Shipping the new
AWQ file while four of its six KLD rows still carry the OLD imatrix build's numbers
would misrepresent the artifact — the rows would silently describe a different file.

Each eval is a SEPARATE distribution with its own FP16 baseline; they are never
concatenated, and a number from one is not comparable to a number from another.

`external` and `tools` are re-measured here even though the pipeline already did
them — same model, same corpus, same baseline, so a disagreement would mean
something is wrong. It is a free reproducibility check.

⚠️ The four chat-templated evals (`tools`, `agentic`, `broad`, `cal8k`) are
quant-vs-quant ONLY: llama-perplexity has no `--parse-special`, so chat markers
tokenize as ordinary BPE and absolute PPL is off-distribution there. KLD and
top_p stay valid.

    PYTHONPATH=src .venv/bin/python scripts/exp062_kld_all_evals.py \
        --quant out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf \
        --out out/exp-062-32k/eval/kld_all_evals.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import kld  # noqa: E402

CORPORA = REPO / "out/exp-060-32k/corpora"
BASELINES = REPO / "out/exp-060-32k"

# eval name -> (corpus file, baseline file). Pinned to exp-060 on purpose: holding
# the eval side fixed is what makes the new calibration corpus the only variable.
EVALS = {
    "external": ("corpus.eval.txt", "baseline.external.kld"),
    "general": ("corpus.eval.general.txt", "baseline.general.kld"),
    "tools": ("corpus.eval.tools.txt", "baseline.tools.kld"),
    "agentic": ("corpus.eval.agentic.txt", "baseline.agentic.kld"),
    "broad": ("corpus.eval.broad.txt", "baseline.broad.kld"),
    "cal8k": ("corpus.eval.cal8k.txt", "baseline.cal8k.kld"),
}

FIELDS = ["eval", "ppl", "ppl_ratio", "mean_kld", "median_kld", "same_top_p", "rms_dp"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--eval-ctx", type=int, default=8192,
                    help="MUST stay 8192 — the FP16 baselines were built at 8192 and "
                         "a different ctx would need all 143 GB of them regenerated")
    ap.add_argument("--evals", nargs="+", choices=sorted(EVALS), default=sorted(EVALS))
    a = ap.parse_args()

    if not a.quant.exists():
        print(f"FATAL: no quant at {a.quant}", file=sys.stderr)
        return 1
    logs = a.log_dir or (a.out.parent / "logs-kld")
    logs.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in a.evals:
        corpus_name, baseline_name = EVALS[name]
        corpus, baseline = CORPORA / corpus_name, BASELINES / baseline_name
        for p in (corpus, baseline):
            if not p.exists():
                print(f"  {name:9s} SKIP — missing {p}")
                break
        else:
            print(f"  {name:9s} running …", flush=True)
            t0 = time.time()
            m = kld.evaluate(a.quant, corpus, baseline, ctx=a.eval_ctx,
                             log=logs / f"kld-{name}.log")
            rows.append({
                "eval": name, "ppl": m.ppl, "ppl_ratio": m.ppl_ratio,
                "mean_kld": m.mean_kld, "median_kld": m.median_kld,
                "same_top_p": m.same_top_p, "rms_dp": m.rms_dp,
            })
            print(f"  {name:9s} ppl={m.ppl}  mean_kld={m.mean_kld}  "
                  f"median_kld={m.median_kld}  same_top_p={m.same_top_p}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} evals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

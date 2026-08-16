"""Assemble the exp-062 decision table: new AWQ rungs vs the shipped ladder.

Pulls the four artifacts the decision actually rests on into one view:

  * KLD / PPL / bpw   — out/exp-062-awq-*/results.csv vs out/exp-060-32k/results.csv
  * tool-call         — out/exp-062-32k/eval/toolcall_smoke.csv
  * agentic (SWE)     — /workspace/swe-mimic/swe_mimic_results_exp062.csv

Reading guide, because the metrics disagree on purpose:

  **Rank on tool-call accuracy; treat KLD as a guardrail, not the verdict.** The
  gemma-4-31B precedent that motivated this whole approach was WORSE on median KLD
  (1.804 vs 1.571) and on top_p (43.9% vs 46.6%) and still won by +54% on tool
  arguments (0.171 -> 0.263). A rung that improves KLD while losing tool accuracy
  is not an improvement for this repo's users.

  **Respect the noise floor.** Greedy decoding over 92 smoke turns puts the
  binomial SE near p=0.3 at ~4.8pp; the 174-turn full holdout is ~3.3pp. A +0.01
  "win" is noise. Ship on a gap of the gemma order (+0.09), or run more seeds.
  This script prints the SE next to every tool-call delta so the comparison cannot
  be read without it.

  **The SWE column is one instance.** It is a smoke test for "can it still drive an
  agent loop and call tools", not a benchmark. resolved=1 vs 0 on a single instance
  is not a ranking signal; a *crash* or a collapse in tool-error rate is.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNGS = ("IQ2_M", "IQ3_M", "IQ4_XS")

SHIPPED_RESULTS = REPO / "out/exp-060-32k/results.csv"
NEW_RESULTS = {r: REPO / f"out/exp-062-awq-{r.lower().replace('_', '')}/results.csv"
               for r in RUNGS}
SMOKE = REPO / "out/exp-062-32k/eval/toolcall_smoke.csv"
SWE = Path("/workspace/swe-mimic/swe_mimic_results_exp062.csv")


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    v = (row or {}).get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rung_of(label: str) -> str | None:
    for r in RUNGS:
        if re.search(rf"\b{r}\b", label):
            return r
    return None


def _bench_by_rung(rows: list[dict]) -> dict[str, dict]:
    """Keep the external-eval row per rung (the published headline footing)."""
    out: dict[str, dict] = {}
    for row in rows:
        label = row.get("model", "")
        rung = _rung_of(label)
        if rung is None:
            continue
        # Several eval distributions share a results.csv; the ladder's headline
        # numbers are the external ones. Take rows that are explicitly external,
        # plus untagged rows (a single-eval workspace writes no eval= tag).
        if "eval=" not in label or "eval=external" in label:
            out[rung] = row
    return out


def _se(p: float | None, n: int | None) -> float | None:
    if p is None or not n:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def main() -> int:
    shipped = _bench_by_rung(_rows(SHIPPED_RESULTS))
    new = {r: _bench_by_rung(_rows(p)).get(r) for r, p in NEW_RESULTS.items()}

    smoke = {row["model"]: row for row in _rows(SMOKE)}
    swe: dict[str, dict] = {}
    for row in _rows(SWE):
        swe[row.get("label", "")] = row

    print("=" * 78)
    print("exp-062: AWQ + tool-dense corpus + safe_v2 template  vs  shipped ladder")
    print("=" * 78)

    print("\n--- 1. Distributional (guardrail): external eval, KLD vs FP16 ---")
    print(f"  {'rung':8s} {'':10s} {'bpw':>7s} {'ppl':>9s} {'mean_kld':>9s} "
          f"{'med_kld':>9s} {'same_top_p':>10s}")
    for r in RUNGS:
        for tag, row in (("shipped", shipped.get(r)), ("new-awq", new.get(r))):
            if not row:
                print(f"  {r:8s} {tag:10s} {'—  not available yet':>40s}")
                continue
            print(f"  {r:8s} {tag:10s} {_f(row,'bpw') or 0:7.3f} "
                  f"{_f(row,'ppl') or 0:9.3f} {_f(row,'mean_kld') or 0:9.4f} "
                  f"{_f(row,'median_kld') or 0:9.4f} "
                  f"{_f(row,'same_top_p') or 0:10.4f}")
        s, n = shipped.get(r), new.get(r)
        if s and n:
            dk = (_f(n, "mean_kld") or 0) - (_f(s, "mean_kld") or 0)
            db = (_f(n, "bpw") or 0) - (_f(s, "bpw") or 0)
            verdict = "better" if dk < 0 else "worse"
            print(f"  {'':8s} {'delta':10s} {db:+7.3f} {'':9s} {dk:+9.4f}"
                  f"   ({verdict} KLD — guardrail only, not the verdict)")
        print()

    print("--- 2. Tool-call (THE decision metric): greedy, 25-session holdout ---")
    print(f"  {'rung':8s} {'':12s} {'tool_sel':>9s} {'param':>8s} {'schema':>8s} {'n':>5s}")
    for r in RUNGS:
        pairs = [("new-awq", smoke.get(f"{r}-awq-new")),
                 ("shipped", smoke.get(f"{r}-shipped"))]
        for tag, row in pairs:
            if not row:
                print(f"  {r:8s} {tag:12s} {'— not run yet':>32s}")
                continue
            print(f"  {r:8s} {tag:12s} {_f(row,'tool_selection_acc') or 0:9.3f} "
                  f"{_f(row,'param_acc_mean') or 0:8.3f} "
                  f"{_f(row,'schema_valid_rate') or 0:8.3f} "
                  f"{int(_f(row,'n_scored') or 0):5d}")
        a, b = smoke.get(f"{r}-awq-new"), smoke.get(f"{r}-shipped")
        if a and b:
            pa, pb = _f(a, "param_acc_mean"), _f(b, "param_acc_mean")
            na = int(_f(a, "n_scored") or 0)
            d = (pa or 0) - (pb or 0)
            se = _se(pb, na)
            # SE of a difference of two proportions on the SAME turns is not
            # sqrt(2)*se — the runs are paired and highly correlated. sqrt(2)*se
            # is the conservative unpaired bound; quote it rather than understate.
            bound = (se or 0) * math.sqrt(2)
            call = "NOISE" if abs(d) < 2 * bound else ("WIN" if d > 0 else "REGRESSION")
            print(f"  {'':8s} {'delta':12s} {d:+9.3f}   "
                  f"(unpaired 1-SE bound ~{bound:.3f}) -> {call}")
        print()

    print("--- 3. Agentic smoke (SWE-mimic, one instance — liveness, not ranking) ---")
    if not swe:
        print("  not run yet")
    else:
        print(f"  {'label':14s} {'resolved':>8s} {'patch':>6s} {'f2p':>7s} {'p2p':>7s} "
              f"{'steps':>6s} {'tool_err':>8s} {'exit':>10s}")
        for label, row in swe.items():
            print(f"  {label:14s} {row.get('resolved',''):>8s} "
                  f"{row.get('patch_produced',''):>6s} "
                  f"{row.get('f2p_passed','')}/{row.get('f2p_total',''):<5s} "
                  f"{row.get('p2p_passed','')}/{row.get('p2p_total',''):<5s} "
                  f"{row.get('steps',''):>6s} {row.get('tool_err_rate',''):>8s} "
                  f"{row.get('exit_status',''):>10s}")

    print("\n--- Verdict rule ---")
    print("  Swap a rung ONLY if tool-call param_acc improves by more than the")
    print("  noise bound above AND schema validity does not regress AND the SWE")
    print("  smoke still completes. KLD moving the wrong way is acceptable — the")
    print("  gemma precedent did exactly that and was the right call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

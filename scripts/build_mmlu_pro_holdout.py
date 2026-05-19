#!/usr/bin/env python3
"""Sample an MMLU-Pro holdout from ``TIGER-Lab/MMLU-Pro``.

Writes a JSON file with two sections:

  * ``shots[subject]`` — the first ``--n-shot`` validation/dev examples for
    the subject, used as few-shot demonstrations.
  * ``samples`` — ``--n-per-subject`` test questions randomly sampled (with
    a seeded RNG) from each requested subject, concatenated.

Default: 25 samples each from ``computer science`` and ``math``, 2-shot.
Tweak with ``--subjects``, ``--n-per-subject``, ``--n-shot``, ``--seed``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.mmlu_pro import (
    MmluProHoldout,
    MmluProSample,
    save_holdout,
)


def _to_sample(row: dict) -> MmluProSample:
    """Map a Hugging Face MMLU-Pro row to an :class:`MmluProSample`."""
    return MmluProSample(
        question_id=row["question_id"],
        subject=row["category"],
        question=row["question"],
        options=list(row["options"]),
        answer=row["answer"],
        answer_index=int(row["answer_index"]),
        cot_content=row.get("cot_content", "") or "",
    )


def _sample_subject(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subjects", nargs="+",
        default=["computer science", "math"],
        help="MMLU-Pro categories to include (default: computer science math)",
    )
    p.add_argument("--n-per-subject", type=int, default=25)
    p.add_argument("--n-shot", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=Path,
        default=_REPO / "out" / "mmlu_pro" / "holdout.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--dataset", default="TIGER-Lab/MMLU-Pro",
        help="HF dataset id (default: TIGER-Lab/MMLU-Pro)",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Heavy import deferred so --help doesn't pay the cost.
    from datasets import load_dataset

    print(f"Loading {args.dataset} (test + validation)…", flush=True)
    ds_test = load_dataset(args.dataset, split="test")
    ds_dev = load_dataset(args.dataset, split="validation")

    requested = {s.lower() for s in args.subjects}
    test_by_subj: dict[str, list[dict]] = {}
    dev_by_subj: dict[str, list[dict]] = {}
    for row in ds_test:
        if row["category"].lower() in requested:
            test_by_subj.setdefault(row["category"], []).append(row)
    for row in ds_dev:
        if row["category"].lower() in requested:
            dev_by_subj.setdefault(row["category"], []).append(row)

    canonical_subjects = sorted(test_by_subj.keys())
    if not canonical_subjects:
        print(f"ERROR: none of {args.subjects!r} matched any category in {args.dataset}",
              file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    shots: dict[str, list[MmluProSample]] = {}
    samples: list[MmluProSample] = []

    for subj in canonical_subjects:
        dev_rows = dev_by_subj.get(subj, [])
        if len(dev_rows) < args.n_shot:
            print(f"WARN: only {len(dev_rows)} dev rows for {subj!r}; "
                  f"requested {args.n_shot} shots", file=sys.stderr)
        shots[subj] = [_to_sample(r) for r in dev_rows[: args.n_shot]]

        test_rows = test_by_subj[subj]
        picked = _sample_subject(test_rows, args.n_per_subject, rng)
        samples.extend(_to_sample(r) for r in picked)
        print(f"  {subj:20s}  test={len(test_rows):4d}  dev={len(dev_rows):3d}  "
              f"sampled={len(picked)}", flush=True)

    holdout = MmluProHoldout(
        subjects=canonical_subjects,
        n_per_subject=args.n_per_subject,
        n_shot=args.n_shot,
        seed=args.seed,
        shots=shots,
        samples=samples,
    )
    save_holdout(holdout, args.out)
    print(f"\nWrote {args.out}  ({len(samples)} samples, "
          f"{sum(len(v) for v in shots.values())} shots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

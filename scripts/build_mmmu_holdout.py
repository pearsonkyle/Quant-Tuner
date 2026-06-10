#!/usr/bin/env python3
"""Build an MMMU holdout from the test split of ``MMMU/MMMU``.

Downloads every requested subject from HuggingFace, samples up to
``--n-per-subject`` questions from the **test** split (answers are now public
on the MMMU/MMMU dataset), and pulls up to ``--n-shot`` examples from the
**dev** split for use as few-shot demonstrations.

Images are saved as JPEG files under ``<out-dir>/images/`` (named
``<sample_id>_<slot>.jpg``).  The holdout JSON is written to
``<out-dir>/holdout.json`` and references images by their filename relative to
``<out-dir>/images/``; the two outputs therefore travel together as a single
directory.

Default: 25 questions per subject across all 30 subjects (750 total), 0-shot.
Pass ``--subjects`` to restrict to a subset, ``--n-shot 2`` for 2-shot
demonstrations from the dev split.

Usage examples
--------------
# Full 30-subject test set (all questions, no sampling cap)
uv run python scripts/build_mmmu_holdout.py --n-per-subject 9999

# Quick smoke-test: 5 subjects, 10 questions each, 2-shot
uv run python scripts/build_mmmu_holdout.py \\
    --subjects Computer_Science Math Physics Chemistry Biology \\
    --n-per-subject 10 --n-shot 2

# Default (25 × 30 subjects, 0-shot)
uv run python scripts/build_mmmu_holdout.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.mmmu import (
    MMMU_SUBJECTS,
    MmmuHoldout,
    MmmuSample,
    hf_row_to_sample,
    save_holdout,
)

_DEFAULT_OUT = _REPO / "out" / "mmmu" / "holdout.json"


def _safe_id(raw_id: str) -> str:
    """Sanitise a raw HuggingFace row id into a safe filename stem."""
    return raw_id.replace("/", "_").replace(" ", "_")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--subjects", nargs="+", default=MMMU_SUBJECTS,
        metavar="SUBJECT",
        help=(
            "MMMU subject config names to include "
            f"(default: all {len(MMMU_SUBJECTS)} subjects). "
            "Example: Computer_Science Math Physics"
        ),
    )
    p.add_argument(
        "--n-per-subject", type=int, default=25,
        help="Max questions sampled from the test split per subject (default: 25). "
             "Pass a large number (e.g. 9999) to use all available questions.",
    )
    p.add_argument(
        "--n-shot", type=int, default=0,
        help="Number of dev-split examples to use as few-shot demonstrations "
             "per subject (default: 0).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help=f"Output holdout JSON path (default: {_DEFAULT_OUT}). "
             "Images are saved to a sibling images/ directory.",
    )
    p.add_argument(
        "--dataset", default="MMMU/MMMU",
        help="HuggingFace dataset id (default: MMMU/MMMU).",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    out_path: Path = args.out
    images_dir = out_path.parent / "images"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Heavy import deferred so --help is fast.
    from datasets import load_dataset

    rng = random.Random(args.seed)
    shots: dict[str, list[MmmuSample]] = {}
    samples: list[MmmuSample] = []
    canonical_subjects: list[str] = []

    for subject in args.subjects:
        print(f"\nLoading {args.dataset} / {subject} …", flush=True)

        # --- test split (evaluation samples) ---
        try:
            ds_test = load_dataset(args.dataset, subject, split="test")
        except Exception as exc:
            print(f"  WARN: could not load test split for {subject!r}: {exc}",
                  file=sys.stderr, flush=True)
            continue

        test_rows = list(ds_test)
        n_available = len(test_rows)

        if args.n_per_subject < n_available:
            picked_rows = rng.sample(test_rows, args.n_per_subject)
        else:
            picked_rows = list(test_rows)

        subject_samples: list[MmmuSample] = []
        for row in picked_rows:
            raw_id = str(row.get("id", f"{subject}_test_{len(subject_samples)}"))
            sid = _safe_id(raw_id)
            s = hf_row_to_sample(row, sid, images_dir)
            subject_samples.append(s)

        samples.extend(subject_samples)
        canonical_subjects.append(subject)

        print(
            f"  test: {n_available} available → {len(subject_samples)} sampled",
            flush=True,
        )

        # --- dev split (few-shot demonstrations) ---
        if args.n_shot > 0:
            try:
                ds_dev = load_dataset(args.dataset, subject, split="dev")
                dev_rows = list(ds_dev)[: args.n_shot]
                shot_list: list[MmmuSample] = []
                for row in dev_rows:
                    raw_id = str(row.get("id", f"{subject}_dev_{len(shot_list)}"))
                    sid = _safe_id(raw_id)
                    s = hf_row_to_sample(row, sid, images_dir)
                    shot_list.append(s)
                shots[subject] = shot_list
                print(f"  dev:  {len(dev_rows)} shots loaded", flush=True)
            except Exception as exc:
                print(f"  WARN: could not load dev split for {subject!r}: {exc}",
                      file=sys.stderr, flush=True)

    if not canonical_subjects:
        print("ERROR: no subjects could be loaded.", file=sys.stderr)
        return 1

    holdout = MmmuHoldout(
        subjects=canonical_subjects,
        n_per_subject=args.n_per_subject,
        n_shot=args.n_shot,
        seed=args.seed,
        shots=shots,
        samples=samples,
    )
    save_holdout(holdout, out_path)

    total_shots = sum(len(v) for v in shots.values())
    print(
        f"\nWrote {out_path}"
        f"  ({len(samples)} samples across {len(canonical_subjects)} subjects"
        + (f", {total_shots} shots" if total_shots else "")
        + ")",
        flush=True,
    )
    print(f"Images saved to: {images_dir}  "
          f"({len(list(images_dir.glob('*.jpg')))} JPEG files)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

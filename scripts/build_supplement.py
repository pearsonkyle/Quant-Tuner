#!/usr/bin/env python3
"""Build the hand-authored broad-domain calibration/MTP supplement.

The source tree is `calibration_supplements/broad/<area>/<subject>.txt`, plain UTF-8,
blank-line-separated samples (same format spec as `calibration_supplements/mmmu/`).

This script does three things:

  stats  — per-area / per-file token accounting against the 1M-token target
  lint   — format checks (encoding, special tokens, separator discipline, headers)
  build  — deterministic disjoint split into a *calibration* half and an *MTP
           training* half, plus a manifest recording exactly what went where

The split is at **sample** granularity (blank-line-delimited blocks) and is
**stratified per source file**, so both halves carry every subject in the same
proportion. It is seeded (default 42) and content-addressed: the manifest records a
sha256 of each half, so a rebuild that produces different bytes is visible.

Why disjoint halves matter: per CLAUDE.md the repo's standing invariant is that
calibration and eval slices never overlap. The MTP half is training data for a draft
head; the calibration half feeds imatrix/AWQ/GPTQ. Letting a sample appear in both
would make any MTP acceptance number measured on calibrated quants partly a
memorization readout.

Usage:
    python scripts/build_supplement.py stats
    python scripts/build_supplement.py lint
    python scripts/build_supplement.py build --out out/supplement
    python scripts/build_supplement.py build --out out/supplement \
        --tokenizer ~/.cache/huggingface/.../snapshot   # exact counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Parsing and splitting live in the library, not here: `quant_tuner.datasets.broad_supplement`
# also builds the published dataset from this same tree, and if the two could parse or split
# it differently, they eventually would. Importing is what makes the drift impossible.
from quant_tuner.datasets.broad_supplement import (  # noqa: E402
    CHARS_PER_TOKEN,
    FORBIDDEN_PATTERNS,
    SRC_ROOT,
    TARGET_TOKENS,
    assign_halves,
    source_files,
    split_samples,
)


@dataclass
class FileStat:
    path: Path
    area: str
    subject: str
    chars: int
    samples: int
    tokens: int
    exact: bool


@dataclass
class LintIssue:
    path: Path
    line: int
    message: str


@dataclass
class Tokenizer:
    """Exact tokenizer if one was resolved, else a chars/token estimator."""

    impl: object | None = None

    @property
    def exact(self) -> bool:
        return self.impl is not None

    def count(self, text: str) -> int:
        if self.impl is None:
            return int(len(text) / CHARS_PER_TOKEN)
        return len(self.impl(text, add_special_tokens=False)["input_ids"])


def load_tokenizer(path: str | None) -> Tokenizer:
    if not path:
        return Tokenizer(None)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("warning: transformers not installed; falling back to estimates", file=sys.stderr)
        return Tokenizer(None)
    return Tokenizer(AutoTokenizer.from_pretrained(path, trust_remote_code=True))


def collect(tok: Tokenizer) -> list[FileStat]:
    stats: list[FileStat] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(SRC_ROOT)
        stats.append(
            FileStat(
                path=path,
                area=rel.parts[0] if len(rel.parts) > 1 else "_root",
                subject=path.stem,
                chars=len(text),
                samples=len(split_samples(text)),
                tokens=tok.count(text),
                exact=tok.exact,
            )
        )
    return stats


def cmd_stats(args: argparse.Namespace) -> int:
    tok = load_tokenizer(args.tokenizer)
    stats = collect(tok)
    if not stats:
        print(f"no source files under {SRC_ROOT}")
        return 1

    by_area: dict[str, list[FileStat]] = {}
    for s in stats:
        by_area.setdefault(s.area, []).append(s)

    label = "tokens" if tok.exact else "tokens~"
    total = sum(s.tokens for s in stats)

    print(f"{'area / subject':<46}{'samples':>9}{label:>12}{'chars':>11}")
    print("-" * 78)
    for area in sorted(by_area):
        files = by_area[area]
        at = sum(f.tokens for f in files)
        print(f"{area:<46}{sum(f.samples for f in files):>9}{at:>12,}{sum(f.chars for f in files):>11,}")
        if args.verbose:
            for f in sorted(files, key=lambda x: x.subject):
                print(f"  {f.subject:<44}{f.samples:>9}{f.tokens:>12,}{f.chars:>11,}")
    print("-" * 78)
    print(f"{'TOTAL':<46}{sum(s.samples for s in stats):>9}{total:>12,}{sum(s.chars for s in stats):>11,}")
    pct = 100.0 * total / TARGET_TOKENS
    print(f"\ntarget {TARGET_TOKENS:,} tokens — {pct:.1f}% complete, {max(0, TARGET_TOKENS - total):,} to go")
    if not tok.exact:
        print(f"(estimates at {CHARS_PER_TOKEN} chars/token; pass --tokenizer PATH for exact counts)")
    return 0


def lint_file(path: Path) -> list[LintIssue]:
    issues: list[LintIssue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [LintIssue(path, 0, f"not valid UTF-8: {exc}")]

    for i, line in enumerate(text.splitlines(), start=1):
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(line):
                issues.append(LintIssue(path, i, f"raw special token {pat.pattern!r}"))
    if "\t" in text:
        issues.append(LintIssue(path, 0, "contains tab characters (use spaces)"))
    if "\r\n" in text:
        issues.append(LintIssue(path, 0, "contains CRLF line endings"))
    if not text.endswith("\n"):
        issues.append(LintIssue(path, 0, "missing trailing newline"))
    if re.search(r"\n{4,}", text):
        issues.append(LintIssue(path, 0, "3+ consecutive blank lines (separator is exactly one)"))
    samples = split_samples(text)
    if not samples:
        issues.append(LintIssue(path, 0, "no samples found"))
    for s in samples:
        if len(s) < 40:
            issues.append(LintIssue(path, 0, f"suspiciously short sample: {s[:50]!r}"))
    return issues


def cmd_lint(args: argparse.Namespace) -> int:
    files = source_files()
    if not files:
        print(f"no source files under {SRC_ROOT}")
        return 1
    issues = [i for f in files for i in lint_file(f)]
    for i in issues:
        loc = f"{i.path.relative_to(REPO)}:{i.line}" if i.line else str(i.path.relative_to(REPO))
        print(f"{loc}: {i.message}")
    print(f"\n{len(files)} files, {len(issues)} issue(s)")
    return 1 if issues else 0


def cmd_build(args: argparse.Namespace) -> int:
    tok = load_tokenizer(args.tokenizer)
    files = source_files()
    if not files:
        print(f"no source files under {SRC_ROOT}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib: list[str] = []
    mtp: list[str] = []
    per_file: list[dict] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        samples = split_samples(text)
        rel = path.relative_to(SRC_ROOT)
        # Per-file seeding (see `assign_halves`): adding a NEW file must never reshuffle
        # an existing one, or every batch would migrate samples across halves and silently
        # contaminate whatever had already been trained or calibrated on.
        halves = assign_halves(len(samples), str(rel), args.seed, args.calib_fraction)
        calib.extend(s for s, h in zip(samples, halves, strict=True) if h == "calib")
        mtp.extend(s for s, h in zip(samples, halves, strict=True) if h == "mtp")
        per_file.append(
            {
                "file": str(rel),
                "area": rel.parts[0] if len(rel.parts) > 1 else "_root",
                "samples": len(samples),
                "calib_samples": halves.count("calib"),
                "mtp_samples": halves.count("mtp"),
            }
        )

    calib_text = "\n\n".join(calib) + "\n"
    mtp_text = "\n\n".join(mtp) + "\n"
    calib_path = out / "supplement.calib.txt"
    mtp_path = out / "supplement.mtp.txt"
    calib_path.write_text(calib_text, encoding="utf-8")
    mtp_path.write_text(mtp_text, encoding="utf-8")

    # Cheap belt-and-braces on the disjointness claim the docstring makes.
    overlap = set(calib) & set(mtp)
    if overlap:
        print(f"ERROR: {len(overlap)} sample(s) in both halves", file=sys.stderr)
        return 1

    manifest = {
        "seed": args.seed,
        "calib_fraction": args.calib_fraction,
        "tokens_exact": tok.exact,
        "tokenizer": args.tokenizer,
        "calib": {
            "path": str(calib_path),
            "samples": len(calib),
            "chars": len(calib_text),
            "tokens": tok.count(calib_text),
            "sha256": hashlib.sha256(calib_text.encode()).hexdigest(),
        },
        "mtp": {
            "path": str(mtp_path),
            "samples": len(mtp),
            "chars": len(mtp_text),
            "tokens": tok.count(mtp_text),
            "sha256": hashlib.sha256(mtp_text.encode()).hexdigest(),
        },
        "files": per_file,
    }
    (out / "supplement_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    unit = "tokens" if tok.exact else "tokens (est)"
    print(f"calib: {len(calib):>6} samples  {manifest['calib']['tokens']:>9,} {unit}  -> {calib_path}")
    print(f"mtp:   {len(mtp):>6} samples  {manifest['mtp']['tokens']:>9,} {unit}  -> {mtp_path}")
    print(f"manifest -> {out / 'supplement_manifest.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="token accounting against the 1M target")
    p.add_argument("--tokenizer", help="HF model path for exact token counts")
    p.add_argument("-v", "--verbose", action="store_true", help="per-subject breakdown")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("lint", help="format checks")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("build", help="deterministic disjoint calib/MTP split")
    p.add_argument("--out", default="out/supplement")
    p.add_argument("--seed", default="42")
    p.add_argument("--calib-fraction", type=float, default=0.5)
    p.add_argument("--tokenizer", help="HF model path for exact token counts")
    p.set_defaults(func=cmd_build)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

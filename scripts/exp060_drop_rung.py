"""Remove a rung from an exp-060 run: its GGUF, its bench rows, and its agent-bench row.

Dry-run by default. A rung is spread across six per-eval CSVs plus the SWE mimic results,
and a half-deleted rung is worse than either state — the card would render a column with
some evals present and others silently missing.

    python scripts/exp060_drop_rung.py --run exp-060-32k --rung IQ4_XS
    python scripts/exp060_drop_rung.py --run exp-060-32k --rung IQ4_XS --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWE_RESULTS = Path("/workspace/swe-mimic/swe_mimic_results.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp-060-32k")
    ap.add_argument("--rung", required=True, help="e.g. IQ4_XS")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()

    root = REPO / "out" / a.run
    rung = a.rung
    sub = root / rung.lower()

    # The label column is "<model>|<QUANT>|<method> // eval=... // baseline=FP16", so match
    # on the pipe-delimited field rather than a bare substring: "IQ4_XS" is also a substring
    # of nothing else today, but "Q4_K" would match "Q4_K_M" and quietly drop two rungs.
    needle = f"|{rung}|"

    planned: list[str] = []
    for csv in sorted(root.glob("results*.csv")):
        lines = csv.read_text().splitlines(keepends=True)
        keep = [ln for ln in lines if needle not in ln]
        n = len(lines) - len(keep)
        if n:
            planned.append(f"  {csv.relative_to(root)}: drop {n} row(s)")
            if a.apply:
                shutil.copy2(csv, csv.with_suffix(csv.suffix + ".bak"))
                csv.write_text("".join(keep))

    # Task-level evals key on the GGUF basename, not the "|QUANT|" label, so they need
    # their own match — dropping the rung from results*.csv alone would leave the card's
    # tool-call row still populated for a model that no longer exists.
    for extra in sorted((root / "eval").glob("*_results.csv")):
        lines = extra.read_text().splitlines(keepends=True)
        keep = [ln for ln in lines if f"-{rung}.gguf" not in ln]
        n = len(lines) - len(keep)
        if n:
            planned.append(f"  {extra.relative_to(root)}: drop {n} row(s)")
            if a.apply:
                shutil.copy2(extra, extra.with_suffix(extra.suffix + ".bak"))
                extra.write_text("".join(keep))

    if SWE_RESULTS.exists():
        lines = SWE_RESULTS.read_text().splitlines(keepends=True)
        keep = [ln for ln in lines if not ln.startswith(f"{rung},")]
        n = len(lines) - len(keep)
        if n:
            planned.append(f"  {SWE_RESULTS}: drop {n} row(s)")
            if a.apply:
                shutil.copy2(SWE_RESULTS, SWE_RESULTS.with_suffix(".csv.bak"))
                SWE_RESULTS.write_text("".join(keep))

    if sub.exists():
        size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
        planned.append(f"  {sub.relative_to(root)}/: remove dir ({size / 1024**3:.2f} GiB)")
        if a.apply:
            shutil.rmtree(sub)

    if not planned:
        print(f"nothing to drop for {rung} in {root}")
        return 0

    print(f"{'DROPPING' if a.apply else 'WOULD DROP'} {rung} from {root}:")
    print(*planned, sep="\n")
    if not a.apply:
        print("\nDRY RUN — re-run with --apply. CSVs are backed up to *.bak.")
    else:
        print("\ndone. Remember: the model card and upload_to_hf.py payload still "
              "reference this rung.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Patch the gemma-4-31B-it-awq-2bit-GGUF release with exp-020 outputs.

Reads results.csv from `out/exp-020/google__gemma-4-31B-it/{gate,
imatrix-only,plain}/` and the F16 baseline log, then:

  1. Renders the §3 Comparison table block in the format already used by
     `uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/README.md` and
     substitutes it in place.
  2. Rewrites the §2.3 Validation row to describe the MMMU swap.
  3. Copies the three new GGUFs from `gate/` into the upload dir
     (overwriting the old shipped artifacts).

Idempotent. Refuses to clobber the README if the §3 anchor strings can't
be located (so a future README edit doesn't silently corrupt the file).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/update_release_assets_exp020.py
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP20 = REPO / "out" / "exp-020" / "google__gemma-4-31B-it"
UPLOAD = REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
README = UPLOAD / "README.md"

GATE_CSV = EXP20 / "gate" / "results.csv"
IMAT_CSV = EXP20 / "imatrix-only" / "results.csv"
PLAIN_CSV = EXP20 / "plain" / "results.csv"
BASELINE_LOG = EXP20 / "logs" / "baseline.log"

GGUFS = [
    ("IQ2_XS", "gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf"),
    ("IQ2_M",  "gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf"),
    ("Q2_K_S", "gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf"),
]

# Shipped alongside the GGUFs so downstream users can rebuild exp-020.
CORPORA_FILES = [
    "corpus.cal.txt",
    "corpus.val.txt",
    "corpus.eval.txt",
    "corpora_audit.json",
]


def _load_row(csv_path: Path, quant: str, technique_substring: str) -> dict:
    assert csv_path.exists(), f"missing {csv_path}"
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if f"|{quant}|" in model and technique_substring in model:
                return row
    raise LookupError(f"no row for quant={quant} technique~={technique_substring} in {csv_path}")


def _f16_ppl() -> float:
    assert BASELINE_LOG.exists(), f"missing {BASELINE_LOG}"
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", BASELINE_LOG.read_text())
    assert m, "could not parse FP16 PPL from baseline.log"
    return float(m.group(1))


def _fmt_ppl(v: str | float) -> str:
    return f"{float(v):.2f}"


def _fmt_kld(v: str | float) -> str:
    return f"{float(v):.5f}"


def _fmt_top(v: str | float) -> str:
    # CSVs store same_top_p as a percentage in [0, 100].
    return f"{float(v):.2f}%"


def _fmt_size(v: str | float) -> str:
    return f"{float(v):.2f}"


def _fmt_bpw(v: str | float) -> str:
    return f"{float(v):.3f}"


def _render_table() -> str:
    fp16_size = 57.20
    fp16_bpw = 16.005
    fp16_ppl = _f16_ppl()

    rows = []
    rows.append(
        "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |"
    )
    rows.append("|---|---|---:|---:|---:|---:|---:|")
    rows.append(
        f"| FP16 | none (reference) | {fp16_size:.2f} | {fp16_bpw:.3f} | "
        f"**{fp16_ppl:.2f}** | 0.00000 | 100.00% |"
    )

    for q in ("IQ2_XS", "IQ2_M", "Q2_K_S"):
        imat = _load_row(IMAT_CSV, q, "|imatrix|")
        gate = _load_row(GATE_CSV, q, "|awq-cv-gate+imatrix|")
        rows.append(
            f"| {q} | imatrix only | "
            f"{_fmt_size(imat['size_gib'])} | {_fmt_bpw(imat['bpw'])} | "
            f"{_fmt_ppl(imat['ppl'])} | {_fmt_kld(imat['mean_kld'])} | "
            f"{_fmt_top(imat['same_top_p'])} |"
        )
        rows.append(
            f"| **{q}** | **AWQ cv-gate + imatrix** | "
            f"**{_fmt_size(gate['size_gib'])}** | **{_fmt_bpw(gate['bpw'])}** | "
            f"**{_fmt_ppl(gate['ppl'])}** | **{_fmt_kld(gate['mean_kld'])}** | "
            f"**{_fmt_top(gate['same_top_p'])}** |"
        )

    plain = _load_row(PLAIN_CSV, "Q2_K", "|plain|")
    rows.append(
        f"| Q2_K | plain (no imatrix, no AWQ) | "
        f"{_fmt_size(plain['size_gib'])} | {_fmt_bpw(plain['bpw'])} | "
        f"{_fmt_ppl(plain['ppl'])} | {_fmt_kld(plain['mean_kld'])} | "
        f"{_fmt_top(plain['same_top_p'])} |"
    )
    return "\n".join(rows)


# Anchor strings used to locate the §3 table and §2.3 validation row.
# Both come from the current README; if either changes, this script must
# be updated rather than silently doing the wrong thing.
TABLE_START_ANCHOR = (
    "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |"
)
TABLE_END_ANCHOR_RE = re.compile(
    r"^\| Q2_K \| plain \(no imatrix, no AWQ\) \|.*\|$",
    re.MULTILINE,
)
VALIDATION_ROW_OLD = (
    "<tr><td style=\"padding:8px 10px; border:1px solid #bfdbfe;\">"
    "<b>Validation</b></td><td style=\"padding:8px 10px; border:1px solid #bfdbfe;\">"
    "~10k tokens of usage-log (disjoint sessions) + a small Rust/JSON/YAML supplement"
    "</td><td style=\"padding:8px 10px; border:1px solid #bfdbfe;\">"
    "held-out gate for per-tensor α</td></tr>"
)
VALIDATION_ROW_NEW = (
    "<tr><td style=\"padding:8px 10px; border:1px solid #bfdbfe;\">"
    "<b>Validation</b></td><td style=\"padding:8px 10px; border:1px solid #bfdbfe;\">"
    "MMMU disciplines corpus (~100–200k tokens) drawn from "
    "<code>calibration_supplements/mmmu/combined.txt</code> — out-of-distribution "
    "relative to the calibration mix"
    "</td><td style=\"padding:8px 10px; border:1px solid #bfdbfe;\">"
    "held-out gate for per-tensor α</td></tr>"
)


def _patch_readme(new_table: str) -> None:
    text = README.read_text()

    # Patch §3 table.
    start = text.find(TABLE_START_ANCHOR)
    if start == -1:
        raise RuntimeError(f"§3 table anchor not found in {README}")
    end_m = TABLE_END_ANCHOR_RE.search(text, pos=start)
    if end_m is None:
        raise RuntimeError(f"§3 table end anchor not found in {README}")
    text = text[:start] + new_table + text[end_m.end():]

    # Patch §2.3 validation row (best-effort: only patch if old text present).
    if VALIDATION_ROW_OLD in text:
        text = text.replace(VALIDATION_ROW_OLD, VALIDATION_ROW_NEW)
    else:
        print(
            "  warn: validation row anchor not found; §2.3 not updated. "
            "Patch by hand if needed.",
            file=sys.stderr,
        )

    README.write_text(text)


def _copy_corpora() -> None:
    src_dir = EXP20 / "corpora"
    dst_dir = UPLOAD / "calibration_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in CORPORA_FILES:
        src = src_dir / name
        if not src.exists():
            print(f"  skip corpus: {src} missing", file=sys.stderr)
            continue
        dst = dst_dir / name
        print(f"  copy: {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
        shutil.copy2(src, dst)


def _copy_ggufs() -> None:
    for quant, name in GGUFS:
        src = EXP20 / "gate" / f"{quant}-awq.gguf"
        dst = UPLOAD / name
        if not src.exists():
            print(f"  skip: {src} missing", file=sys.stderr)
            continue
        print(f"  copy: {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
        shutil.copy2(src, dst)


def main() -> int:
    table = _render_table()
    print("rendered §3 table:")
    print(table)
    print()
    _patch_readme(table)
    print(f"patched {README.relative_to(REPO)}")
    _copy_corpora()
    _copy_ggufs()
    return 0


if __name__ == "__main__":
    sys.exit(main())

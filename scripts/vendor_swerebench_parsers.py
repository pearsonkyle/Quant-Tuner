#!/usr/bin/env python3
"""Re-vendor SWE-rebench-V2's multi-language log parsers into this repo.

    .venv/bin/python scripts/vendor_swerebench_parsers.py            # clone + vendor
    .venv/bin/python scripts/vendor_swerebench_parsers.py --check    # CI-style drift check

Writes ``src/quant_tuner/eval/_swerebench_v2_parsers.py``: upstream's
``lib/agent/log_parsers.py`` **verbatim**, with the ``TestStatus`` import replaced by an
inlined copy of upstream's enum so the module stands alone.

Why vendor rather than reimplement: SWE-rebench-V2 records each instance's
``install_config.log_parser`` by name, and the recorded FAIL_TO_PASS / PASS_TO_PASS ids
are whatever *that exact function* emitted at dataset-build time. A near-miss
reimplementation parses zero matching ids, marks every trajectory unresolved, and reads
identically to "the model failed" — a silent, expensive failure.

Run this after bumping the upstream pin, then re-run
``scripts/validate_swebench_v2_grading.py`` (gold patches must still resolve).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/SWE-rebench/SWE-rebench-V2.git"
SRC_REL = "lib/agent/log_parsers.py"
DEST = _REPO / "src" / "quant_tuner" / "eval" / "_swerebench_v2_parsers.py"

_HEADER = '''"""Vendored multi-language test-log parsers from SWE-rebench-V2 (MIT).

Source: {upstream} (``{src_rel}``)
Commit: {commit}
Copyright (c) 2026 SWE-rebench. MIT License — see the root ``NOTICE``.

Vendored **verbatim** except for this header and the ``TestStatus`` import, which is
inlined below so the module has no dependency on the upstream package layout. Do not
hand-edit the parser bodies: re-vendor with ``scripts/vendor_swerebench_parsers.py``
when bumping, so grading stays bit-compatible with the dataset's own harness.

Why vendored rather than reimplemented: SWE-rebench-V2 records each instance's
``install_config.log_parser`` by name, and ``FAIL_TO_PASS``/``PASS_TO_PASS`` ids are
whatever *that exact function* emitted at dataset-build time. A near-miss
reimplementation would silently parse zero matching ids and mark every trajectory
unresolved — which reads identically to "the model failed."

Every parser has the signature ``(log: str) -> dict[test_id, status]`` where status is
one of ``PASSED`` / ``FAILED`` / ``SKIPPED`` / ``ERROR``. ``NAME_TO_PARSER`` maps the
dataset's ``log_parser`` string to the function.
"""

from enum import Enum


class TestStatus(str, Enum):
    """Inlined from SWE-rebench-V2's ``lib/agent/swe_constants.py``."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


'''


def render(source: str, commit: str) -> str:
    """Upstream source -> the vendored module text."""
    body = source.replace("from lib.agent.swe_constants import TestStatus\n", "")
    lines = body.split("\n")
    if lines[0] != "import json":
        raise SystemExit(f"unexpected upstream layout: first line is {lines[0]!r}")
    imports, rest = lines[:5], lines[5:]
    header = _HEADER.format(upstream=UPSTREAM.removesuffix(".git"), src_rel=SRC_REL,
                            commit=commit)
    return header + "\n".join(imports) + "\n" + "\n".join(rest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="HEAD", help="upstream git ref to vendor (default HEAD)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the vendored copy differs from upstream (no write)")
    ap.add_argument("--from-checkout", type=Path, default=None,
                    help="vendor from an existing local clone instead of cloning "
                         "(use on a machine without network access to github.com)")
    args = ap.parse_args()

    def _git(*cmd: str, cwd: Path | None = None) -> str:
        """Run git, surfacing stderr — a bare CalledProcessError hides 'no such host'."""
        proc = subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"[vendor] git {' '.join(cmd)} failed (rc={proc.returncode}):\n"
                f"{(proc.stderr or proc.stdout).strip()}\n"
                f"[vendor] If the network is restricted, clone SWE-rebench-V2 elsewhere "
                f"and pass --from-checkout <path>."
            )
        return proc.stdout.strip()

    with tempfile.TemporaryDirectory(prefix="swerebench_v2_") as tmp:
        if args.from_checkout:
            clone = args.from_checkout.expanduser().resolve()
            if not (clone / SRC_REL).exists():
                raise SystemExit(f"[vendor] {clone / SRC_REL} not found — is that a "
                                 f"SWE-rebench-V2 checkout?")
        else:
            clone = Path(tmp) / "src"
            _git("clone", "--depth", "1", UPSTREAM, str(clone))
            if args.ref != "HEAD":
                _git("fetch", "--depth", "1", "origin", args.ref, cwd=clone)
                _git("checkout", args.ref, cwd=clone)
        commit = _git("rev-parse", "HEAD", cwd=clone)
        rendered = render((clone / SRC_REL).read_text(), commit)

    if args.check:
        current = DEST.read_text() if DEST.exists() else ""
        if current == rendered:
            print(f"[vendor] up to date with upstream {commit[:12]}")
            return 0
        print(f"[vendor] DRIFT: vendored copy differs from upstream {commit[:12]}.\n"
              f"         Re-run without --check, then re-run "
              f"scripts/validate_swebench_v2_grading.py.", file=sys.stderr)
        return 1

    DEST.write_text(rendered)
    n_parsers = rendered.count("\ndef parse_")
    print(f"[vendor] wrote {DEST.relative_to(_REPO)} "
          f"({len(rendered.splitlines())} lines, {n_parsers} parsers) from {commit[:12]}")
    print("[vendor] now re-run: PYTHONPATH=src .venv/bin/python "
          "scripts/validate_swebench_v2_grading.py --holdout <multilang holdout>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

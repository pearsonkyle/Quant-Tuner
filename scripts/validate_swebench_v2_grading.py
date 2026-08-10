#!/usr/bin/env python3
"""Golden-patch check: does our grader actually grade SWE-rebench-V2 correctly?

Applies each instance's **own gold patch** and asserts the grade comes back
``resolved=True``. This is the equivalent of SWE-rebench-V2's ``eval.py --golden-eval``,
and it is the gate to run *before* spending hours generating trajectories.

Why it matters: for a multi-language dataset the grade depends on three things we
derive per instance — the checkout directory (``/<repo-name>``), the instance's own
``install_config.test_cmd``, and the named log parser. Get any of them wrong and the
parser returns an empty status map, every trajectory scores ``resolved=False``, and the
run looks exactly like "the model solved nothing." A gold patch that fails to resolve
means **our harness is broken**, not the model.

    PYTHONPATH=src .venv/bin/python scripts/validate_swebench_v2_grading.py \
        --holdout out/external/swe-rebench/holdout_multilang.jsonl --per-language 1

Exit code is non-zero if any instance failed to resolve, so this can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.swebench_grade import (  # noqa: E402
    diagnose_container_error,
    grade_instance,
    install_config_of,
    is_v2_instance,
    v2_workdir,
)


def _load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _select(rows: list[dict], per_language: int | None, languages: set[str] | None) -> list[dict]:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[(row.get("language") or "python").lower()].append(row)
    out: list[dict] = []
    for lang in sorted(by_lang):
        if languages and lang not in languages:
            continue
        out.extend(by_lang[lang] if per_language is None else by_lang[lang][:per_language])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--per-language", type=int, default=1,
                    help="instances to check per language (default 1; 0/omit-all with --all)")
    ap.add_argument("--all", action="store_true", help="check every instance in the holdout")
    ap.add_argument("--languages", default=None, help="comma-separated subset to check")
    ap.add_argument("--test-timeout", type=int, default=2400)
    ap.add_argument("--out", type=Path, default=None, help="write a JSON report here")
    ap.add_argument("--keep-going", action="store_true", default=True)
    args = ap.parse_args()

    rows = _load(args.holdout)
    langs = {s.strip().lower() for s in args.languages.split(",")} if args.languages else None
    selected = _select(rows, None if args.all else args.per_language, langs)
    if not selected:
        print("ERROR: nothing selected", file=sys.stderr)
        return 1

    print(f"Golden-patch check over {len(selected)} instance(s) from {args.holdout.name}\n")
    results = []
    for i, inst in enumerate(selected, 1):
        iid = inst["instance_id"]
        lang = (inst.get("language") or "python").lower()
        cfg = install_config_of(inst)
        image = inst.get("image_name") or inst.get("docker_image")
        print(f"[{i}/{len(selected)}] {iid}  ({lang})", flush=True)
        print(f"      image  = {image}", flush=True)
        if is_v2_instance(inst):
            print(f"      cwd    = {v2_workdir(inst)}", flush=True)
            print(f"      parser = {cfg.get('log_parser')}", flush=True)
        t0 = time.time()
        try:
            grade = grade_instance(
                inst, inst.get("patch") or "", image=image, test_timeout=args.test_timeout
            )
        except Exception as e:  # a broken image shouldn't abort the whole sweep
            grade = {"resolved": False, "error": diagnose_container_error(e), "log": ""}
        wall = time.time() - t0

        ok = bool(grade.get("resolved"))
        err = grade.get("error")
        print(f"      -> resolved={ok}  f2p={grade.get('n_fail_to_pass_passed')}/"
              f"{grade.get('n_fail_to_pass')}  p2p={grade.get('n_pass_to_pass_passed')}/"
              f"{grade.get('n_pass_to_pass')}  [{wall:.0f}s]"
              + (f"  error={err}" if err else ""), flush=True)
        if not ok:
            # The log is the only way to tell "image/build broke" from "parser mismatch".
            tail = (grade.get("log") or "")[-1500:]
            print("      --- log tail ---\n" + "\n".join(
                "      " + ln for ln in tail.splitlines()[-25:]), flush=True)
        results.append({
            "instance_id": iid, "language": lang, "resolved": ok, "error": err,
            "log_parser": cfg.get("log_parser"), "image": image, "wall_sec": round(wall, 1),
            "n_fail_to_pass": grade.get("n_fail_to_pass"),
            "n_fail_to_pass_passed": grade.get("n_fail_to_pass_passed"),
        })
        print(flush=True)

    n_ok = sum(1 for r in results if r["resolved"])
    print("=" * 64)
    print(f"GOLDEN-PATCH RESULT: {n_ok}/{len(results)} resolved")
    by_lang: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r["resolved"])
    for lang in sorted(by_lang):
        good = sum(by_lang[lang])
        flag = "ok " if good == len(by_lang[lang]) else "FAIL"
        print(f"  [{flag}] {lang:8s} {good}/{len(by_lang[lang])}")
    if n_ok < len(results):
        print("\nA gold patch that does not resolve means the HARNESS is wrong "
              "(workdir / test_cmd / parser), not the model. Fix before generating.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")
    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())

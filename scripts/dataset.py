"""Build / card / push the datasets declared in ``quant_tuner.datasets.registry``.

    .venv/bin/python scripts/dataset.py list
    .venv/bin/python scripts/dataset.py build swe-agentic-trajectories
    .venv/bin/python scripts/dataset.py push  swe-agentic-trajectories --bump minor -m "add round-3 trajectories"
    .venv/bin/python scripts/dataset.py push  swe-agentic-trajectories --dry-run

Adding a future dataset = one entry in ``registry.REGISTRY``; these commands are shared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.datasets import REGISTRY, get_spec  # noqa: E402
from quant_tuner.datasets.publish import (  # noqa: E402
    build,
    bump_version,
    read_manifest,
    write_card,
)
from quant_tuner.datasets.publish import push as push_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show registered datasets and their local version")

    b = sub.add_parser("build", help="materialize splits + refresh the card")
    b.add_argument("name")

    p = sub.add_parser("push", help="build, version, upload to the Hub, tag")
    p.add_argument("name")
    p.add_argument("--version", default=None, help="explicit X.Y.Z (overrides --bump)")
    p.add_argument("--bump", choices=["major", "minor", "patch"], default="patch")
    p.add_argument("-m", "--note", default="", help="changelog / commit note")
    p.add_argument("--private", action="store_true")
    p.add_argument("--include-withheld", action="store_true",
                   help="also upload publish=False splits (dual-use data). Gated on "
                        "--private — refuses to send withheld splits to a public repo.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-build", action="store_true", help="push what is already staged")

    args = ap.parse_args()

    if args.cmd == "list":
        for spec in REGISTRY:
            m = read_manifest(spec)
            splits = ", ".join(f"{n}={i['rows']}" for n, i in m.get("splits", {}).items())
            print(f"  {spec.name:28s} v{m.get('version','0.0.0'):8s} -> {spec.repo_id}"
                  f"  [{splits or 'not built'}]")
        return 0

    spec = get_spec(args.name)

    if args.cmd == "build":
        manifest = build(spec)
        card = write_card(spec, manifest)
        print(f"[dataset] card -> {card}")
        print(f"[dataset] staged at {spec.stage_dir} (v{manifest.get('version','0.0.0')})")
        return 0

    if args.cmd == "push":
        if not args.no_build:
            build(spec)
        manifest = read_manifest(spec)
        version = args.version or bump_version(manifest.get("version", "0.0.0"), args.bump)
        push_dataset(spec, version=version, note=args.note,
                     private=args.private, dry_run=args.dry_run,
                     include_withheld=args.include_withheld)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

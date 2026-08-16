"""Publish ONLY the rebuilt IQ2_M rung (AWQ) to the existing HF model repo.

DRY-RUN BY DEFAULT. Nothing is uploaded without `--push`.

Why a separate uploader from `out/exp-060-32k/release/upload_to_hf.py`: that one
pushes the card plus all four rungs (~55 GiB). Only IQ2_M changed, so re-uploading
IQ3_M / IQ4_XS / Q5_K_M would move 45 GiB to write back bytes identical to what is
already there.

Three things this enforces, each guarding a failure that is invisible after upload:

1. **The filename's TERMINAL token must be the quant tag** (`…-IQ2_M.gguf`) or
   Hugging Face will not derive an Ollama `:IQ2_M` tag. The staged name is checked,
   not assumed.
2. **The artifact is re-audited immediately before upload** — embedded chat
   template, `general.name`, and the Q8_0 MTP pin — because the thing being
   published is the file, not the build log that described it.
3. **The card must not still describe the OLD rung.** A card claiming the shipped
   IQ2_M's PPL next to a different file is a misrepresentation, so the card is
   scanned for the superseded numbers and for `*pending*` placeholders.

    export HF_TOKEN=hf_...
    PYTHONPATH=src .venv/bin/python scripts/exp062_ship_iq2m.py          # plan + verify
    PYTHONPATH=src .venv/bin/python scripts/exp062_ship_iq2m.py --push   # publish
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_REPO = "pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF"
SRC_QUANT = REPO / "out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf"
STAGE_DIR = REPO / "out/exp-062-release"
# The terminal token IS the Ollama tag. Do not "improve" this name.
RELEASE_NAME = "Qwen3.8-27B-IQ2_M.gguf"
CARD = REPO / "out/exp-060-32k/release/README.md"
TEMPLATE = REPO / "data/chat_templates/qwen3_8_safe_v2.jinja"

# Numbers that belonged to the SUPERSEDED imatrix IQ2_M. If any survive in the card
# the card is describing a file we are not shipping.
STALE_MARKERS = ["56.540", "0.1242", "72.0%"]


def stage(src: Path, dest_dir: Path, name: str) -> Path:
    """Hardlink the built quant under its release name (no second 9.7 GiB copy)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        if dest.stat().st_ino == src.stat().st_ino:
            return dest
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        import shutil
        shutil.copy2(src, dest)
    return dest


def audit(path: Path) -> list[str]:
    """Re-verify the artifact itself. Returns a list of problems (empty = good)."""
    sys.path.insert(0, str(REPO / "vendor/llama.cpp/gguf-py"))
    from gguf import GGUFReader

    problems: list[str] = []
    r = GGUFReader(str(path))

    def field(k: str) -> str | None:
        f = r.fields.get(k)
        return bytes(f.parts[f.data[0]]).decode() if f else None

    name = field("general.name")
    if name != "Qwen3.8-27B":
        problems.append(f"general.name is {name!r}, expected 'Qwen3.8-27B'")

    tmpl = field("tokenizer.chat_template")
    want = TEMPLATE.read_text()
    if tmpl != want:
        problems.append(
            f"embedded chat template does not match {TEMPLATE.name} "
            f"({len(tmpl or '')} vs {len(want)} bytes)")

    blk64 = [t for t in r.tensors if t.name.startswith("blk.64.")]
    types = Counter(t.tensor_type.name for t in blk64)
    if len(blk64) != 15:
        problems.append(f"MTP head has {len(blk64)} tensors, expected 15")
    if types.get("Q8_0", 0) != 8:
        problems.append(f"MTP head has {types.get('Q8_0', 0)} Q8_0 tensors, expected 8")

    print(f"  general.name        : {name!r}")
    print(f"  chat_template bytes : {len(tmpl or '')} (matches repo .jinja: {tmpl == want})")
    print(f"  MTP head            : {len(blk64)} tensors, {dict(types)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    ap.add_argument("--allow-stale-card", action="store_true",
                    help="publish even if the card still shows superseded IQ2_M numbers")
    a = ap.parse_args()

    if not SRC_QUANT.exists():
        print(f"FATAL: no quant at {SRC_QUANT}", file=sys.stderr)
        return 1
    if RELEASE_NAME.rsplit("-", 1)[-1].removesuffix(".gguf") != "IQ2_M":
        print("FATAL: release filename's terminal token is not the quant tag", file=sys.stderr)
        return 1

    dest = stage(SRC_QUANT, STAGE_DIR, RELEASE_NAME)
    size = dest.stat().st_size

    print(f"repo    : {a.repo_id}")
    print(f"source  : {SRC_QUANT.relative_to(REPO)}")
    print(f"staged  : {dest.relative_to(REPO)}  ({size / 1024**3:.2f} GiB)")
    print(f"uploads : README.md + {RELEASE_NAME}   (other rungs untouched)\n")

    print("=== artifact audit (the file, not the build log) ===")
    problems = audit(dest)

    print("\n=== card check ===")
    if not CARD.exists():
        problems.append(f"card missing: {CARD}")
    else:
        card = CARD.read_text()
        if "*pending*" in card:
            problems.append(f"card has {card.count('*pending*')} '*pending*' placeholder(s)")
        stale = [m for m in STALE_MARKERS if m in card]
        if stale and not a.allow_stale_card:
            problems.append(
                f"card still shows superseded IQ2_M numbers {stale} — it would describe "
                f"the OLD rung next to the new file")
        print(f"  card bytes          : {len(card):,}")
        print(f"  superseded numbers  : {stale or 'none'}")

    if problems:
        print("\nBLOCKED:", *problems, sep="\n  - ")
        return 2

    if not a.push:
        print("\nDRY RUN — nothing uploaded. Re-run with --push to publish.")
        return 0
    if not a.token:
        print("\nERROR: no token. Set HF_TOKEN or pass --token.", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=a.token)
    # Card first, so the repo is never live with a new file and a stale description.
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=a.repo_id, repo_type="model")
    print("  uploaded README.md")
    print(f"  uploading {RELEASE_NAME} ({size / 1024**3:.2f} GiB) …", flush=True)
    api.upload_file(path_or_fileobj=str(dest), path_in_repo=RELEASE_NAME,
                    repo_id=a.repo_id, repo_type="model")
    print(f"\ndone → https://huggingface.co/{a.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Add a Multi-Token-Prediction (MTP) drafter to the gemma-4-31B AWQ GGUF package.

Gemma 4 ships a small "assistant" head (`google/gemma-4-31B-it-assistant`,
arch `Gemma4AssistantForCausalLM`) that predicts several future tokens from the
target model's last hidden state. llama.cpp runs it as a speculative *draft*
model via `--spec-type draft-mtp`, so the trunk GGUF is never modified — our
carefully-tuned AWQ tensors stay byte-for-byte identical. One drafter pairs with
*any* quant of the same base model.

Mainline `convert_hf_to_gguf.py` does not yet recognise `Gemma4AssistantForCausalLM`
(ggml-org/llama.cpp#23727 is open), so rather than convert the raw assistant
ourselves we reuse the already-converted, mainline-loadable drafter that Unsloth
published (arch `gemma4-assistant`, derived from the same Google source). This
script downloads it into the package directory, mirroring Unsloth's layout:

    <pkg>/mtp-gemma-4-31B-it.gguf            # Q8_0, root copy for -hf auto-discovery
    <pkg>/MTP/gemma-4-31B-it-Q8_0-MTP.gguf   # Q8_0 (recommended)
    <pkg>/MTP/gemma-4-31B-it-BF16-MTP.gguf   # full precision
    <pkg>/MTP/README.md

One drafter serves every trunk quant of the same base model (it keys off the
target's hidden size / vocab, not the quantization), and it cannot be merged into
the trunk: a GGUF carries one `general.architecture`, and the drafter's
`gemma4-assistant` arch is distinct from the `gemma4` trunk.

Requires a llama.cpp build from after 2026-06-07 (PR ggml-org/llama.cpp#23398);
older builds cannot load the `gemma4-assistant` architecture.

Usage:
    python scripts/add_mtp_drafter.py \
        --pkg uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF

    python scripts/add_mtp_drafter.py --pkg <dir> --q8-only   # skip BF16/F16
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCE_REPO = "unsloth/gemma-4-31B-it-GGUF"

# (repo path, required) — the root Q8_0 copy + the MTP/ folder Unsloth ships.
ROOT_DRAFTER = "mtp-gemma-4-31B-it.gguf"
MTP_FILES = [
    "MTP/gemma-4-31B-it-Q8_0-MTP.gguf",
    "MTP/gemma-4-31B-it-BF16-MTP.gguf",
    "MTP/README.md",
]
# Files kept when --q8-only is passed (drop the ~1.8 GB BF16/F16 mirrors).
Q8_ONLY = {ROOT_DRAFTER, "MTP/gemma-4-31B-it-Q8_0-MTP.gguf", "MTP/README.md"}


def fetch(repo_file: str, pkg: Path) -> Path:
    dest = pkg / repo_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {repo_file}", flush=True)
    cached = hf_hub_download(repo_id=SOURCE_REPO, filename=repo_file)
    # hf_hub_download returns a path inside the hub cache; copy into the package
    # so the package directory is self-contained and uploadable.
    if dest.exists():
        dest.unlink()
    shutil.copy2(cached, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pkg",
        type=Path,
        default=Path("uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF"),
        help="Package directory to add the drafter to.",
    )
    ap.add_argument(
        "--q8-only",
        action="store_true",
        help="Only fetch the Q8_0 drafter (skip the BF16/F16 mirrors).",
    )
    args = ap.parse_args()

    pkg: Path = args.pkg
    if not pkg.is_dir():
        print(f"error: package dir not found: {pkg}", file=sys.stderr)
        return 2

    wanted = [ROOT_DRAFTER, *MTP_FILES]
    if args.q8_only:
        wanted = [f for f in wanted if f in Q8_ONLY]

    print(f"Adding MTP drafter from {SOURCE_REPO} -> {pkg}")
    written = [fetch(f, pkg) for f in wanted]

    print("\nDone. Added:")
    for p in written:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.relative_to(pkg)}  ({size_mb:,.0f} MiB)")

    print(
        "\nRun (needs llama.cpp built after 2026-06-07):\n"
        f"  llama-server -m {pkg.name}/<your-awq-trunk>.gguf \\\n"
        f"      --model-draft {pkg.name}/{ROOT_DRAFTER} \\\n"
        "      --spec-type draft-mtp --spec-draft-n-max 4 -ngl 999 -fa on"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

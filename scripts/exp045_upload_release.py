"""Push the tmax-27b imatrix+MTP GGUF ladder to HF in one commit.

Uploads the six terminal-quant-named GGUFs (`tmax-27b-IQ4_XS.gguf` etc. so
`ollama run …:IQ4_XS` resolves; 2→4-bit, each with the grafted Qwopus nextn head at
blk.64@Q8), the README, the MTP note, and the calibration_data corpora — in a single
atomic create_commit.

⚠️ NOT run automatically. Confirm allenai/tmax-27b's exact license terms and that
the README frontmatter `license:` is correct before pushing. Requires
HUGGING_FACE_HUB_TOKEN / HF_TOKEN in the env, and that the target repo exists (create
it on HF first, or add `api.create_repo(REPO_ID, exist_ok=True)`).

    PYTHONPATH=src .venv/bin/python scripts/exp045_upload_release.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "pearsonkyle/tmax-27b-imatrix-MTP-GGUF"
PKG = Path(__file__).resolve().parents[1] / "uploads" / "pearsonkyle" / (
    "tmax-27b-imatrix-MTP-GGUF"
)

# The 6 shipped quants (Q2_K plain is a reference-only anchor in the README, not
# uploaded). These are the RECALIBRATED builds (2026-07); re-pushing the same
# filenames overwrites the remote blobs in place.
NEW_GGUFS = [
    "tmax-27b-IQ2_XS.gguf",
    "tmax-27b-IQ2_M.gguf",
    "tmax-27b-Q2_K_S.gguf",
    "tmax-27b-IQ3_M.gguf",
    "tmax-27b-IQ4_XS.gguf",
    "tmax-27b-Q5_K_M.gguf",
]
TEXT_FILES = [
    "README.md",
    "MTP/README.md",
    "calibration_data/corpus.cal.txt",
    "calibration_data/corpus.eval.general.txt",
    "calibration_data/corpus.eval.tools.txt",
    "calibration_data/corpora_audit.json",
    "calibration_data/README.md",
]


def main() -> int:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("no HF token in env (HUGGING_FACE_HUB_TOKEN / HF_TOKEN)")

    ops: list = []
    for name in NEW_GGUFS + TEXT_FILES:
        src = PKG / name
        if not src.exists():
            raise FileNotFoundError(src)
        ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(src)))

    print(f"repo={REPO_ID}")
    print(f"  add: {len(NEW_GGUFS)} GGUF + {len(TEXT_FILES)} text")
    total_gib = sum((PKG / g).stat().st_size for g in NEW_GGUFS) / 1024**3
    print(f"  ~{total_gib:.1f} GiB of GGUF to upload (LFS dedup may skip unchanged blobs)")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=ops,
        commit_message=(
            "Recalibrated (2026-07): rebuild all 6 GGUFs on a corrected 500k-token "
            "agentic-coding imatrix (fixed strict-chat-template window packing that "
            "had collapsed calibration to ~37k wiki-dominant tokens); refresh "
            "calibration_data + README"
        ),
    )
    print(f"\nDONE → {getattr(info, 'commit_url', info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

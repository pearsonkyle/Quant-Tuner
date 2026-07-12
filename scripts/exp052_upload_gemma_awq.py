"""exp-052: swap gemma-4-31B IQ2_M to the AWQ build on HF + updated README.

Replaces the shipped imatrix IQ2_M with the AWQ build (same 2.85 bpw / 10.9 GB)
— chosen by a direct tool-call fidelity test (param-acc +54%, far tighter
run-to-run variance) despite slightly worse static median-KLD. Re-pushing the
same filename overwrites the remote blob; the other quants + mmproj + mtp are
unchanged and NOT re-uploaded.

    PYTHONPATH=src .venv/bin/python scripts/exp052_upload_gemma_awq.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "pearsonkyle/gemma4-31b-imatrix-mtp-GGUF"
PKG = Path(__file__).resolve().parents[1] / "uploads" / "pearsonkyle" / (
    "gemma-4-31b-it-imatrix-GGUF"
)

FILES = [
    "gemma-4-31B-it-IQ2_M.gguf",  # now the AWQ build (overwrites the imatrix blob)
    "README.md",
]


def main() -> int:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("no HF token in env (HUGGING_FACE_HUB_TOKEN / HF_TOKEN)")

    ops: list = []
    for name in FILES:
        src = PKG / name
        if not src.exists():
            raise FileNotFoundError(src)
        ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(src)))

    gib = (PKG / FILES[0]).stat().st_size / 1024**3
    print(f"repo={REPO_ID}\n  replace: {FILES[0]} (~{gib:.1f} GiB, AWQ) + README.md")

    api = HfApi(token=token)
    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=ops,
        commit_message=(
            "Swap IQ2_M to AWQ build: +54% tool-argument accuracy and far tighter "
            "run-to-run variance vs plain 2-bit imatrix at the same 2.85 bpw. Static "
            "KLD/PPL slightly favored imatrix but mispredicted agentic tool-calling; "
            "AWQ chosen by a tool-call fidelity replay. README: add Method row, "
            "AWQ static column, tool-call table, how-it-was-made."
        ),
    )
    print(f"\nDONE -> {getattr(info, 'commit_url', info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

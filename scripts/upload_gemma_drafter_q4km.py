"""Push the Q4_K_M MTP drafter swap + README to the gemma-4-31B repo.

exp-049 showed the drafter can be quantized to Q4_K_M with zero acceptance
penalty (statistically identical to Q8_0 at every draft depth), saving ~156 MB.
This ships Q4_K_M as the root auto-discovered drafter and updates the README's
speculative-decoding section (drafter-quant study + figure).

Single atomic create_commit: the (now Q4_K_M) drafter, the new figure, README.

    PYTHONPATH=src .venv/bin/python scripts/upload_gemma_drafter_q4km.py
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

# Files to (re)upload. The drafter keeps its auto-discovery filename but now
# carries Q4_K_M weights, replacing the Q8_0 blob on the remote.
FILES = [
    "mtp-gemma-4-31B-it.gguf",          # now Q4_K_M (was Q8_0)
    "drafter-quant-acceptance.png",     # new figure referenced by README
    "README.md",
]


def main() -> int:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)  # falls back to the cached `hf auth login` token

    ops = []
    for name in FILES:
        src = PKG / name
        if not src.exists():
            raise FileNotFoundError(src)
        ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(src)))

    drafter_mib = (PKG / "mtp-gemma-4-31B-it.gguf").stat().st_size / 1024**2
    print(f"repo={REPO_ID}")
    print(f"  add/replace: {', '.join(FILES)}")
    print(f"  drafter = {drafter_mib:.0f} MiB (Q4_K_M)")

    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=ops,
        commit_message=(
            "Ship Q4_K_M MTP drafter (zero acceptance penalty vs Q8_0, -156 MB); "
            "add drafter-quantization acceptance study + figure to README"
        ),
    )
    print(f"\nDONE -> {getattr(info, 'commit_url', info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

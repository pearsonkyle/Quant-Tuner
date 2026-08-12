"""Push the Qwopus3.6-27B-Coder imatrix+MTP release to HF in one commit.

Six GGUFs (IQ2_XS, IQ2_M, Q2_K_S, Q2_K, IQ3_M, IQ4_XS), each with the MTP head
bundled at Q8_0; terminal-quant filenames (`…-IQ4_XS.gguf` → `ollama run …:IQ4_XS`
works). IQ3_M/IQ4_XS were added by scripts/exp041_extend_iq3m_iq4xs.py on the same
hybrid imatrix + calibration corpora as the 2-bit rows.

Pushes to the existing repo (name kept). Single atomic create_commit: 6 LFS
adds (4 dedup, 2 new) + text adds.

    PYTHONPATH=src .venv/bin/python scripts/upload_qwopus36_release.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO_ID = "pearsonkyle/Qwopus3.6-27B-Coder-2bit-MTP-GGUF"
PKG = Path(__file__).resolve().parents[1] / "uploads" / "pearsonkyle" / (
    "Qwopus3.6-27B-Coder-2bit-MTP-GGUF"
)

NEW_GGUFS = [
    "Qwopus3.6-27B-Coder-IQ2_XS.gguf",
    "Qwopus3.6-27B-Coder-IQ2_M.gguf",
    "Qwopus3.6-27B-Coder-Q2_K_S.gguf",
    "Qwopus3.6-27B-Coder-Q2_K.gguf",
    "Qwopus3.6-27B-Coder-IQ3_M.gguf",
    "Qwopus3.6-27B-Coder-IQ4_XS.gguf",
]
TEXT_FILES = [
    "README.md",
    "MTP/README.md",
    "calibration_data/corpus.cal.txt",
    "calibration_data/corpus.eval.txt",
    "calibration_data/corpus.eval.general.txt",
    "calibration_data/corpus.eval.tools.txt",
    "calibration_data/corpora_audit.json",
    "calibration_data/README.md",
]
# Fresh/renamed repo carries no stale GGUF names to remove.
OLD_GGUFS: list[str] = []


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
    for name in OLD_GGUFS:
        ops.append(CommitOperationDelete(path_in_repo=name))

    print(f"repo={REPO_ID}")
    print(f"  add: {len(NEW_GGUFS)} GGUF + {len(TEXT_FILES)} text")
    print(f"  delete: {len(OLD_GGUFS)} old GGUF")
    total_gib = sum((PKG / g).stat().st_size for g in NEW_GGUFS) / 1024**3
    print(f"  ~{total_gib:.1f} GiB of GGUF to upload (LFS dedup may skip unchanged blobs)")

    api = HfApi(token=token)
    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=ops,
        commit_message=(
            "Re-quant on windowed corpus; tools-KLD table + calibration-scope "
            "disclaimer; terminal-quant filenames for Ollama tags; refresh "
            "calibration_data (tools/general eval corpora)"
        ),
    )
    print(f"\nDONE → {getattr(info, 'commit_url', info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

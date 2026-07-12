"""Push the exp-051 Ornith-1.0-9B ladder update to HF in one atomic commit.

Adds the two NEW quants (terminal-quant names so `ollama run …:IQ2_M` /
`:IQ4_XS` resolve) + the rewritten README + the refreshed calibration corpus.
The 2-bit is an **AWQ** build (agentic tool-call rescue: param-acc 5%→33% vs
plain 2-bit imatrix); IQ4_XS is hybrid imatrix. Q5_K_M + the vision mmproj are
already on the repo unchanged, so they're NOT re-uploaded.

⚠️ NOT run automatically. Requires HUGGING_FACE_HUB_TOKEN / HF_TOKEN in the env.
License is MIT (inherited from deepreinforce-ai/Ornith-1.0-9B) — already in the
README frontmatter.

    PYTHONPATH=src .venv/bin/python scripts/exp051_upload_release.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "pearsonkyle/Ornith-1.0-9B-imatrix-GGUF"
PKG = Path(__file__).resolve().parents[1] / "uploads" / "pearsonkyle" / (
    "Ornith-1.0-9B-imatrix-GGUF"
)

# NEW quants (AWQ 2-bit + imatrix 4-bit). Q5_K_M + mmproj are unchanged on the
# repo and deliberately omitted (no need to re-push ~7 GiB of identical blobs).
NEW_GGUFS = [
    "Ornith-1.0-9B-IQ2_M.gguf",   # AWQ (the only 2-bit that tool-calls)
    "Ornith-1.0-9B-IQ4_XS.gguf",  # hybrid imatrix (best size/quality)
]
TEXT_FILES = [
    "README.md",
    "calibration_data/corpus.cal.txt",       # rebuilt (schema-dedup + wiki interleave)
    "calibration_data/corpora_audit.json",
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

    total_gib = sum((PKG / g).stat().st_size for g in NEW_GGUFS) / 1024**3
    print(f"repo={REPO_ID}")
    print(f"  add: {len(NEW_GGUFS)} GGUF + {len(TEXT_FILES)} text  (~{total_gib:.1f} GiB GGUF)")
    for n in NEW_GGUFS + TEXT_FILES:
        print(f"    + {n}")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=ops,
        commit_message=(
            "Add IQ2_M (AWQ) + IQ4_XS quants; method chosen by agentic tool-call "
            "fidelity. At 2-bit, plain imatrix loses tool-argument precision "
            "(param-acc 5%) despite good KLD; AWQ's activation-aware scaling "
            "recovers it (33%), so IQ2_M ships AWQ. IQ4_XS (imatrix) matches "
            "Q5_K_M agentically at 20% smaller. Refresh README + calibration corpus."
        ),
    )
    print(f"\nDONE → {getattr(info, 'commit_url', info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

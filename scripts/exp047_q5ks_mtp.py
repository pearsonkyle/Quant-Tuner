"""exp-047b addendum: MTP draft-acceptance for the Q5_K_S trunk.

Reuses exp-046's `run_config`; only the Q5_K_S quant, pointed at the root
`mtp-gemma-4-31B-it.gguf` drafter.

    .venv/bin/python scripts/exp047_q5ks_mtp.py \
        --pkg uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF \
        --reps 5 --max-tokens 200 --out out/exp-047/acceptance_q5ks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import exp046_mtp_acceptance as e46  # noqa: E402

QUANT_LABEL = "Q5_K_S"
QUANT_FILE = "gemma-4-31B-it-Q5_K_S.gguf"
DRAFTER_FILE = "mtp-gemma-4-31B-it.gguf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", type=Path,
                    default=REPO / "uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=REPO / "out/exp-047/acceptance_q5ks.json")
    args = ap.parse_args()

    trunk = args.pkg / QUANT_FILE
    drafter = args.pkg / DRAFTER_FILE
    for p, what in ((trunk, "trunk"), (drafter, "drafter")):
        if not p.exists():
            print(f"error: {what} not found: {p}", file=sys.stderr)
            return 2

    log_dir = args.out.parent / "logs"
    rows = []
    for n in e46.N_VALUES:
        print(f"\n>>> {QUANT_LABEL}  n_max={n}", flush=True)
        row = e46.run_config(trunk, drafter, n, args.reps, args.max_tokens,
                             args.ctx, log_dir / f"{QUANT_LABEL}.n{n}.log")
        ar, mal = row["accept_rate"], row["mean_accept_len"]
        print(f"    accept_rate={ar*100:.1f}%  ({row['draft_accepted']}/{row['draft_total']})  "
              f"mean_accept_len={mal:.2f}" if ar is not None else "    no draft stats")
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({QUANT_LABEL: rows}, indent=2))
    print(f"\nwrote {args.out}")

    print("\n=== acceptance rate (%) ===")
    print("| Quant | " + " | ".join(f"n={n}" for n in e46.N_VALUES) + " |")
    print("|" + "---|" * (len(e46.N_VALUES) + 1))
    cells = [f"{r['accept_rate']*100:.1f}%" if r["accept_rate"] is not None else "—" for r in rows]
    print(f"| {QUANT_LABEL} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

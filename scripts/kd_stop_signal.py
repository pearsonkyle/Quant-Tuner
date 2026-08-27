"""What a KD table teaches about STOPPING — the decision this pipeline keeps breaking.

`coverage` says the stored top-K captured the teacher's mass; it says nothing about
whether the teacher is right where it matters. Termination is the failure mode here
(every trained run so far drove `P(<|im_end|> | completed sentence)` from 0.009 to
~0.95), so before three GPU-hours distil against a table, measure the one thing:

    teacher P(stop) AT the corpus's real stop targets  vs  everywhere else

A teacher that cannot separate those two teaches the student nothing about stopping, and
the KL will happily optimize while the stopping policy drifts. A large ratio means the
signal is there to be learned.

This is NOT what `teacher_stop_probe.py` measures. That scores a handful of synthetic
positions; this scores every labeled position in the corpus the student actually trains
on. The two can disagree and the corpus-conditioned one is what KD transfers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def stop_signal(table: Path, corpus: Path, stop_id: int = 106) -> dict:
    t = torch.load(table, weights_only=False, mmap=True)
    b = torch.load(corpus, weights_only=False, mmap=True)
    lab = b["labels"][:, 1:]                     # targets, aligned with keep positions
    win, pos, idx, logp = t["win"], t["pos"], t["idx"], t["logp"]
    hit = idx == stop_id
    if not bool((hit.sum(-1) == 1).all()):
        raise ValueError(f"stop id {stop_id} is not forced into every support row — "
                         f"rebuild the table with --include-ids {stop_id}")
    p_stop = logp.float()[hit].exp()
    is_stop = lab[win.long(), pos.long()] == stop_id

    out = {"table": str(table), "corpus": str(corpus), "stop_id": stop_id,
           "teacher": t.get("teacher"), "n_rows": int(len(p_stop)),
           "n_stop_targets": int(is_stop.sum())}
    for name, m in (("at_stop_target", is_stop), ("elsewhere", ~is_stop)):
        v = p_stop[m]
        # quantile() caps at ~16M elements; a deterministic stride keeps it exact enough
        # for a distribution summary without a sort of 6M floats per call.
        s = v[:: max(1, v.numel() // 1_000_000)]
        out[name] = {"n": int(v.numel()), "mean": float(v.mean()),
                     **{f"p{int(q * 100):02d}": float(x) for q, x in
                        zip(Q, torch.quantile(s, torch.tensor(Q)), strict=True)}}
    out["ratio_mean"] = out["at_stop_target"]["mean"] / max(1e-12, out["elsewhere"]["mean"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--stop-id", type=int, default=106)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = stop_signal(args.table, args.corpus, args.stop_id)
    a, e = d["at_stop_target"], d["elsewhere"]
    print(f"[kd-stop] teacher {d['teacher']}  ({d['n_stop_targets']:,} stop targets of "
          f"{d['n_rows']:,} rows)")
    print(f"  P(stop) at a stop target  mean {a['mean']:.4f}  median {a['p50']:.4f}  "
          f"p75 {a['p75']:.4f}")
    print(f"  P(stop) elsewhere         mean {e['mean']:.6f}  median {e['p50']:.6f}")
    print(f"  discriminative ratio      {d['ratio_mean']:,.0f}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(d, indent=1) + "\n")
    print(f"[kd-stop] -> {args.out}")


if __name__ == "__main__":
    main()

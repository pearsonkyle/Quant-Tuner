"""Capture the dense teacher's top-K distribution at rep-context span positions.

Produces the RepKD table for --steer-rep-kd: tail-bucket KL at the loop-risk decision
points, turning the repetition hinge ("don't be certain about the verbatim repeat")
into "match what the teacher does in this state". The table is fingerprinted to the
exact RepBatch contexts — the trainer refuses a mismatch, so pass the SAME
--n/--seed/--k the run will use.

    PYTHONPATH=src .venv/bin/python scripts/capture_rep_teacher.py \
        --teacher SWE-Lego/SWE-Lego-Qwen3-32B --k 2,3,4,5 \
        --out out/exp-058/kd/rep_teacher_32b_k2345.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.kd_precompute import _topk_rows, load_teacher  # noqa: E402
from quant_tuner.qat.steer import RepBatch, rep_fingerprint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student-model", default="out/exp-057/model",
                    help="tokenizer source — must match the training run's model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--k", default="2,3,4,5", help="identical-round counts, round-robin")
    ap.add_argument("--bank", default=None,
                    help="real-material bank (build_rep_bank.py) — must match training")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student_model)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    bank = None
    if args.bank:
        import json
        bank = json.loads(Path(args.bank).read_text())
    batch = RepBatch.build(tok, n=args.n, seed=args.seed, k=ks, bank=bank)
    fp = rep_fingerprint(batch)
    print(f"[rep-kd] {batch.ids.shape[0]} contexts (k={ks}), fingerprint {fp}", flush=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = load_teacher(args.teacher, device=args.device, dtype=dtype)
    model.eval()

    idxs, logps, tails, off = [], [], [], [0]
    with torch.no_grad():
        for i in range(batch.ids.shape[0]):
            lo, hi = int(batch.span[i, 0]), int(batch.span[i, 1])
            lg = model(input_ids=batch.ids[i:i + 1].to(args.device),
                       attention_mask=batch.attn[i:i + 1].to(args.device)).logits[0]
            vals, ids_k, tail = _topk_rows(lg[lo - 1:hi - 1], args.topk, None)
            idxs.append(ids_k.to(torch.int32).cpu())
            logps.append(vals.to(torch.float16).cpu())
            tails.append(tail.to(torch.float16).cpu())
            off.append(off[-1] + (hi - lo))
            # what the teacher thinks of the verbatim repeat, for the record
            tgt = batch.ids[i, lo:hi].to(args.device)
            lp = torch.log_softmax(lg[lo - 1:hi - 1].float(), -1).gather(
                -1, tgt.unsqueeze(-1)).mean()
            print(f"[rep-kd] row {i} (k={ks[i % len(ks)]}): teacher P(verbatim repeat) "
                  f"= {lp.exp().item():.4f}", flush=True)

    payload = {
        "idx": torch.cat(idxs), "logp": torch.cat(logps), "tail": torch.cat(tails),
        "row_off": torch.tensor(off, dtype=torch.long),
        "fingerprint": fp, "teacher": str(args.teacher), "topk": args.topk,
        "n": args.n, "seed": args.seed, "k": ks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[rep-kd] saved {payload['idx'].shape[0]} positions x top-{args.topk} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()

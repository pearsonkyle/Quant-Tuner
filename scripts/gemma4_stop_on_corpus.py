"""P(stop) at REAL corpus stop targets — the probe the synthetic one is a proxy for.

`stop_probe.py` scores seven hand-written positions. That is cheap enough to run every
25 steps during training, and it is what every abort in this study fired on — but it is
seven prompts, and a model can fail them while still ending real turns correctly (or
pass them while looping, which is how the shipped Bonsai model looked). Before spending
more GPU on fixing what the synthetic probe reports, measure the thing it stands in for.

Method, deliberately mirroring `kd_stop_signal.py` so student and teacher are read the
same way: sample real stop targets from the **held-out val corpus**, take the ``--ctx``
tokens of real context ending just before each, and read P(<turn|>) at the position that
must predict it. Sample the same number of ordinary supervised positions as a control —
a model that raised P(stop) *everywhere* has not learned to terminate, it has learned to
stop, and only the pair of numbers tells those apart.

Runs on CPU beside a live trainer.

    python scripts/gemma4_stop_on_corpus.py --ckpt <trained_latents.pt> \\
        --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj --out stop_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemma4_stage_damage import latent_modules, load_latents  # noqa: E402

from quant_tuner.qat.train import wrap_model  # noqa: E402

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
STOP = 106


def sample_positions(blob, n: int, ctx: int, seed: int) -> tuple[list, list]:
    """(stop targets, ordinary supervised positions) as (window, index) pairs.

    Both drawn with at least ``ctx`` tokens of real context in front of them, so the two
    populations differ only in what the next token is.
    """
    lab = blob["labels"][:, 1:]
    g = torch.Generator().manual_seed(seed)
    out = []
    for want_stop in (True, False):
        m = (lab == STOP) if want_stop else ((lab != -100) & (lab != STOP))
        m[:, :ctx] = False                      # need a full context window behind it
        idx = m.nonzero()
        pick = torch.randperm(idx.shape[0], generator=g)[:n]
        out.append([(int(idx[i][0]), int(idx[i][1])) for i in pick])
    return out[0], out[1]


@torch.no_grad()
def p_stop(model, blob, pos, ctx: int) -> torch.Tensor:
    """P(<turn|>) at each (window, index) — the distribution over the NEXT token."""
    ids_all = blob["ids"]
    out = []
    for w, i in pos:
        # labels are targets for ids[1:], so label index i is predicted from ids[..i]
        window = ids_all[w, i + 1 - ctx : i + 1][None]
        lg = model(window).logits[0, -1].float()
        out.append(torch.softmax(lg, -1)[STOP])
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", type=Path,
                    default=Path("out/gemma4-ternary/corpus_sft_gemma4_val_32768.pt"))
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="omit for the shipped model; --ternary-layers alone gives the "
                         "untrained ternarization")
    ap.add_argument("--ternary-layers", default=None)
    ap.add_argument("--dense-kind", action="append", default=[])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import Gemma4ForConditionalGeneration

    blob = torch.load(args.corpus, weights_only=False, mmap=True)
    stops, others = sample_positions(blob, args.n, args.ctx, args.seed)
    label = args.label or (args.ckpt.parent.name if args.ckpt else "shipped")
    print(f"[{label}] {len(stops)} real stop targets + {len(others)} ordinary positions, "
          f"ctx {args.ctx}, from {args.corpus.name}", flush=True)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    model.eval()
    if args.ternary_layers:
        wrap_model(model, 0, layer_spec=args.ternary_layers,
                   ternary_spec=args.ternary_layers, dense_kinds=tuple(args.dense_kind))
        print(f"[{label}] wrapped {len(latent_modules(model))} ternary latents", flush=True)
    if args.ckpt:
        n_t, n_d = load_latents(model, args.ckpt)
        print(f"[{label}] loaded {n_t} latents + {n_d} dense", flush=True)

    t0 = time.time()
    at = p_stop(model, blob, stops, args.ctx)
    el = p_stop(model, blob, others, args.ctx)
    res = {"label": label, "ckpt": str(args.ckpt) if args.ckpt else None,
           "n": args.n, "ctx": args.ctx, "seed": args.seed,
           "at_stop_target": {"mean": float(at.mean()), "median": float(at.median()),
                              "p25": float(at.quantile(0.25)), "p75": float(at.quantile(0.75)),
                              "frac_top1": float((at > 0.5).float().mean())},
           "elsewhere": {"mean": float(el.mean()), "median": float(el.median()),
                         "p75": float(el.quantile(0.75))}}
    res["ratio_mean"] = res["at_stop_target"]["mean"] / max(1e-12, res["elsewhere"]["mean"])
    a, e = res["at_stop_target"], res["elsewhere"]
    print(f"[{label}] P(stop) AT a real stop target  mean {a['mean']:.4f}  "
          f"median {a['median']:.4f}  p25/p75 {a['p25']:.4f}/{a['p75']:.4f}  "
          f"|  >0.5 on {a['frac_top1']:.0%} of them", flush=True)
    print(f"[{label}] P(stop) elsewhere              mean {e['mean']:.6f}  "
          f"median {e['median']:.6f}", flush=True)
    print(f"[{label}] discriminative ratio {res['ratio_mean']:,.0f}x  "
          f"({time.time()-t0:.0f}s)", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

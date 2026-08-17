"""End-to-end ternarization damage, per layer and per tensor kind, with no training.

The weight-space probe (``gemma4_ternary_damage.py``) says how far each tensor moves
when it is put on the ternary grid. It does NOT say how much that movement matters,
and on gemma-4-E4B it turns out to say almost nothing useful: every tensor scores
~0.43 relative Frobenius error, the value a Gaussian gives, so the ranking it induces
is noise. This script measures the quantity a progressive schedule actually needs —
**how much the model's output distribution moves when you ternarize one group of
tensors and nothing else.**

Method: hold the dense model's final logits as the reference, then for each candidate
group (one decoder layer, or one tensor kind across all layers) swap that group's
weights for their TWN projection, re-run the same tokens, and report
``KLD(reference || candidate)`` plus top-1 agreement and NLL. Restore, repeat. One
forward pass per group, no gradients, no GPU — it runs while the card is busy.

The output is the ordering a gradual schedule should follow: ternarize the groups
with the smallest KLD first, leave the largest for last (or leave them dense). The
``--cumulative`` mode then walks that ordering and reports how damage accumulates,
which is what says whether "fully ternary" is reachable at all.

Eval text comes from OUR corpus (``sft.jsonl.gz``), rendered through the model's own
chat template, and defaults to a **non-train split** so the number is not read on
data a later fine-tune would train on.

Usage::

    python scripts/gemma4_layer_damage.py --split test --windows 2 --window 2048 \
        --out out/gemma4-ternary/layer_damage.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import torch

from quant_tuner.qat.ternary import DEFAULT_GROUP_SIZE, TWN_THRESH, ternarize_group

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
DEFAULT_SFT = Path("out/corpora/qwen3-universal-v2/sft.jsonl.gz")


# ---------------------------------------------------------------- eval windows


def _normalize(msgs: list[dict]) -> list[dict]:
    """Make rows safe for gemma-4's template.

    The template raises when ``tool_calls[].function.arguments`` is a JSON *string*
    rather than a mapping, which our logs carry both ways.
    """
    out = []
    for m in msgs:
        m = dict(m)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except Exception:
                    fn["arguments"] = {"_raw": a}
        out.append(m)
    return out


def build_windows(
    tok, sft: Path, *, split: str, window: int, n_windows: int
) -> torch.Tensor:
    """``[n_windows, window]`` token ids from ``split`` of the SFT corpus.

    **One window per conversation**, taken from the head of each. Our sessions run to
    tens of thousands of tokens, so concatenating them into one stream and slicing
    would draw every window from the first conversation or two — a damage number read
    off a single session, which is not a distribution. Taking one window each spreads
    the eval over ``n_windows`` distinct conversations instead.
    """
    got: list[list[int]] = []
    seen = 0
    with gzip.open(sft, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != split:
                continue
            seen += 1
            try:
                text = tok.apply_chat_template(
                    _normalize(r["messages"]), tools=r.get("tools") or None,
                    tokenize=False,
                )
            except Exception:
                continue
            ids = tok(text, add_special_tokens=False).input_ids
            if len(ids) < window:      # short rows can't fill a window on their own
                continue
            got.append(ids[:window])
            if len(got) == n_windows:
                break
    if len(got) < n_windows:
        raise SystemExit(
            f"split={split!r}: only {len(got)} of {seen} conversations reach "
            f"{window} tokens, need {n_windows}"
        )
    print(f"[eval] {n_windows} windows of {window} tokens, one from each of "
          f"{n_windows} distinct conversations (split={split})", flush=True)
    return torch.tensor(got)


# ---------------------------------------------------------------- measurement


@torch.no_grad()
def _logits(model, windows: torch.Tensor) -> list[torch.Tensor]:
    """Final logits per window, kept as fp16 on CPU (a 262k-vocab fp32 tensor is 2 GB)."""
    return [model(w[None]).logits[0].to(torch.float16) for w in windows]


@torch.no_grad()
def compare(ref: list[torch.Tensor], cand: list[torch.Tensor],
            windows: torch.Tensor) -> dict[str, float]:
    """KLD(ref||cand), top-1 agreement and candidate NLL, pooled over all positions.

    Reductions are fp32 and chunked over positions for the same reason
    ``bench/kld_hf.py`` chunks: a ``[2048, 262144]`` fp32 tensor is 2 GB and two of
    them are live at once.
    """
    kld_sum = agree = nll_sum = 0.0
    n = 0
    for r16, c16, w in zip(ref, cand, windows):
        for i in range(0, r16.shape[0] - 1, 256):
            r = r16[i : i + 256].float()
            c = c16[i : i + 256].float()
            lr = torch.log_softmax(r, -1)
            lc = torch.log_softmax(c, -1)
            kld_sum += float((lr.exp() * (lr - lc)).sum(-1).sum())
            agree += float((r.argmax(-1) == c.argmax(-1)).sum())
            tgt = w[i + 1 : i + 1 + r.shape[0]]
            nll_sum += float(-lc[torch.arange(len(tgt)), tgt].sum())
            n += r.shape[0]
    return {"kld": kld_sum / n, "top1_agree": agree / n,
            "nll": nll_sum / n, "ppl": float(torch.exp(torch.tensor(nll_sum / n)))}


def linears_of(model) -> dict[str, torch.nn.Linear]:
    """Every ``nn.Linear`` in the language model, keyed by dotted name."""
    lm = model.model.language_model
    return {n: m for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear)}


def _layer_of(name: str) -> int | None:
    m = re.match(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _kind_of(name: str) -> str:
    return re.sub(r"^layers\.\d+\.", "", name).removesuffix(".weight")


class Ternarized:
    """Context manager: TWN-project the named linears on entry, restore on exit."""

    def __init__(self, lins: dict[str, torch.nn.Linear], names: list[str],
                 *, group_size: int, thresh: float) -> None:
        self.lins, self.names = lins, names
        self.group_size, self.thresh = group_size, thresh
        self.saved: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for n in self.names:
            W = self.lins[n].weight
            if W.shape[1] % self.group_size:
                continue
            self.saved[n] = W.detach().clone()
            _, _, wh = ternarize_group(W.detach().float(), group_size=self.group_size,
                                       thresh=self.thresh)
            W.data.copy_(wh.to(W.dtype))
        return self

    def __exit__(self, *exc):
        for n, W0 in self.saved.items():
            self.lins[n].weight.data.copy_(W0)
        self.saved.clear()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    ap.add_argument("--split", default="test", help="SFT split for the eval text")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--thresh", type=float, default=TWN_THRESH)
    ap.add_argument("--threads", type=int, default=192)
    ap.add_argument("--mode", choices=["layers", "kinds", "both"], default="both")
    ap.add_argument("--cumulative", action="store_true",
                    help="after ranking layers, walk the ranking and report accumulation")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.model)
    windows = build_windows(tok, args.sft, split=args.split,
                            window=args.window, n_windows=args.windows)

    print(f"[load] {args.model} fp32 on cpu", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    model.eval()
    lins = linears_of(model)
    print(f"[load] {len(lins)} linear modules in language_model", flush=True)

    t0 = time.time()
    ref = _logits(model, windows)
    print(f"[ref] dense reference logits in {time.time()-t0:.0f}s", flush=True)

    groups: list[tuple[str, list[str]]] = []
    if args.mode in ("layers", "both"):
        by_layer: dict[int, list[str]] = {}
        for n in lins:
            li = _layer_of(n)
            if li is not None:
                by_layer.setdefault(li, []).append(n)
        groups += [(f"layer.{li:02d}", by_layer[li]) for li in sorted(by_layer)]
    if args.mode in ("kinds", "both"):
        by_kind: dict[str, list[str]] = {}
        for n in lins:
            by_kind.setdefault(_kind_of(n), []).append(n)
        groups += [(f"kind.{k}", v) for k, v in sorted(by_kind.items())]

    results = []
    for i, (label, names) in enumerate(groups, 1):
        t = time.time()
        with Ternarized(lins, names, group_size=args.group_size, thresh=args.thresh):
            m = compare(ref, _logits(model, windows), windows)
        m.update(group=label, n_tensors=len(names),
                 n_params=sum(lins[n].weight.numel() for n in names))
        results.append(m)
        print(f"[{i:3d}/{len(groups)}] {label:28s} kld={m['kld']:8.4f} "
              f"top1={m['top1_agree']:.3f} ppl={m['ppl']:8.2f} "
              f"({time.time()-t:.0f}s)", flush=True)

    layer_rows = [r for r in results if r["group"].startswith("layer.")]
    order = [r["group"] for r in sorted(layer_rows, key=lambda r: r["kld"])]
    print("\n=== layers, least-damaging first ===")
    for r in sorted(layer_rows, key=lambda r: r["kld"]):
        print(f"  {r['group']:12s} kld={r['kld']:8.4f} top1={r['top1_agree']:.3f}")

    cumulative = []
    if args.cumulative and layer_rows:
        by_layer = {f"layer.{_layer_of(n):02d}": None for n in lins if _layer_of(n) is not None}
        names_for = {}
        for n in lins:
            li = _layer_of(n)
            if li is not None:
                names_for.setdefault(f"layer.{li:02d}", []).append(n)
        acc: list[str] = []
        print("\n=== cumulative along the ranking ===")
        for k, g in enumerate(order, 1):
            acc += names_for[g]
            if k % 6 and k != len(order):
                continue
            with Ternarized(lins, acc, group_size=args.group_size, thresh=args.thresh):
                m = compare(ref, _logits(model, windows), windows)
            m.update(n_layers=k, last_added=g)
            cumulative.append(m)
            print(f"  {k:2d}/{len(order)} layers ternary  kld={m['kld']:9.4f} "
                  f"top1={m['top1_agree']:.3f} ppl={m['ppl']:10.2f}", flush=True)

    blob = {"model": args.model, "split": args.split, "window": args.window,
            "windows": args.windows, "group_size": args.group_size,
            "thresh": args.thresh, "results": results, "layer_order": order,
            "cumulative": cumulative}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

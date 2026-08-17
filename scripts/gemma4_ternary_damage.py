"""Weight-space ternarization damage profile for a dense (non-native-ternary) model.

The cheapest falsifying experiment in the gemma-4 ternary study, and the one that
answers the scheduling question directly: **which tensors move least when you put
them on the ternary grid?** It needs no forward pass, no GPU and no corpus — just
the checkpoint — so it runs while the card is busy.

For every ``nn.Linear`` weight it reports the relative Frobenius error of the
per-group TWN projection (:mod:`quant_tuner.qat.ternary`, the same quantizer the
trainer puts in the loop), plus the code histogram. Two readings:

* **Ranking.** Sort tensors by relative error and you have the depth/type order a
  progressive schedule should follow — ternarize the tensors that barely move
  first, leave the ones that move most for last (or never).
* **Level.** The absolute number says how far from the grid the model starts.
  A natively-ternary model scores ~0 here by construction (that is the step-0
  exactness property ``qat/ternary.py`` is built around); a dense bf16 model does
  not, and the gap is the size of the repair job.

Weight-space error is a PROXY, not the damage itself — it ignores how much each
tensor's output actually matters downstream, which is what an activation-weighted
or end-to-end perplexity probe measures. Read it as the cheap first cut that tells
you where to spend the expensive probes.

Usage::

    python scripts/gemma4_ternary_damage.py \
        --model google/gemma-4-E4B-it-qat-q4_0-unquantized \
        --out out/gemma4-ternary/damage.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from quant_tuner.qat.ternary import DEFAULT_GROUP_SIZE, TWN_THRESH, ternarize_group


def _snapshot_dir(model: str) -> Path:
    """Resolve a repo id or a local path to a directory holding the safetensors."""
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def _safetensors_files(d: Path) -> list[Path]:
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {d}")
    return files


def damage_for_weight(
    W: torch.Tensor, *, group_size: int, thresh: float
) -> dict[str, float]:
    """Relative TWN projection error and code statistics for one weight."""
    W32 = W.to(torch.float32)
    codes, _scale, w_hat = ternarize_group(W32, group_size=group_size, thresh=thresh)
    err = torch.linalg.vector_norm(W32 - w_hat)
    nrm = torch.linalg.vector_norm(W32)
    n = codes.numel()
    return {
        "rel_fro": float(err / nrm),
        "frac_zero": float((codes == 0).sum() / n),
        # Cosine similarity of the flattened weights: insensitive to a global scale
        # error, so it separates "wrong direction" from "wrong magnitude".
        "cos": float(
            torch.nn.functional.cosine_similarity(
                W32.reshape(1, -1), w_hat.reshape(1, -1)
            )
        ),
        "numel": float(n),
    }


def profile(
    model: str,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    thresh: float = TWN_THRESH,
    include: str = r"\.language_model\..*(proj|gate)\.weight$",
) -> dict:
    """Damage profile over every checkpoint tensor matching ``include``."""
    d = _snapshot_dir(model)
    pat = re.compile(include)
    rows: list[dict] = []
    for f in _safetensors_files(d):
        with safe_open(f, framework="pt") as h:
            for name in h.keys():  # noqa: SIM118 — safe_open is not iterable
                if not pat.search(name):
                    continue
                W = h.get_tensor(name)
                if W.ndim != 2:
                    continue
                if W.shape[1] % group_size != 0:
                    rows.append({"name": name, "skipped": "in%group != 0"})
                    continue
                rows.append({"name": name, **damage_for_weight(
                    W, group_size=group_size, thresh=thresh)})
    rows.sort(key=lambda r: r.get("rel_fro", -1.0))
    return {
        "model": model,
        "group_size": group_size,
        "thresh": thresh,
        "n_tensors": len(rows),
        "tensors": rows,
    }


def _layer_of(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _kind_of(name: str) -> str:
    return re.sub(r"^.*\.layers\.\d+\.", "", name).removesuffix(".weight") or name


def summarize(prof: dict) -> str:
    """Human-readable roll-ups: by tensor kind and by depth."""
    rows = [r for r in prof["tensors"] if "rel_fro" in r]
    out = [f"# {prof['model']}  g{prof['group_size']}  thresh {prof['thresh']}",
           f"# {len(rows)} tensors profiled", ""]

    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(_kind_of(r["name"]), []).append(r)
    out.append(f"{'tensor kind':38s} {'n':>4s} {'rel_fro':>9s} {'min':>8s} "
               f"{'max':>8s} {'frac0':>7s} {'params':>9s}")
    for kind, rs in sorted(by_kind.items(),
                           key=lambda kv: sum(r["rel_fro"] for r in kv[1]) / len(kv[1])):
        e = [r["rel_fro"] for r in rs]
        z = sum(r["frac_zero"] for r in rs) / len(rs)
        p = sum(r["numel"] for r in rs)
        out.append(f"{kind:38s} {len(rs):4d} {sum(e)/len(e):9.4f} {min(e):8.4f} "
                   f"{max(e):8.4f} {z:7.3f} {p/1e6:8.1f}M")

    out += ["", f"{'layer':>5s} {'rel_fro':>9s} {'min':>8s} {'max':>8s}  worst tensor"]
    by_layer: dict[int, list[dict]] = {}
    for r in rows:
        li = _layer_of(r["name"])
        if li is not None:
            by_layer.setdefault(li, []).append(r)
    for li in sorted(by_layer):
        rs = by_layer[li]
        e = [r["rel_fro"] for r in rs]
        worst = max(rs, key=lambda r: r["rel_fro"])
        out.append(f"{li:5d} {sum(e)/len(e):9.4f} {min(e):8.4f} {max(e):8.4f}  "
                   f"{_kind_of(worst['name'])}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E4B-it-qat-q4_0-unquantized")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--thresh", type=float, default=TWN_THRESH)
    ap.add_argument("--include", default=r"\.language_model\..*(proj|gate)\.weight$",
                    help="regex over checkpoint tensor names")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    prof = profile(args.model, group_size=args.group_size, thresh=args.thresh,
                   include=args.include)
    print(summarize(prof))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prof, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

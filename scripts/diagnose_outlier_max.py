"""Phase A of exp-006: characterize the per-tensor-class distribution of
`max|a|` and `√E[a⁴]` from the exp-002 forward_stats.npz.

Goal: identify whether the `attn_qkv` and `attn_gate` projections (mapped
from Qwen3-Next `linear_attn.in_proj_qkv` / `in_proj_z`) have pathological
single-channel spikes that distort the L1-normalized outlier_max signal.

Output: per-tensor-class summary (mean, std, max, p99, max/p99 skew ratio)
to stdout AND `out/exp-006/diagnostic.md`. Flags classes with `max/p99 > 10`
(extreme skew) as suspects.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.calibrate.imatrix import ForwardStats

STATS = REPO / "out/exp-002/Jackrong__Qwopus3.5-9B-Coder/forward_stats.npz"
OUT_MD = REPO / "out/exp-006/diagnostic.md"

# Map `blk.N.{name}.weight` -> tensor class for grouping.
_CLASS_RE = re.compile(r"^blk\.\d+\.(?P<cls>[^.]+)\.weight$")


def _classify(name: str) -> str:
    m = _CLASS_RE.match(name)
    return m.group("cls") if m else name


def _per_channel_skew(v: np.ndarray) -> dict[str, float]:
    """Compute summary stats over one channel-vector."""
    if v.size == 0 or not np.isfinite(v).any():
        return {"mean": float("nan"), "std": float("nan"),
                "max": float("nan"), "p99": float("nan"),
                "max_over_p99": float("nan"), "max_over_median": float("nan")}
    finite = v[np.isfinite(v)]
    p99 = float(np.percentile(finite, 99))
    median = float(np.percentile(finite, 50))
    mx = float(finite.max())
    return {
        "mean": float(finite.mean()),
        "std":  float(finite.std()),
        "max":  mx,
        "p99":  p99,
        "max_over_p99":    (mx / p99) if p99 > 0 else float("inf"),
        "max_over_median": (mx / median) if median > 0 else float("inf"),
    }


def _aggregate_by_class(stats: ForwardStats, key: str) -> dict[str, dict]:
    """For each tensor class, aggregate per-channel stats of the given signal."""
    source = stats.e_a4 if key == "e_a4" else stats.max_abs
    by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for tname, vec in source.items():
        v = np.sqrt(np.maximum(vec, 0.0)) if key == "e_a4" else vec
        by_class[_classify(tname)].append(v.astype(np.float32))

    out: dict[str, dict] = {}
    for cls, vecs in by_class.items():
        all_vals = np.concatenate(vecs)
        out[cls] = {
            "n_tensors": len(vecs),
            "n_channels": int(all_vals.size),
            **_per_channel_skew(all_vals),
        }
    return out


def _fmt_row(cls: str, d: dict, sus: bool) -> str:
    marker = " ⚠" if sus else ""
    return (f"| {cls}{marker} | {d['n_tensors']} | {d['n_channels']:,} | "
            f"{d['mean']:.4g} | {d['std']:.4g} | {d['max']:.4g} | "
            f"{d['p99']:.4g} | {d['max_over_p99']:.2f}× | {d['max_over_median']:.2f}× |")


def _section(stats: ForwardStats, signal_label: str, signal_key: str,
             threshold: float = 10.0) -> tuple[list[str], list[str]]:
    by_cls = _aggregate_by_class(stats, signal_key)
    lines: list[str] = [
        f"### `{signal_label}` per-channel distribution by tensor class\n",
        "| class | n_tensors | n_channels | mean | std | max | p99 | max/p99 | max/median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    suspects: list[str] = []
    for cls in sorted(by_cls.keys()):
        d = by_cls[cls]
        sus = d["max_over_p99"] > threshold and np.isfinite(d["max_over_p99"])
        if sus:
            suspects.append(cls)
        lines.append(_fmt_row(cls, d, sus))
    lines.append(f"\n_⚠ = max/p99 > {threshold:.0f}× (extreme single-channel skew suspected)_\n")
    return lines, suspects


def main() -> int:
    if not STATS.exists():
        print(f"missing {STATS} — run scripts/run_exp002_outliers.py first", file=sys.stderr)
        return 2
    stats = ForwardStats.load(STATS)
    print(f"Loaded forward_stats: {len(stats.max_abs)} tensors\n")

    md: list[str] = [
        "# exp-006 diagnostic: outlier_max regression root-cause\n",
        f"Source: `{STATS.relative_to(REPO)}`\n",
        f"Tensors: {len(stats.max_abs)}\n",
        "## Question\n",
        "When the HF↔GGUF mapping was extended from 52% → 100% coverage in "
        "exp-002, the `outlier_max` variant's KLD got *worse* (0.543 → 0.562) "
        "while `outlier_l4` improved. The hypothesis: the newly-hooked "
        "`attn_qkv` and `attn_gate` projections (mapped from Qwen3-Next's "
        "`linear_attn.in_proj_qkv` / `in_proj_z`) have extreme single-"
        "channel `max|a|` spikes that distort L1-normalized per-channel "
        "ranking.\n",
        "## Findings\n",
    ]

    max_section, max_suspects = _section(stats, "max|a|", "max_abs")
    md.extend(max_section)
    print("\n".join(max_section))

    l4_section, l4_suspects = _section(stats, "√E[a⁴]", "e_a4")
    md.extend(l4_section)
    print("\n".join(l4_section))

    md.append("## Interpretation\n")
    if max_suspects:
        md.append(
            f"- `max|a|` flagged extreme-skew classes: **{', '.join(max_suspects)}**. "
            "If these include `attn_qkv` and/or `attn_gate`, the exp-006 hypothesis "
            "(`outlier_max` regressed because of `linear_attn`-mapped tensors with "
            "single-channel spikes) is confirmed.\n"
        )
    else:
        md.append("- No tensor class shows `max|a|` skew over the 10× threshold. "
                  "The regression must have a different cause — consider rerunning "
                  "with a different threshold or rethinking the hypothesis.\n")
    if l4_suspects and l4_suspects != max_suspects:
        md.append(
            f"- `√E[a⁴]` also flagged: **{', '.join(l4_suspects)}**. "
            "Heavy-tail averaging didn't fully tame the skew here either.\n"
        )
    elif not l4_suspects:
        md.append("- `√E[a⁴]` shows no skewed classes — consistent with "
                  "`outlier_l4` IMPROVING when the same tensors were added.\n")

    md.append("\n## Recommended cells to run from `scripts/run_exp006.py`\n")
    if any(s in ("attn_qkv", "attn_gate") for s in max_suspects):
        md.append("- **F1, F2, F3** are well-motivated — run them.\n")
    else:
        md.append("- F1/F2/F3 (tensor-class fallback for outlier_max) may not "
                  "fire as expected — consider revising the fallback boundary.\n")
    md.append("- **H1–H4** (heavy-tail mixing) are independent of the diagnosis "
              "and worth running regardless.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"\nwrote {OUT_MD.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Knowledge probes: is a fact *erased* by quantization, or merely *suppressed*?

A probe is a short prompt with a known gold continuation (e.g. "The currency
used in the country shaped like a boot is" -> " Euro"). Running it through the
lens gives the gold token's rank at every layer. We classify each probe on a
quant as:

- ``correct``   — gold is the model's final top-1;
- ``suppressed`` — gold enters the lens top-k at late layers (the knowledge is
  present in the residual stream) but is not the final prediction;
- ``absent``    — gold never enters the lens top-k at any layer.

Comparing a quant against FP16, the interesting transition is
``correct -> suppressed`` (knowledge survives, readout degrades — recoverable
with mild steering) versus ``correct -> absent`` (erased). Calibration tends to
shift failures toward *suppressed*, which is the mechanistic story for why
calibrated low-bit quants respond to light intervention while uncalibrated ones
do not.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Probe:
    id: str
    prompt: str
    gold: str
    domain: str = "general"


@dataclass
class ProbeResult:
    probe_id: str
    domain: str
    gold: str
    rank_final: int
    rank_by_layer: list[int]
    classification: str          # "correct" | "suppressed" | "absent"


def load_probes(path: Path) -> list[Probe]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(Probe(id=d["id"], prompt=d["prompt"], gold=d["gold"],
                         domain=d.get("domain", "general")))
    return out


def _classify(rank_final: int, rank_by_layer: list[int], k: int, late_frac: float) -> str:
    if rank_final == 0:
        return "correct"
    n = len(rank_by_layer)
    late_start = int(n * (1.0 - late_frac))
    late = rank_by_layer[late_start:] or rank_by_layer[-1:]
    if any(r < k for r in late):
        return "suppressed"
    if all(r >= k for r in rank_by_layer):
        return "absent"
    return "suppressed"  # in top-k early but not late — knowledge existed, still recoverable


def run_probes(
    client,
    readout,
    probes: list[Probe],
    *,
    k: int = 40,
    late_frac: float = 0.34,
    capture_layers: list[int] | None = None,
) -> list[ProbeResult]:
    """Lens the gold token's rank across layers for each probe."""
    from quant_tuner.lens.readout import LensReadout

    results: list[ProbeResult] = []
    n_layer = int(client.props()["n_layer"])
    layers = capture_layers or list(range(n_layer))
    for probe in probes:
        tokens = client.tokenize(probe.prompt, add_special=True, parse_special=True)
        gold_ids = client.tokenize(" " + probe.gold, add_special=False, parse_special=False)
        gold_ids = gold_ids or client.tokenize(probe.gold, add_special=False, parse_special=False)
        if not tokens or not gold_ids:
            continue
        gold = int(gold_ids[0])
        fr = client.forward(tokens, capture_layers=layers, dtype="f16")
        rank_by_layer = []
        for layer in layers:
            h = fr.activations[layer][-1][None, :]
            logits = readout.lens_logits(h, layer, use_lens=True)
            rank = int(LensReadout.ranks_of(logits, np.asarray([gold], dtype=np.int64))[0, 0])
            rank_by_layer.append(rank)
        rank_final = rank_by_layer[-1]
        results.append(ProbeResult(
            probe_id=probe.id, domain=probe.domain, gold=probe.gold,
            rank_final=rank_final, rank_by_layer=rank_by_layer,
            classification=_classify(rank_final, rank_by_layer, k, late_frac),
        ))
    return results


@dataclass
class ProbeComparison:
    ref_model: str
    quant_model: str
    transitions: dict[str, int] = field(default_factory=dict)   # "correct->suppressed" -> count
    by_domain: dict[str, dict[str, int]] = field(default_factory=dict)
    regressed: list[dict] = field(default_factory=list)         # correct-on-ref, worse-on-quant

    def summary(self) -> dict[str, float]:
        total = sum(self.transitions.values()) or 1
        return {
            "probe_erased": self.transitions.get("correct->absent", 0),
            "probe_suppressed": self.transitions.get("correct->suppressed", 0),
            "probe_preserved": self.transitions.get("correct->correct", 0),
            "probe_suppressed_frac_of_lost": _suppressed_frac(self.transitions),
            "probe_n": total,
        }


def _suppressed_frac(transitions: dict[str, int]) -> float:
    lost = transitions.get("correct->absent", 0) + transitions.get("correct->suppressed", 0)
    if lost == 0:
        return 0.0
    return transitions.get("correct->suppressed", 0) / lost


def compare_probes(
    quant: list[ProbeResult],
    ref: list[ProbeResult],
) -> ProbeComparison:
    """ref-class -> quant-class transition matrix, plus per-domain + regressions."""
    ref_by_id = {r.probe_id: r for r in ref}
    transitions: dict[str, int] = {}
    by_domain: dict[str, dict[str, int]] = {}
    regressed: list[dict] = []
    for q in quant:
        r = ref_by_id.get(q.probe_id)
        if r is None:
            continue
        key = f"{r.classification}->{q.classification}"
        transitions[key] = transitions.get(key, 0) + 1
        by_domain.setdefault(q.domain, {})
        by_domain[q.domain][key] = by_domain[q.domain].get(key, 0) + 1
        if r.classification == "correct" and q.classification != "correct":
            regressed.append({
                "probe_id": q.probe_id, "domain": q.domain, "gold": q.gold,
                "ref_class": r.classification, "quant_class": q.classification,
                "quant_rank_final": q.rank_final,
                "quant_rank_by_layer": q.rank_by_layer,
            })
    return ProbeComparison(
        ref_model="ref", quant_model="quant",
        transitions=transitions, by_domain=by_domain, regressed=regressed,
    )


def render_probe_report(cmp: ProbeComparison, out_md: Path) -> Path:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    s = cmp.summary()
    lines = [
        f"# Knowledge probe comparison: {cmp.quant_model} vs {cmp.ref_model}",
        "",
        f"- probes: {s['probe_n']}",
        f"- preserved (correct->correct): {s['probe_preserved']}",
        f"- suppressed (correct->suppressed): {s['probe_suppressed']}",
        f"- erased (correct->absent): {s['probe_erased']}",
        f"- of the lost, fraction merely suppressed: {s['probe_suppressed_frac_of_lost']:.2f}",
        "",
        "## Transition matrix",
        "",
        "| ref → quant | count |",
        "| --- | --- |",
    ]
    for key, count in sorted(cmp.transitions.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {key} | {count} |")
    lines.append("")
    lines.append("## Most-regressed probes")
    lines.append("")
    lines.append("| probe | domain | gold | ref | quant | final rank |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(cmp.regressed, key=lambda x: -x["quant_rank_final"])[:25]:
        lines.append(
            f"| {r['probe_id']} | {r['domain']} | {r['gold']!r} | "
            f"{r['ref_class']} | {r['quant_class']} | {r['quant_rank_final']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    return out_md


def write_probe_results(results: list[ProbeResult], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

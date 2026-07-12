"""Tool-call representation inspection: replay decision points through the lens.

quant-tuner calibrates on tool-use logs, so the question this module answers is
*where in the network does a tool-call decision form, and does quantization
preserve it?* It walks the same assistant-turn structure as
:func:`quant_tuner.eval.toolcall.eval_per_turn` (parity is unit-tested), and at
each decision point captures the residual stream over the decision tail, then
reports:

- the lens rank of the gold tool-call opener and gold tool-name tokens at the
  final position (how confidently the model is about to call the right tool);
- the *emergence layer*: the first layer at which the gold token enters the
  lens top-k (calibration tends to preserve the late-layer emergence while
  uncalibrated low-bit quants push it later / never);
- divergence vs a reference (FP16) capture at the decision positions.

Reference runs are ordinary capture runs tagged ``role=reference`` produced
once from the FP16 GGUF over the same holdout; token sequences match by
construction (same template engine).
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from quant_tuner.eval.toolcall import (
    DEFAULT_SYSTEM_PROMPT,
    _annotate_failing_calls,
    maybe_inject_system,
    prior_tool_result,
    strip_for_api,
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionPoint:
    """One assistant turn whose formation we want to lens."""

    session_id: str | None
    turn: int
    msg_index: int
    kind: str                      # "tool_call" | "post_result"
    prefix: list[dict]             # strip_for_api(msgs[:i]) — exact eval-parity prefix
    tools: list[dict]
    gold_tool: str | None          # first ground-truth tool name (tool_call turns)
    prior_was_error: bool


def iter_decision_points(
    session: dict,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = 8,
) -> Iterator[DecisionPoint]:
    """Yield decision points, matching ``eval_per_turn``'s turn walk exactly.

    The prefix and turn-counting logic mirror
    :func:`quant_tuner.eval.toolcall.eval_per_turn` so a lens replay lands on
    the same model inputs the scoring harness uses.
    """
    msgs = maybe_inject_system(session["messages"], system_prompt)
    _annotate_failing_calls(msgs)
    tools = session.get("tools", [])
    sid = session.get("session_id")

    turn_idx = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        has_tool_calls = bool(m.get("tool_calls"))
        prior = prior_tool_result(msgs, i)
        if not has_tool_calls and prior is None:
            continue
        if turn_idx >= max_turns:
            break
        turn_idx += 1

        gold_tool = None
        if has_tool_calls:
            tcs = m.get("tool_calls") or []
            if tcs:
                fn = tcs[0].get("function") or {}
                gold_tool = fn.get("name") or tcs[0].get("name")
        yield DecisionPoint(
            session_id=sid,
            turn=turn_idx,
            msg_index=i,
            kind="tool_call" if has_tool_calls else "post_result",
            prefix=strip_for_api(msgs[:i]),
            tools=tools,
            gold_tool=gold_tool,
            prior_was_error=bool(prior is not None and prior.get("_failing_call")),
        )


def render_prefix_tokens(client, dp: DecisionPoint) -> list[int]:
    """Template + tokenize a decision prefix into the exact model input tokens.

    Uses the server's chat-template engine (``add_assistant=True`` opens the
    assistant turn) and special-token-aware tokenization — the same path the
    OpenAI endpoints use, so the tokens are what the model would actually see.
    """
    prompt = client.apply_template(dp.prefix, add_assistant=True)
    return client.tokenize(prompt, add_special=True, parse_special=True)


def gold_token_ids(client, dp: DecisionPoint) -> dict[str, list[int]]:
    """Token ids for the gold tool-call opener and tool name, if determinable.

    The opener is inferred from the template by diffing an empty assistant turn
    against one carrying a tool call; if that can't be determined we fall back
    to the first tokens of the tool name rendered in context.
    """
    out: dict[str, list[int]] = {}
    if dp.gold_tool:
        # tool name as it appears mid-text (leading space matches most BPE)
        out["tool_name"] = client.tokenize(
            " " + dp.gold_tool, add_special=False, parse_special=False
        )
        bare = client.tokenize(dp.gold_tool, add_special=False, parse_special=False)
        if bare:
            out["tool_name_bare"] = bare
    return out


@dataclass
class DecisionResult:
    session_id: str | None
    turn: int
    kind: str
    gold_tool: str | None
    prior_was_error: bool
    n_prefix_tokens: int
    gold_tool_rank_final: int | None       # lens rank of gold tool name at final pos
    emergence_layer: int | None            # first layer gold enters lens top-k
    decision_kld_vs_ref: float | None      # final KLD vs reference at decision pos
    layer_divergence_auc: float | None     # mean rel-l2 layer profile vs reference


@dataclass
class ToolcallLensSummary:
    model: str                             # basename(quant_path) — leaderboard join key
    n_decisions: int
    gold_tool_rank_final_mean: float
    emergence_layer_mean: float
    decision_kld_vs_ref_mean: float | None
    layer_divergence_auc_mean: float | None
    n_error_recovery: int
    per_decision: list[dict] = field(default_factory=list)

    def sidecar_row(self) -> dict:
        return {
            "model": self.model,
            "lens_gold_rank": round(self.gold_tool_rank_final_mean, 3),
            "lens_emerge_layer": round(self.emergence_layer_mean, 2),
            "lens_decision_kld": (
                round(self.decision_kld_vs_ref_mean, 5)
                if self.decision_kld_vs_ref_mean is not None else ""
            ),
            "lens_n_decisions": self.n_decisions,
        }


def _emergence_layer(grid, layers: list[int], gold_ids: list[int], pos: int) -> int | None:
    """First layer (ascending) at which any gold id is in that layer's top-k."""
    if not gold_ids:
        return None
    gold = set(int(g) for g in gold_ids)
    pi = -1  # decision position is the final kept position
    for li, layer in enumerate(layers):
        top = grid.ids[li, pi]
        if gold & set(int(t) for t in top):
            return layer
    return None


def _gold_rank_final(readout, run, layer: int, gold_ids: list[int]) -> int | None:
    """Best (lowest) lens rank of any gold id at the final position, final layer."""
    if not gold_ids:
        return None
    from quant_tuner.lens.readout import LensReadout

    h = run.activations(layer)[-1][None, :]
    logits = readout.lens_logits(h, layer, use_lens=True)
    ranks = LensReadout.ranks_of(logits, np.asarray(gold_ids, dtype=np.int64))
    return int(ranks.min())


def replay_toolcall_lens(
    holdout: Path,
    model_gguf: Path,
    lens_path: Path,
    *,
    runs_dir: Path,
    reference_runs_dir: Path | None = None,
    tail_positions: int = 16,
    max_sessions: int | None = None,
    max_turns: int = 8,
    ctx: int = 8192,
    ngl: int = 99,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    label: str | None = None,
) -> ToolcallLensSummary:
    """Replay a tool-call holdout through the lens; summarize per model.

    ``reference_runs_dir`` (if given) should hold FP16 capture runs keyed by
    ``run_id`` so decision KLD / layer divergence can be computed; runs are
    matched by identical decision-tail token sequence.
    """
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.capture import capture_run, load_run
    from quant_tuner.lens.diff import diff_runs
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    holdout = Path(holdout)
    sessions = [json.loads(line) for line in holdout.read_text().splitlines() if line.strip()]
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    lens = JacobianLensGGUF.load(str(lens_path))
    weights = ReadoutWeights.from_gguf(str(model_gguf))
    readout = LensReadout(weights, lens)
    model_label = label or Path(model_gguf).name

    ref_readout = None
    if reference_runs_dir is not None:
        # reference readout uses the same lens but the FP16 model's own head
        ref_runs = {m.run_id: m for m in _index_runs(reference_runs_dir)}
    else:
        ref_runs = {}

    results: list[DecisionResult] = []
    with running_jlens_server(model_gguf, ctx=ctx, ngl=ngl) as client:
        for session in sessions:
            for dp in iter_decision_points(session, system_prompt=system_prompt, max_turns=max_turns):
                tokens = render_prefix_tokens(client, dp)
                if len(tokens) < 2:
                    continue
                keep = list(range(-min(tail_positions, len(tokens)), 0))
                run_dir = capture_run(
                    client, tokens, runs_dir=runs_dir,
                    keep_positions=keep, logits_positions=[-1],
                    tags={"model": model_label, "session": dp.session_id or "",
                          "turn": dp.turn, "role": "candidate"},
                )
                run = load_run(run_dir)
                grid = run.readout_topk(readout, k=8)
                golds = gold_token_ids(client, dp)
                gold_ids = golds.get("tool_name", []) + golds.get("tool_name_bare", [])

                final_layer = run.manifest.capture_layers[-1]
                rank_final = _gold_rank_final(readout, run, final_layer, gold_ids)
                emerge = _emergence_layer(grid, run.manifest.capture_layers, gold_ids, -1)

                decision_kld = None
                div_auc = None
                ref_dir = _match_reference(ref_runs, reference_runs_dir, run)
                if ref_dir is not None:
                    ref_run = load_run(ref_dir)
                    if ref_readout is None:
                        ref_weights = ReadoutWeights.from_gguf(ref_run.manifest.model_path)
                        ref_readout = LensReadout(ref_weights, lens)
                    try:
                        d = diff_runs(run, ref_run, readout_run=readout, readout_ref=ref_readout)
                        decision_kld = float(d.final_kld[-1])
                        div_auc = float(d.layer_rel_l2.mean())
                    except ValueError as e:
                        logger.debug("no diff for %s: %s", run.manifest.run_id, e)

                results.append(DecisionResult(
                    session_id=dp.session_id, turn=dp.turn, kind=dp.kind,
                    gold_tool=dp.gold_tool, prior_was_error=dp.prior_was_error,
                    n_prefix_tokens=len(tokens),
                    gold_tool_rank_final=rank_final,
                    emergence_layer=emerge,
                    decision_kld_vs_ref=decision_kld,
                    layer_divergence_auc=div_auc,
                ))

    return _summarize(model_label, results)


def _index_runs(runs_dir: Path):
    from quant_tuner.lens.capture import find_runs

    return find_runs(runs_dir)


def _match_reference(ref_index, reference_runs_dir, run):
    """Find a reference run over the same decision-tail tokens as ``run``."""
    if not reference_runs_dir:
        return None

    for run_id, manifest in ref_index.items():
        if manifest.tokens == run.manifest.tokens and manifest.positions == run.manifest.positions:
            return Path(reference_runs_dir) / run_id
    return None


def _summarize(model: str, results: list[DecisionResult]) -> ToolcallLensSummary:
    ranks = [r.gold_tool_rank_final for r in results if r.gold_tool_rank_final is not None]
    emerges = [r.emergence_layer for r in results if r.emergence_layer is not None]
    klds = [r.decision_kld_vs_ref for r in results if r.decision_kld_vs_ref is not None]
    aucs = [r.layer_divergence_auc for r in results if r.layer_divergence_auc is not None]
    return ToolcallLensSummary(
        model=model,
        n_decisions=len(results),
        gold_tool_rank_final_mean=float(np.mean(ranks)) if ranks else float("nan"),
        emergence_layer_mean=float(np.mean(emerges)) if emerges else float("nan"),
        decision_kld_vs_ref_mean=float(np.mean(klds)) if klds else None,
        layer_divergence_auc_mean=float(np.mean(aucs)) if aucs else None,
        n_error_recovery=sum(1 for r in results if r.prior_was_error),
        per_decision=[asdict(r) for r in results],
    )


def write_lens_sidecar(summaries: list[ToolcallLensSummary], out_csv: Path) -> None:
    """Write a leaderboard sidecar CSV keyed by model basename."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.sidecar_row() for s in summaries]
    fields = ["model", "lens_gold_rank", "lens_emerge_layer", "lens_decision_kld", "lens_n_decisions"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_decisions_jsonl(summary: ToolcallLensSummary, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in summary.per_decision:
            f.write(json.dumps(rec) + "\n")

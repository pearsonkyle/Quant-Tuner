"""Recipe-driven end-to-end pipeline: extract → calibrate → quantize → bench.

Reads a :class:`~quant_tuner.config.RunConfig` (typically from a YAML recipe)
and produces a quantized GGUF + a row in ``workspace/results.csv``. Each
phase is idempotent — re-running on a populated workspace skips finished work
and just rebuilds anything missing.

The CLI commands ``quant-tuner run --recipe …`` and
``quant-tuner bench --recipe …`` are thin wrappers around the functions here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import awq, gptq, imatrix
from quant_tuner.config import RunConfig
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp, mtp
from quant_tuner.paths import Workspace
from quant_tuner.quantize import convert, gguf

# llama-imatrix context length (all three calibration methods). 4096 matches
# the packer's per_session_cap=3500 windows — at the old default of 512 every
# window straddled ~7 imatrix chunks and the repeated system/schema text was
# re-sliced across most of them. Override per-recipe via
# calibration.params.imatrix_ctx.
DEFAULT_IMATRIX_CTX = 4096

# ---------------------------------------------------------------------------
# Phase 1: workspace + model extraction + F16 conversion
# ---------------------------------------------------------------------------


def prepare_workspace(cfg: RunConfig) -> Workspace:
    """Materialize the workspace subdirectories declared by :class:`Workspace`."""
    ws = Workspace(root=cfg.workspace)
    ws.ensure()
    (ws.root / "logs").mkdir(exist_ok=True)
    return ws


def extract_and_convert(cfg: RunConfig, ws: Workspace) -> Path:
    """Download/extract the HF model and convert it to an F16 GGUF.

    Returns the F16 GGUF path. The intermediate HF checkpoint is left under
    ``ws.model_extracted`` for AWQ/GPTQ passes that need it.
    """
    logs = ws.root / "logs"
    f16 = ws.gguf_dir / "model-f16.gguf"

    step("extract HF model", ws.model_extracted / "config.json",
         lambda: extract.extract_text_lm(
             source=cfg.model, output_dir=ws.model_extracted,
             keep_mtp=cfg.extract.keep_mtp))

    step("convert HF -> F16 GGUF", f16,
         lambda: convert.hf_to_f16_gguf(ws.model_extracted, f16, log=logs / "convert.log"))

    return f16


# ---------------------------------------------------------------------------
# Phase 2: corpus prep
# ---------------------------------------------------------------------------


def prepare_corpora(cfg: RunConfig, ws: Workspace) -> tuple[Path, Path]:
    """Build calibration and eval corpora from the configured log file.

    Returns ``(train_corpus, eval_corpus)``. If ``cfg.data.corpus`` is set, the
    file is reused directly as the train corpus and only the eval corpus is
    built; otherwise both are produced from ``cfg.data.logs`` via the
    stratified packing in :mod:`quant_tuner.data.split`.
    """
    train = ws.corpus_dir / "train.txt"
    eval_ = ws.corpus_dir / "eval.txt"

    if cfg.data.corpus is not None and not train.exists():
        # Pre-built corpus path: copy the train side; eval is still derived from logs.
        train.write_text(Path(cfg.data.corpus).read_text())

    if cfg.bench.eval_corpus is not None and not eval_.exists():
        # Pre-built eval corpus (e.g. build_corpora.py's external
        # corpus.eval.txt). Preferred for PPL/KLD: llama-perplexity has no
        # --parse-special, so chat-templated eval text tokenizes control
        # markers as plain BPE — a distribution the model never sees.
        eval_.write_text(Path(cfg.bench.eval_corpus).read_text())

    if not (train.exists() and eval_.exists()):
        if cfg.data.logs is None:
            raise ValueError(
                "data.logs is required to build corpora (or set data.corpus to a "
                "pre-built train corpus AND provide an eval set)."
            )
        with phase("build corpora from logs"):
            _build_corpora(cfg, ws, train, eval_)

    return train, eval_


def _build_corpora(cfg: RunConfig, ws: Workspace, train: Path, eval_: Path) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ws.model_extracted, fix_mistral_regex=True)
    sessions = ingest.load_sessions(cfg.data.logs)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    log(f"  {len(sessions)} sessions after filtering")

    splits = split.split_sessions(
        sessions,
        train_frac=cfg.data.train_frac,
        test_frac=cfg.data.test_frac,
        holdout_frac=cfg.data.holdout_frac,
        seed=42,
    )
    log(f"  split sizes: train={len(splits['train'])} "
        f"test={len(splits['test'])} holdout={len(splits['holdout'])}")

    if not train.exists():
        chunks, _kept, total, audit = split.stratified_pack(
            splits["train"], tok,
            target_tokens=cfg.data.train_max_tokens,
            per_session_cap=cfg.data.per_session_cap, seed=42,
            system_prose_budget=cfg.data.system_prose_budget,
            full_prose_quota=cfg.data.full_prose_quota,
            max_windows_per_session=cfg.data.max_windows_per_session,
            tool_schema_quota=cfg.data.tool_schema_quota,
        )
        split.write_corpus(chunks, train, supplement=cfg.data.supplement)
        (ws.corpus_dir / "train_audit.json").write_text(
            json.dumps(audit, indent=2, default=str))
        suffix = f" (+ supplement {cfg.data.supplement.name})" if cfg.data.supplement else ""
        log(f"  train: {total:,} tokens -> {train.name}{suffix}")
        w = audit.get("windowing")
        if w:
            log(f"         {w['windows_emitted']} windows / {w['sessions_kept']} sessions, "
                f"tool-turn token share {w['tool_turn_token_share']:.0%}")

    if not eval_.exists():
        ev_chunks, _ek, ev_total, ev_audit = split.stratified_pack(
            splits["test"], tok,
            target_tokens=cfg.data.eval_max_tokens,
            per_session_cap=4_000, seed=43,
        )
        split.write_corpus(ev_chunks, eval_)
        (ws.corpus_dir / "eval_audit.json").write_text(
            json.dumps(ev_audit, indent=2, default=str))
        log(f"  eval:  {ev_total:,} tokens -> {eval_.name}")

    # Persist the holdout slice for downstream tool-call eval (eval/holdout.py).
    holdout_path = ws.corpus_dir / "holdout.jsonl"
    if not holdout_path.exists():
        split.write_split_jsonl(splits["holdout"], holdout_path)


# ---------------------------------------------------------------------------
# Phase 3: calibration (method-dependent)
# ---------------------------------------------------------------------------


def calibrate(
    cfg: RunConfig, ws: Workspace, f16: Path, train_corpus: Path, eval_corpus: Path,
) -> dict[str, Path]:
    """Run the configured calibration method and return artifacts produced.

    The dict keys depend on the method:
      * ``none`` — empty (no calibration)
      * ``imatrix`` — ``{"imatrix": <path>, "f16": <unchanged f16>}``
      * ``awq`` — ``{"imatrix": <imatrix collected on the folded F16, optionally
        re-weighted via params.imatrix_variant>, "f16": <f16 with scales folded>}``
      * ``gptq`` — ``{"imatrix": <imatrix collected on the GPTQ-rounded F16,
        optionally re-weighted via params.imatrix_variant>, "f16": <new f16
        with rounded weights>}``
    """
    method = cfg.calibration.method
    logs = ws.root / "logs"

    if method == "none":
        return {}

    if method == "imatrix":
        return _calibrate_imatrix(cfg, ws, f16, train_corpus, logs)

    if method == "awq":
        # AWQ computes its imatrix AFTER the fold (see _calibrate_awq).
        return _calibrate_awq(cfg, ws, f16, train_corpus, logs)

    if method == "gptq":
        # GPTQ computes its imatrix AFTER the rounding (see _calibrate_gptq).
        return _calibrate_gptq(cfg, ws, f16, train_corpus, eval_corpus, logs)

    raise ValueError(f"unknown calibration method: {method!r}")


def _calibrate_imatrix(
    cfg: RunConfig, ws: Workspace, f16: Path, train_corpus: Path, logs: Path,
) -> dict[str, Path]:
    base_imatrix = ws.calibration_dir / "imatrix-base.gguf"
    tuned = ws.calibration_dir / f"imatrix-{cfg.calibration.variant}.gguf"

    params: dict[str, Any] = dict(cfg.calibration.params)
    imatrix_ctx = int(params.pop("imatrix_ctx", DEFAULT_IMATRIX_CTX))

    step("llama-imatrix (base)", base_imatrix,
         lambda: llama_cpp.imatrix(f16, train_corpus, base_imatrix,
                                   ctx=imatrix_ctx, log=logs / "imatrix-base.log"))

    variant = cfg.calibration.variant
    if variant == "default":
        # Use the base imatrix as-is.
        return {"imatrix": base_imatrix, "f16": f16}

    step(f"build imatrix variant '{variant}'", tuned,
         lambda: imatrix.calibrate(
             variant=variant, f16_gguf=f16,
             base_imatrix=base_imatrix, out_path=tuned,
             **params))
    return {"imatrix": tuned, "f16": f16}


# Recipe params consumed by awq.apply() rather than awq.calibrate(). Routed
# separately so recipes can configure the fold (e.g. rmsnorm_plus_one for
# non-Qwen3.5 norms) without crashing the calibrate call.
_AWQ_APPLY_PARAMS = ("rmsnorm_plus_one", "sanity_max_rel", "sanity_tokens")


def _calibrate_awq(
    cfg: RunConfig, ws: Workspace, f16: Path, train_corpus: Path, logs: Path,
) -> dict[str, Path]:
    awq_bundle = ws.calibration_dir / "awq.pt"
    model_awq = ws.root / "model_awq"
    f16_awq = ws.gguf_dir / "model-f16-awq.gguf"

    params: dict[str, Any] = dict(cfg.calibration.params)
    if cfg.calibration.variant == "a050":
        params.setdefault("force_alpha", 0.5)
    # Match the α-search proxy to the target quant's error shape (2-bit ->
    # q2k_b16, 3-bit -> q3k_b16, IQ2_* -> codebook proxies, else int4_g128).
    # Low-bit ftypes (IQ1/IQ2/IQ3/Q2_K/Q3_K) are tensor *mixes* — llama-quantize
    # bumps attn_v/attn_output/ffn_down to higher types per ftype (see
    # awq.proxy_for_member) — so additionally score each member against its
    # real target type. For pure ftypes in those families (Q3_K_S, IQ3_S) the
    # mix resolves to zero overrides. Pinning `params.proxy` in a recipe opts
    # out of both; set `params.proxy_mix` explicitly to combine a pinned base
    # proxy with the mix.
    if "proxy" not in params:
        params["proxy"] = awq.proxy_for_quant_type(cfg.quantize.type)
        if cfg.quantize.type.upper().startswith(("IQ1", "IQ2", "IQ3", "Q2", "Q3")):
            params.setdefault("proxy_mix", cfg.quantize.type)

    apply_params: dict[str, Any] = {
        k: params.pop(k) for k in _AWQ_APPLY_PARAMS if k in params
    }
    for shared in ("device", "dtype"):
        if shared in params:
            apply_params[shared] = params[shared]
    # Optional second-stage imatrix re-weighting on the folded model
    # (e.g. "hybrid_custom" to stack the two winning calibrations).
    imatrix_variant = params.pop("imatrix_variant", "default")
    imatrix_ctx = int(params.pop("imatrix_ctx", DEFAULT_IMATRIX_CTX))

    step("AWQ calibrate (capture mean|x| + grid α)", awq_bundle,
         lambda: awq.calibrate(ws.model_extracted, train_corpus, awq_bundle, **params))

    step("AWQ apply (fold scales into RMSNorm)", model_awq / "config.json",
         lambda: awq.apply(ws.model_extracted, awq_bundle, model_awq, **apply_params))

    step("convert AWQ-folded HF -> F16 GGUF", f16_awq,
         lambda: convert.hf_to_f16_gguf(model_awq, f16_awq, log=logs / "convert-awq.log"))

    # The imatrix MUST be collected on the folded F16: AWQ rescales each input
    # channel, so E[a^2] from the unfolded model would over-weight exactly the
    # channels AWQ already boosted and under-protect the rest.
    return _imatrix_with_variant(
        ws, "awq", f16_awq, model_awq, train_corpus, logs,
        imatrix_variant=imatrix_variant, imatrix_ctx=imatrix_ctx,
        collect_label="llama-imatrix (on AWQ-folded F16)")


def _imatrix_with_variant(
    ws: Workspace, method: str, f16_src: Path, model_dir: Path,
    train_corpus: Path, logs: Path, *,
    imatrix_variant: str, imatrix_ctx: int, collect_label: str,
) -> dict[str, Path]:
    """Collect ``imatrix-{method}.gguf`` on ``f16_src`` and, when
    ``imatrix_variant`` is not ``"default"``, re-weight it into
    ``imatrix-{method}-{variant}.gguf`` — the shared tail of the AWQ (folded
    F16) and GPTQ (rounded F16) calibration branches."""
    base = ws.calibration_dir / f"imatrix-{method}.gguf"
    step(collect_label, base,
         lambda: llama_cpp.imatrix(f16_src, train_corpus, base,
                                   ctx=imatrix_ctx, log=logs / f"imatrix-{method}.log"))

    if imatrix_variant == "default":
        return {"imatrix": base, "f16": f16_src}

    tuned = ws.calibration_dir / f"imatrix-{method}-{imatrix_variant}.gguf"
    step(f"build imatrix variant '{imatrix_variant}' ({method})", tuned,
         lambda: imatrix.calibrate(
             # recipes pass free-form strings; imatrix.calibrate validates
             variant=cast(imatrix.Variant, imatrix_variant), f16_gguf=f16_src,
             base_imatrix=base, out_path=tuned,
             model_dir=model_dir, calibration_text=train_corpus,
             forward_stats_path=ws.calibration_dir / f"forward-stats-{method}.npz"))
    return {"imatrix": tuned, "f16": f16_src}


# Recipe params consumed by gptq.apply() rather than gptq.calibrate().
_GPTQ_APPLY_PARAMS = (
    "n_bits", "group_size", "dampen", "actorder", "sym", "grid_mix",
    "name_filter", "sanity_tokens", "sanity_max_rel",
)

# Bits-aware guardrail defaults: at 2-3 bits, substantially larger PPL/logit
# drift is expected and legitimate; the 4-bit thresholds would abort runs
# that are doing exactly what was asked.
_GPTQ_PPL_MAX_RATIO = {2: 4.0, 3: 2.0}
_GPTQ_SANITY_MAX_REL = {2: 1.0, 3: 0.75}


def _calibrate_gptq(
    cfg: RunConfig, ws: Workspace, f16: Path, train_corpus: Path,
    eval_corpus: Path, logs: Path,
) -> dict[str, Path]:
    hessians = ws.calibration_dir / "hessians"
    model_gptq = ws.root / "model_gptq"
    f16_gptq = ws.gguf_dir / "model-f16-gptq.gguf"

    params: dict[str, Any] = dict(cfg.calibration.params)
    # Optional second-stage imatrix re-weighting on the rounded model
    # (e.g. "hybrid_custom" to stack the two winning calibrations).
    imatrix_variant = params.pop("imatrix_variant", "default")
    imatrix_ctx = int(params.pop("imatrix_ctx", DEFAULT_IMATRIX_CTX))
    apply_params: dict[str, Any] = {
        k: params.pop(k) for k in _GPTQ_APPLY_PARAMS if k in params
    }
    if "dtype" in params:
        apply_params.setdefault("dtype", params["dtype"])
    # Default the rounding grid to the target quant's shape (2-bit -> asym
    # g16, 3-bit -> asym g16, else sym g32). Recipe params take precedence.
    # Mixed ftypes bump some tensors above the base grid (attn_v -> Q4_K at
    # 2 bits under GQA >= 4, Q4_K_M attn_v/ffn_down -> Q6_K, ...; see
    # _quant_mix.target_type_for_member) — default grid_mix so each tensor is
    # rounded on the grid matching its real target type. Pure ftypes resolve
    # to zero overrides, so this is a no-op for them. Pinning any base-grid
    # param in a recipe opts out of the mix; set `grid_mix` explicitly to
    # combine a pinned base grid with the mix (`grid_mix: null` disables it).
    user_pinned_grid = any(k in apply_params for k in ("n_bits", "group_size", "sym"))
    for k, v in gptq.grid_for_quant_type(cfg.quantize.type).items():
        apply_params.setdefault(k, v)
    if not user_pinned_grid:
        apply_params.setdefault("grid_mix", cfg.quantize.type)
    n_bits = int(apply_params["n_bits"])
    apply_params.setdefault("sanity_max_rel", _GPTQ_SANITY_MAX_REL.get(n_bits, 0.5))
    ppl_max_ratio = params.pop(
        "ppl_max_ratio", _GPTQ_PPL_MAX_RATIO.get(n_bits, 1.5))

    step("GPTQ calibrate (Hessians)", hessians / "_done",
         lambda: (gptq.calibrate(ws.model_extracted, train_corpus, hessians, **params)
                  or (hessians / "_done").touch()))

    step("GPTQ apply (round + error-compensate)", model_gptq / "config.json",
         lambda: gptq.apply(ws.model_extracted, hessians, model_gptq, **apply_params))

    step("convert GPTQ-rounded HF -> F16 GGUF", f16_gptq,
         lambda: convert.hf_to_f16_gguf(model_gptq, f16_gptq, log=logs / "convert-gptq.log"))

    # Sanity-check: refuse to proceed if rounded F16 is too far from the reference.
    baseline_ppl_file = ws.eval_dir / "baseline-ppl.txt"
    if baseline_ppl_file.exists():
        reference_ppl = float(baseline_ppl_file.read_text())
    else:
        reference_ppl = gptq.verify_perplexity(
            f16, eval_corpus, ctx=cfg.bench.eval_ctx, log=logs / "baseline-ppl.log",
        )
        baseline_ppl_file.write_text(str(reference_ppl))
    gptq.verify_perplexity(
        f16_gptq, eval_corpus,
        reference_ppl=reference_ppl, max_ratio=ppl_max_ratio,
        ctx=cfg.bench.eval_ctx, log=logs / "gptq-verify-ppl.log",
    )

    # The imatrix is collected on the ROUNDED F16 (after the guardrail — no
    # point spending an imatrix run on a model about to be rejected): the
    # weight-aware variants score `‖W[:,c]‖²·E[a²]` and llama-quantize sees
    # the rounded weights, so column norms of the original F16 would be
    # norms of weights that no longer exist. E[a²] drift from rounding is
    # second-order (GPTQ does not rescale channels the way AWQ folding does).
    return _imatrix_with_variant(
        ws, "gptq", f16_gptq, model_gptq, train_corpus, logs,
        imatrix_variant=imatrix_variant, imatrix_ctx=imatrix_ctx,
        collect_label="llama-imatrix (on GPTQ-rounded F16)")


# ---------------------------------------------------------------------------
# Phase 4: quantize
# ---------------------------------------------------------------------------


def quantize_model(
    cfg: RunConfig, ws: Workspace, f16: Path, artifacts: dict[str, Path],
) -> Path:
    """Produce the final quantized GGUF using the calibration artifacts."""
    src_f16 = artifacts.get("f16", f16)
    imatrix_path = artifacts.get("imatrix")
    if imatrix_path is None and cfg.quantize.type.upper().startswith(("IQ1", "IQ2")):
        raise ValueError(
            f"quantize.type={cfg.quantize.type} requires an importance matrix — "
            f"llama-quantize refuses IQ1/IQ2 without one. Use calibration.method "
            f"imatrix/awq/gptq instead of {cfg.calibration.method!r}."
        )
    suffix = "" if imatrix_path is None else f"-{cfg.calibration.method}"
    # Variant must be part of the filename: otherwise re-running the same
    # workspace with a different variant finds the old GGUF, skips
    # llama-quantize, and benches the stale file under the new label. The
    # same applies to params.imatrix_variant (the AWQ/GPTQ second-stage
    # imatrix re-weighting).
    if imatrix_path is not None and cfg.calibration.variant != "default":
        suffix += f"-{cfg.calibration.variant}"
    imatrix_variant = cfg.calibration.params.get("imatrix_variant", "default")
    if imatrix_path is not None and imatrix_variant != "default":
        suffix += f"-{imatrix_variant}"
    tensor_types = dict(cfg.quantize.tensor_types or {})
    if cfg.extract.keep_mtp and cfg.quantize.mtp_pin:
        # Resolve the draft layer from the GGUF itself; a pattern that matches nothing
        # would silently quantize the head with the trunk.
        info = mtp.describe(src_f16, cfg.quantize.mtp_pin)
        if info["has_mtp"]:
            tensor_types.update(info["pin"])
            log(f"  MTP draft layer(s) {info['mtp_layer_indices']} pinned "
                f"{cfg.quantize.mtp_pin}")
        else:
            log("  extract.keep_mtp is set but the F16 GGUF has no draft layer — "
                "nothing to pin (check the conversion log)")
        suffix += "-mtp"
    out_quant = ws.gguf_dir / f"{cfg.quantize.type}{suffix}.gguf"
    logs = ws.root / "logs"

    step(f"llama-quantize {cfg.quantize.type}", out_quant,
         lambda: gguf.quantize(src_f16, out_quant, cfg.quantize.type,
                               imatrix=imatrix_path,
                               tensor_types=tensor_types or None,
                               log=logs / f"quantize-{cfg.quantize.type}.log"))
    return out_quant


# ---------------------------------------------------------------------------
# Phase 5: bench
# ---------------------------------------------------------------------------


def bench(
    cfg: RunConfig, ws: Workspace, quant: Path, f16: Path, eval_corpus: Path,
) -> runner.BenchRow:
    """Bench the quantized model and append a row to ``workspace/results.csv``."""
    logs = ws.root / "logs"
    eval_baseline = ws.eval_dir / "baseline.kld"
    reference = cfg.bench.reference or f16

    step("build KLD baseline (vs reference)", eval_baseline,
         lambda: kld.build_baseline(reference, eval_corpus, eval_baseline,
                                    ctx=cfg.bench.eval_ctx, log=logs / "baseline.log"))

    # Optional in-distribution tools suite: separate corpus, separate baseline.
    tools_corpus: Path | None = None
    tools_baseline: Path | None = None
    if cfg.bench.eval_tools_corpus is not None:
        tools_corpus = Path(cfg.bench.eval_tools_corpus)
        tools_baseline = ws.eval_dir / "baseline-tools.kld"
        step("build KLD baseline (tools eval)", tools_baseline,
             lambda: kld.build_baseline(reference, tools_corpus, tools_baseline,
                                        ctx=cfg.bench.eval_ctx,
                                        log=logs / "baseline-tools.log"))

    n_params = bpw_mod.n_params(reference)
    label = f"{cfg.calibration.method}/{cfg.quantize.type}-{cfg.calibration.variant}"
    with phase(f"bench {label}"):
        row = runner.bench_one(
            quant, label,
            reference_n_params=n_params,
            eval_dataset=eval_corpus,
            eval_baseline=eval_baseline,
            tools_dataset=tools_corpus,
            tools_baseline=tools_baseline,
            eval_ctx=cfg.bench.eval_ctx,
            log_dir=logs,
            suite=cfg.bench.suite,
        )
    runner.append_row(ws.results_csv, row)
    return row


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run_pipeline(cfg: RunConfig) -> runner.BenchRow:
    """Execute the full recipe: extract → calibrate → quantize → bench."""
    ws = prepare_workspace(cfg)
    log(f"workspace: {ws.root}")
    log(f"recipe:    {cfg.name}  (method={cfg.calibration.method}/{cfg.calibration.variant} "
        f"quant={cfg.quantize.type})")

    f16 = extract_and_convert(cfg, ws)
    train_corpus, eval_corpus = prepare_corpora(cfg, ws)
    artifacts = calibrate(cfg, ws, f16, train_corpus, eval_corpus)
    quant = quantize_model(cfg, ws, f16, artifacts)
    row = bench(cfg, ws, quant, f16, eval_corpus)

    log("=" * 60)
    log(f"DONE  bpw={row.bpw:.3f}  mean_kld={row.mean_kld}  "
        f"same_top_p={row.same_top_p}  decode={row.decode_tok_s}")
    return row

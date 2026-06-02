"""Backfill the DB from existing CSV files (one-time historical migration).

The legacy ``results.csv`` / ``kld_results.csv`` cram the quant identity into a
pipe-delimited ``model`` label whose field order varies per experiment. The
``label_format`` arg tells the importer how to split it. Reps-aggregated CSVs key
on the GGUF basename, which we match against ``Quant.quant_path``'s basename.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sqlmodel import Session, select

from quant_tuner.db.io import get_or_create_dataset, upsert_quant
from quant_tuner.db.models import MMLU_SUBJECTS, MmluRep, Quant, ToolcallRep


def _f(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _i(s: str | None) -> int | None:
    v = _f(s)
    return int(v) if v is not None else None


def _parse_label(label: str, label_format: str) -> dict[str, str | None]:
    """Split a pipe ``model`` label into identity fields.

    Supported formats:
      - ``technique_dataset``: ``repo|technique|dataset`` (exp-001)
      - ``dataset_quant``:     ``repo|dataset|quant_type`` (exp-001-ext / -27b)
    """
    parts = label.split("|")
    if label_format == "technique_dataset":
        repo = parts[0]
        technique = parts[1] if len(parts) > 1 else "none"
        dataset = parts[2] if len(parts) > 2 else None
        return {"model_repo": repo, "technique": technique, "dataset": dataset,
                "quant_type": None}
    if label_format == "dataset_quant":
        repo = parts[0]
        dataset = parts[1] if len(parts) > 1 else None
        quant_type = parts[2] if len(parts) > 2 else None
        # F16 rows carry technique "none"; quantized rows get their technique from
        # whether a dataset was used (caller can override post-import if needed).
        technique = "none" if (quant_type or "").upper() in {"F16", "FP16"} else "imatrix"
        return {"model_repo": repo, "technique": technique, "dataset": dataset,
                "quant_type": quant_type}
    raise ValueError(f"unknown label_format {label_format!r}")


def import_results_csv(
    session: Session,
    path: Path,
    *,
    label_format: str = "dataset_quant",
) -> int:
    """Import a bench ``results.csv`` / ``kld_results.csv``. Returns row count."""
    n = 0
    with path.open() as f:
        for r in csv.DictReader(f):
            ident = _parse_label(r["model"], label_format)
            quant_path = r.get("quant_path") or None
            quant_type = ident["quant_type"]
            if quant_type is None and quant_path:
                # fall back to the GGUF stem prefix, e.g. Q6_K-combined.gguf
                quant_type = Path(quant_path).stem.split("-")[0]
            dataset = get_or_create_dataset(session, ident["dataset"])
            upsert_quant(
                session,
                model_repo=ident["model_repo"] or "UNKNOWN",
                quant_type=quant_type or "UNKNOWN",
                technique=ident["technique"] or "none",
                dataset=dataset,
                size_gib=_f(r.get("size_gib")),
                bpw=_f(r.get("bpw")),
                ppl=_f(r.get("ppl")),
                ppl_ratio=_f(r.get("ppl_ratio")),
                kld_mean=_f(r.get("mean_kld")),
                kld_median=_f(r.get("median_kld")),
                same_top_p=_f(r.get("same_top_p")),
                rms_dp=_f(r.get("rms_dp")),
                prefill_tok_s=_f(r.get("prefill_tok_s")),
                prefill_stdev=_f(r.get("prefill_stdev")),
                decode_tok_s=_f(r.get("decode_tok_s")),
                decode_stdev=_f(r.get("decode_stdev")),
                ttft_2k_ms=_f(r.get("ttft_2k_ms")),
                ttft_stdev_ms=_f(r.get("ttft_stdev_ms")),
                bench_repetitions=_i(r.get("bench_repetitions")),
                quant_path=quant_path,
            )
            n += 1
    return n


def _load_gguf_map(gguf_map: Path | None) -> dict[str, list[tuple[str, str]]]:
    """basename -> [(repo_id, quant_type), ...] from a ``gguf_map.json``.

    A basename can map to several repos (each experiment ships its own
    ``model-f16.gguf`` etc.); we keep them in file order so ambiguous matches can
    be assigned positionally against the reps CSV's repeated blocks.
    """
    if gguf_map is None:
        return {}
    raw = json.loads(gguf_map.read_text())
    out: dict[str, list[tuple[str, str]]] = {}
    for full_path, meta in raw.items():
        name = Path(full_path).name
        out.setdefault(name, []).append((meta["repo_id"], meta["quant_type"]))
    return out


def _quant_by_identity(
    session: Session, repo_id: str, quant_type: str
) -> Quant | None:
    return session.exec(
        select(Quant).where(
            Quant.model_repo == repo_id, Quant.quant_type == quant_type
        )
    ).first()


def _quant_by_gguf(session: Session, gguf_name: str) -> list[Quant]:
    return [
        q
        for q in session.exec(select(Quant))
        if q.quant_path and Path(q.quant_path).name == gguf_name
    ]


def import_reps_aggregated_csv(
    session: Session,
    path: Path,
    benchmark: str,
    *,
    gguf_map: Path | None = None,
) -> int:
    """Import an aggregated reps CSV as a single synthetic rep per quant.

    The aggregated CSV only carries mean/stdev (not the individual trials), so we
    materialize one rep row holding the means.

    The CSV keys ``model`` on the GGUF basename, which collides across repos. When
    ``gguf_map`` is supplied it disambiguates basename -> (repo_id, quant_type);
    repeated basenames are assigned positionally in map order. Without it, rows
    whose basename matches more than one quant are skipped with a warning.

    Returns the number of quants matched.
    """
    import warnings

    name_to_identities = _load_gguf_map(gguf_map)
    seen_counts: dict[str, int] = {}
    matched = 0

    with path.open() as f:
        for r in csv.DictReader(f):
            gguf_name = r["model"]
            quant = _resolve_quant(
                session, gguf_name, name_to_identities, seen_counts
            )
            if quant is None:
                warnings.warn(
                    f"reps row {gguf_name!r} matched no unique quant; skipped "
                    f"(supply --gguf-map to disambiguate)",
                    stacklevel=2,
                )
                continue
            sampling = _sampling_from_row(r)
            if benchmark == "mmlu":
                _add_mmlu_agg_row(session, quant, r, sampling)
            elif benchmark == "toolcall":
                _add_toolcall_agg_row(session, quant, r, sampling)
            else:
                raise ValueError(f"unknown benchmark {benchmark!r}")
            matched += 1
    return matched


def _resolve_quant(
    session: Session,
    gguf_name: str,
    name_to_identities: dict[str, list[tuple[str, str]]],
    seen_counts: dict[str, int],
) -> Quant | None:
    """Resolve a reps-CSV basename to a single Quant, using the gguf_map if given."""
    identities = name_to_identities.get(gguf_name)
    if identities:
        idx = seen_counts.get(gguf_name, 0)
        seen_counts[gguf_name] = idx + 1
        if idx < len(identities):
            repo_id, quant_type = identities[idx]
            return _quant_by_identity(session, repo_id, quant_type)
        return None
    # No map: fall back to basename match, but only if unambiguous.
    candidates = _quant_by_gguf(session, gguf_name)
    return candidates[0] if len(candidates) == 1 else None


def _sampling_from_row(r: dict[str, str]) -> dict[str, float | int | None]:
    return {
        "temperature": _f(r.get("temperature")),
        "top_p": _f(r.get("top_p")),
        "top_k": _i(r.get("top_k")),
        "min_p": _f(r.get("min_p")),
        "presence_penalty": _f(r.get("presence_penalty")),
        "repetition_penalty": _f(r.get("repetition_penalty")),
    }


def _add_mmlu_agg_row(session, quant, r, sampling) -> None:
    for existing in list(quant.mmlu_reps):
        session.delete(existing)
    session.flush()
    subj_cols = {
        f"acc_{s}": _f(r.get(f"{s}_accuracy_mean")) for s in MMLU_SUBJECTS
    }
    session.add(
        MmluRep(
            quant_id=quant.id,
            rep=0,
            accuracy=_f(r.get("accuracy_mean")),
            n_total=_i(r.get("n_total_mean")),
            n_correct=_i(r.get("n_correct_mean")),
            n_unparseable=_i(r.get("n_unparseable_mean")),
            **subj_cols,
            **sampling,
        )
    )


def _add_toolcall_agg_row(session, quant, r, sampling) -> None:
    for existing in list(quant.toolcall_reps):
        session.delete(existing)
    session.flush()
    session.add(
        ToolcallRep(
            quant_id=quant.id,
            rep=0,
            tool_selection_acc=_f(r.get("tool_selection_acc_mean")),
            param_acc_mean=_f(r.get("param_acc_mean_mean")),
            schema_valid_rate=_f(r.get("schema_valid_rate_mean")),
            continuation_type_match_rate=_f(r.get("continuation_type_match_rate_mean")),
            recovery_appropriate_rate=_f(r.get("recovery_appropriate_rate_mean")),
            n_scored=_i(r.get("n_scored_mean")),
            n_total=_i(r.get("n_total_mean")),
            n_post_result=_i(r.get("n_post_result_mean")),
            n_post_result_errors=_i(r.get("n_post_result_errors_mean")),
            **sampling,
        )
    )

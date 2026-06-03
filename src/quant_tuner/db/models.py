"""SQLModel tables for the quant-tracking database.

Four tables:

- ``Dataset``     — calibration/eval corpus identity (name, path, token count).
- ``Quant``       — one row per unique quantization (decomposed natural key +
                    one-shot size/fidelity/speed metrics + file paths).
- ``MmluRep``     — one row per MMLU-Pro trial (sampling params + per-subject acc).
- ``ToolcallRep`` — one row per tool-call trial (sampling params + headline metrics).

The natural key for a quant is
``(model_repo, quant_type, technique, variant, calib_ctx, dataset_id)``; a unique
constraint enforces it so re-running a quant updates in place rather than dupes.

Schema philosophy: fixed columns, full superset. Per-subject MMLU columns and the
extra KLD moments (stdev/skew/kurtosis) exist now even though the pipeline does not
yet compute all of them — they stay ``None`` until it does. MMMU later becomes a
sibling rep table, no migration to the tables below.
"""

from datetime import UTC, datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel

# The 14 MMLU-Pro subjects, in the order the user enumerated them. Used to build
# the fixed per-subject columns on ``MmluRep`` and to map ``by_subject`` dicts.
MMLU_SUBJECTS: tuple[str, ...] = (
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Dataset(SQLModel, table=True):
    """A calibration/eval corpus. Deduplicated by (name, path)."""

    __tablename__ = "dataset"
    __table_args__ = (UniqueConstraint("name", "path", name="uq_dataset_name_path"),)

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    path: str | None = Field(default=None)
    token_count: int | None = Field(default=None)
    notes: str | None = Field(default=None)

    quants: list["Quant"] = Relationship(back_populates="dataset")


class Quant(SQLModel, table=True):
    """One unique quantization plus its one-shot (single-run) metrics."""

    __tablename__ = "quant"
    __table_args__ = (
        UniqueConstraint(
            "model_repo",
            "quant_type",
            "technique",
            "variant",
            "calib_ctx",
            "dataset_id",
            name="uq_quant_identity",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)

    # --- natural key ------------------------------------------------------
    model_repo: str = Field(index=True)
    quant_type: str = Field(index=True)
    technique: str = Field(default="none", index=True)
    variant: str = Field(default="default")
    calib_ctx: int | None = Field(default=None)
    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")

    # --- size / weight stats ---------------------------------------------
    size_gib: float | None = Field(default=None)
    bpw: float | None = Field(default=None)
    n_params: int | None = Field(default=None)

    # --- fidelity (one-shot KLD run; superset incl. uncomputed moments) ---
    ppl: float | None = Field(default=None)
    ppl_ratio: float | None = Field(default=None)
    kld_mean: float | None = Field(default=None)
    kld_median: float | None = Field(default=None)
    kld_stdev: float | None = Field(default=None)
    kld_skew: float | None = Field(default=None)
    kld_kurtosis: float | None = Field(default=None)
    same_top_p: float | None = Field(default=None)
    rms_dp: float | None = Field(default=None)

    # --- speed (one-shot llama-bench) ------------------------------------
    prefill_tok_s: float | None = Field(default=None)
    prefill_stdev: float | None = Field(default=None)
    decode_tok_s: float | None = Field(default=None)
    decode_stdev: float | None = Field(default=None)
    ttft_2k_ms: float | None = Field(default=None)
    ttft_stdev_ms: float | None = Field(default=None)
    bench_repetitions: int | None = Field(default=None)

    # --- file paths + meta -----------------------------------------------
    quant_path: str | None = Field(default=None)
    f16_path: str | None = Field(default=None)
    imatrix_path: str | None = Field(default=None)
    workspace: str | None = Field(default=None)
    calib_params_json: str | None = Field(default=None)
    notes: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    dataset: Dataset | None = Relationship(back_populates="quants")
    mmlu_reps: list["MmluRep"] = Relationship(
        back_populates="quant",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    toolcall_reps: list["ToolcallRep"] = Relationship(
        back_populates="quant",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class _SamplingMixin(SQLModel):
    """Sampling params recorded with every trial (mirrors ``eval.toolcall.Sampling``)."""

    temperature: float | None = Field(default=None)
    top_p: float | None = Field(default=None)
    top_k: int | None = Field(default=None)
    min_p: float | None = Field(default=None)
    presence_penalty: float | None = Field(default=None)
    repetition_penalty: float | None = Field(default=None)


class MmluRep(_SamplingMixin, table=True):
    """One MMLU-Pro trial. Aggregate (mean/stdev/top@k) computed on read."""

    __tablename__ = "mmlu_rep"

    id: int | None = Field(default=None, primary_key=True)
    quant_id: int = Field(foreign_key="quant.id", index=True)

    rep: int = Field(default=0)
    seed: int | None = Field(default=None)
    elapsed_sec: float | None = Field(default=None)
    error: str | None = Field(default=None)

    accuracy: float | None = Field(default=None)
    n_total: int | None = Field(default=None)
    n_correct: int | None = Field(default=None)
    n_unparseable: int | None = Field(default=None)

    # --- per-subject accuracy (full 14-subject superset) ------------------
    acc_biology: float | None = Field(default=None)
    acc_business: float | None = Field(default=None)
    acc_chemistry: float | None = Field(default=None)
    acc_computer_science: float | None = Field(default=None)
    acc_economics: float | None = Field(default=None)
    acc_engineering: float | None = Field(default=None)
    acc_health: float | None = Field(default=None)
    acc_history: float | None = Field(default=None)
    acc_law: float | None = Field(default=None)
    acc_math: float | None = Field(default=None)
    acc_other: float | None = Field(default=None)
    acc_philosophy: float | None = Field(default=None)
    acc_physics: float | None = Field(default=None)
    acc_psychology: float | None = Field(default=None)

    quant: Quant = Relationship(back_populates="mmlu_reps")


class ToolcallRep(_SamplingMixin, table=True):
    """One tool-call trial. Aggregate (mean/stdev/top@k) computed on read."""

    __tablename__ = "toolcall_rep"

    id: int | None = Field(default=None, primary_key=True)
    quant_id: int = Field(foreign_key="quant.id", index=True)

    rep: int = Field(default=0)
    seed: int | None = Field(default=None)
    elapsed_sec: float | None = Field(default=None)
    error: str | None = Field(default=None)

    tool_selection_acc: float | None = Field(default=None)
    param_acc_mean: float | None = Field(default=None)
    param_acc_strict_mean: float | None = Field(default=None)
    schema_valid_rate: float | None = Field(default=None)
    continuation_type_match_rate: float | None = Field(default=None)
    recovery_appropriate_rate: float | None = Field(default=None)
    n_scored: int | None = Field(default=None)
    n_total: int | None = Field(default=None)
    n_post_result: int | None = Field(default=None)
    n_post_result_errors: int | None = Field(default=None)

    quant: Quant = Relationship(back_populates="toolcall_reps")

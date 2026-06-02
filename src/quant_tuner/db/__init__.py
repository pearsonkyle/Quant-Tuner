"""Local SQLite database for tracking quantizations and their benchmark metrics.

Source of truth for every quant (decomposed natural key, one-shot KLD/PPL/BPW/
speed metrics, file paths) plus per-trial benchmark reps (MMLU-Pro, tool-call).
A CSV exporter regenerates the legacy ``results.csv`` / ``reps_*.csv`` files so
existing ``render_exp*.py`` scripts keep working.

See ``src/quant_tuner/db/models.py`` for the schema.
"""

from __future__ import annotations

from quant_tuner.db.engine import (
    DEFAULT_DB_PATH,
    db_path,
    get_engine,
    init_db,
    session_scope,
)
from quant_tuner.db.models import (
    MMLU_SUBJECTS,
    Dataset,
    MmluRep,
    Quant,
    ToolcallRep,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "MMLU_SUBJECTS",
    "Dataset",
    "MmluRep",
    "Quant",
    "ToolcallRep",
    "db_path",
    "get_engine",
    "init_db",
    "session_scope",
]

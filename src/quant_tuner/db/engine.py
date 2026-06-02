"""Engine + session helpers for the quant-tracking SQLite database.

The DB file defaults to ``out/quant_tuner.db`` under the repo root and is
overridable via the ``QUANT_TUNER_DB`` env var (mirrors the ``LLAMA_CPP_DIR``
override pattern in ``paths.py``). Pass ``":memory:"`` for an ephemeral DB
(used by the test-suite).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

# Importing models registers the tables on ``SQLModel.metadata`` so
# ``init_db`` / ``create_all`` see them. Kept as a side-effecting import.
from quant_tuner.db import models as _models  # noqa: F401
from quant_tuner.paths import REPO_ROOT

DEFAULT_DB_PATH = REPO_ROOT / "out" / "quant_tuner.db"


def db_path(path: str | Path | None = None) -> str:
    """Resolve the DB path: explicit arg > ``QUANT_TUNER_DB`` env > default."""
    if path is not None:
        return str(path)
    return os.environ.get("QUANT_TUNER_DB", str(DEFAULT_DB_PATH))


def get_engine(path: str | Path | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLite engine. Ensures the parent directory exists for file DBs."""
    resolved = db_path(path)
    if resolved != ":memory:":
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    url = "sqlite://" if resolved == ":memory:" else f"sqlite:///{resolved}"
    # check_same_thread=False keeps the engine usable from helper threads;
    # all writes here are still serialized through one Session at a time.
    return create_engine(url, echo=echo, connect_args={"check_same_thread": False})


def init_db(engine: Engine) -> None:
    """Create all tables if they don't already exist."""
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Transactional session: commit on success, rollback on error."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

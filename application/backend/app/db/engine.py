# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from sqlite3 import Connection
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from app.settings import get_settings

settings = get_settings()

db_engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30},
    # Using NullPool to disable connection pooling, which is necessary for SQLite when using multiprocessing
    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    poolclass=NullPool,
    echo=settings.db_echo,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Connection, _: Any) -> None:
    """Enable foreign key support for SQLite."""
    # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(bind=db_engine)
Base = declarative_base()


@contextmanager
def get_db_session() -> Iterator[Session]:
    """Context manager to get a database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    else:
        db.commit()
    finally:
        db.close()

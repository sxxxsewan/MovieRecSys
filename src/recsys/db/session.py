"""Database URL and SQLAlchemy engine helpers."""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine


def get_database_url(url_env: str = "DATABASE_URL") -> str:
    """Return the configured database URL or the local SQLite default."""
    return os.getenv(url_env) or "sqlite:///data/recsys_app.db"


def create_engine_from_url(database_url: str | None = None) -> Engine:
    """Create an engine for PostgreSQL or the SQLite Colab/local fallback."""
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)

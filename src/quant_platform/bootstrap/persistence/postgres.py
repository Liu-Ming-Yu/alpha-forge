"""Postgres adapter composition helpers."""

from __future__ import annotations

from typing import Any


def create_pg_engine(dsn: str, **kwargs: Any) -> Any:
    """Create the configured async Postgres engine."""
    from quant_platform.infrastructure.postgres.repositories import (
        create_pg_engine as _create_pg_engine,
    )

    return _create_pg_engine(dsn, **kwargs)

"""Historical bootstrap DDL retained as packaged reference data."""

from __future__ import annotations

from importlib.resources import files

BOOTSTRAP_SQL = (
    files("quant_platform.infrastructure.postgres")
    .joinpath("schema_history.sql")
    .read_text(encoding="utf-8")
)

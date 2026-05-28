"""V2 dataset-catalog composition helpers."""

from __future__ import annotations

from typing import Any


def build_dataset_catalog(dsn: str) -> Any:
    """Build the configured V2 dataset catalog adapter."""
    from quant_platform.infrastructure.v2.postgres import (
        build_dataset_catalog as _build_dataset_catalog,
    )

    return _build_dataset_catalog(dsn)

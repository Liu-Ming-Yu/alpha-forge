"""Instrument-contract loading helpers for operator/bootstrap flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.operator.cli_inputs import (
    load_instrument_contracts as _load_contracts,
)

if TYPE_CHECKING:
    import uuid
    from pathlib import Path


def load_instrument_contracts(path: str | Path) -> dict[uuid.UUID, dict[str, object]]:
    """Load instrument-contract JSON keyed by instrument UUID."""
    return _load_contracts(path)

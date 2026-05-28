"""Shared helpers for operator adapters."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from quant_platform.config import PlatformSettings


def contract_reference_prices(
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[uuid.UUID, Decimal]:
    """Return positive ``last_close`` prices from contract metadata."""
    prices: dict[uuid.UUID, Decimal] = {}
    for instrument_id, contract in contracts.items():
        raw_last_close = contract.get("last_close")
        if raw_last_close is None:
            continue
        price = Decimal(str(raw_last_close))
        if price > 0:
            prices[instrument_id] = price
    return prices


def default_paper_soak_dir(settings: PlatformSettings) -> Path:
    """Return the canonical paper-soak artifact directory."""
    return Path(settings.storage.object_store_root) / "paper_soak"


def latest_paper_soak(settings: PlatformSettings) -> Path | None:
    """Return the newest paper-soak artifact, if any."""
    root = default_paper_soak_dir(settings)
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.glob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


__all__ = ["contract_reference_prices", "default_paper_soak_dir", "latest_paper_soak"]

"""Dataset quorum governance operations."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.results import ResultPresentation, UseCaseResult

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import DatasetCatalog
    from quant_platform.core.domain.market_data import MarketBar


async def dataset_quorum_command(
    settings: PlatformSettings,
    *,
    subcommand: str,
    dataset_kind: str,
    as_of: datetime,
    vendor_bars: Path | None = None,
    required_vendor_count: int = 2,
    max_disagreement_bps: Decimal = Decimal("50"),
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.infrastructure.v2.postgres import build_dataset_catalog

    catalog = build_dataset_catalog(settings.storage.postgres_dsn)
    if subcommand == "latest":
        evidence = await catalog.latest_quorum_evidence(dataset_kind, as_of=as_of)
        if evidence is None:
            return UseCaseResult(
                payload={"dataset_kind": dataset_kind, "evidence": None},
                presentation=ResultPresentation.JSON,
            )
        return UseCaseResult(
            payload={
                "evidence_id": str(evidence.evidence_id),
                "dataset_kind": evidence.dataset_kind,
                "as_of": evidence.as_of.astimezone(UTC).isoformat(),
                "vendors": list(evidence.vendors),
                "passed": evidence.passed,
                "required_vendor_count": evidence.required_vendor_count,
                "max_disagreement_bps": float(evidence.max_disagreement_bps),
                "details": evidence.details,
            },
            presentation=ResultPresentation.JSON,
        )
    if subcommand == "record":
        if vendor_bars is None:
            raise OperatorUsageError("dataset-quorum record requires --vendor-bars")
        return await _record_dataset_quorum(
            catalog,
            dataset_kind=dataset_kind,
            as_of=as_of,
            vendor_bars=vendor_bars,
            required_vendor_count=required_vendor_count,
            max_disagreement_bps=max_disagreement_bps,
        )
    raise OperatorUsageError(f"unknown dataset-quorum command: {subcommand}")


async def _record_dataset_quorum(
    catalog: DatasetCatalog,
    *,
    dataset_kind: str,
    as_of: datetime,
    vendor_bars: Path,
    required_vendor_count: int,
    max_disagreement_bps: Decimal,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.data_service.maintenance.dataset_quorum import (
        compute_dataset_quorum_evidence,
    )

    raw = json.loads(Path(vendor_bars).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OperatorUsageError("--vendor-bars must contain a JSON object: {vendor: [bars]}")
    parsed_vendor_bars: dict[str, list[MarketBar]] = {}
    for vendor, bar_list in raw.items():
        if not isinstance(bar_list, list):
            raise OperatorUsageError(f"vendor {vendor!r} must map to a list of bars")
        parsed_vendor_bars[vendor] = [_market_bar_from_payload(vendor, entry) for entry in bar_list]

    evidence = compute_dataset_quorum_evidence(
        parsed_vendor_bars,
        dataset_kind=dataset_kind,
        as_of=as_of,
        required_vendor_count=required_vendor_count,
        max_disagreement_bps=max_disagreement_bps,
    )
    await catalog.record_quorum_evidence(evidence)
    return UseCaseResult(
        payload={
            "evidence_id": str(evidence.evidence_id),
            "dataset_kind": evidence.dataset_kind,
            "passed": evidence.passed,
            "vendors": list(evidence.vendors),
            "max_disagreement_bps": float(evidence.max_disagreement_bps),
            "details": evidence.details,
        },
        presentation=ResultPresentation.JSON,
    )


def _market_bar_from_payload(vendor: str, entry: object) -> MarketBar:
    from quant_platform.core.domain.market_data import MarketBar

    if not isinstance(entry, dict):
        raise OperatorUsageError(f"vendor {vendor!r} bar must be an object")
    close = Decimal(str(entry["close"]))
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=uuid.UUID(str(entry["instrument_id"])),
        timestamp=datetime.fromisoformat(str(entry["timestamp"])),
        bar_seconds=int(entry.get("bar_seconds", 86400)),
        open=Decimal(str(entry.get("open", close))),
        high=Decimal(str(entry.get("high", close))),
        low=Decimal(str(entry.get("low", close))),
        close=close,
        volume=int(entry.get("volume", 0)),
    )


__all__ = ["dataset_quorum_command"]

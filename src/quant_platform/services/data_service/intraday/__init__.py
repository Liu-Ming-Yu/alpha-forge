"""Vendor-neutral intraday bar import, validation, and dataset evidence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data import (
    BarDataset,
    DataLakeLayer,
    DataQualityStatus,
    DatasetQuorumEvidence,
    VendorBarBatch,
)
from quant_platform.services.data_service.intraday.intraday_file_loader import (
    FileHistoricalBarVendorAdapter,
    load_vendor_bar_batch_from_file,
    write_vendor_bar_batch_to_file,
)
from quant_platform.services.data_service.intraday.intraday_schema import (
    INTRADAY_SCHEMA_HASH,
    canonical_intraday_bar_id,
)
from quant_platform.services.data_service.intraday.intraday_validation import (
    INTRADAY_BAR_SECONDS,
    IntradayValidationIssue,
    IntradayValidationReport,
    validate_vendor_bar_batch,
    validation_payload,
)
from quant_platform.services.data_service.maintenance.dataset_quorum import (
    compute_dataset_quorum_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.contracts.data import (
        DatasetCatalog,
        HistoricalBarVendorAdapter,
        HistoricalDataStore,
    )

__all__ = [
    "FileHistoricalBarVendorAdapter",
    "INTRADAY_BAR_SECONDS",
    "INTRADAY_SCHEMA_HASH",
    "IntradayImportResult",
    "IntradayValidationIssue",
    "IntradayValidationReport",
    "build_intraday_vendor_adapter",
    "canonical_intraday_bar_id",
    "compute_intraday_quorum_evidence",
    "import_result_payload",
    "import_vendor_bar_batch",
    "load_vendor_bar_batch_from_file",
    "validate_vendor_bar_batch",
    "validation_payload",
    "write_vendor_bar_batch_to_file",
]


@dataclass(frozen=True)
class IntradayImportResult:
    """Result of storing a validated intraday vendor batch."""

    dataset: BarDataset
    validation: IntradayValidationReport
    rows_stored: int


async def import_vendor_bar_batch(
    batch: VendorBarBatch,
    *,
    store: HistoricalDataStore,
    catalog: DatasetCatalog | None = None,
    expected_instruments: set[uuid.UUID] | None = None,
    quality: DataQualityStatus = DataQualityStatus.APPROVED,
) -> IntradayImportResult:
    """Validate, persist, and optionally catalog one intraday vendor batch."""
    validation = validate_vendor_bar_batch(batch, expected_instruments=expected_instruments)
    if not validation.passed:
        quality = DataQualityStatus.QUARANTINED

    await store.store_bars(list(batch.bars))
    start_at = validation.start_at or batch.fetched_at
    end_at = validation.end_at or batch.fetched_at
    dataset = BarDataset(
        dataset_id=uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"intraday:{batch.vendor}:{batch.source_uri}:{batch.bar_seconds}:{start_at}:{end_at}",
        ),
        layer=DataLakeLayer.SILVER,
        vendor=batch.vendor,
        bar_seconds=batch.bar_seconds,
        start_at=start_at,
        end_at=end_at,
        as_of=batch.fetched_at,
        available_at=batch.fetched_at,
        schema_hash=INTRADAY_SCHEMA_HASH,
        source_uri=batch.source_uri,
        row_count=len(batch.bars),
        quality=quality,
    )
    if catalog is not None:
        await catalog.register_bar_dataset(dataset)
    return IntradayImportResult(dataset=dataset, validation=validation, rows_stored=len(batch.bars))


def compute_intraday_quorum_evidence(
    batches: Mapping[str, VendorBarBatch],
    *,
    as_of: datetime,
    required_vendor_count: int = 2,
    max_disagreement_bps: Decimal = Decimal("50"),
) -> DatasetQuorumEvidence:
    """Compute vendor-quorum evidence for 1-minute intraday bars."""
    return compute_dataset_quorum_evidence(
        {vendor: list(batch.bars) for vendor, batch in batches.items()},
        dataset_kind="bars_intraday_1m",
        as_of=as_of,
        required_vendor_count=required_vendor_count,
        max_disagreement_bps=max_disagreement_bps,
    )


def build_intraday_vendor_adapter(
    *,
    vendor: str,
    settings: object,
    symbol_by_instrument_id: Mapping[uuid.UUID, str],
) -> HistoricalBarVendorAdapter:
    """Build a configured HistoricalBarVendorAdapter by vendor name."""
    normalized = vendor.strip().lower()
    if normalized == "polygon":
        from quant_platform.services.data_service.intraday.polygon_intraday import (
            PolygonHistoricalBarVendorAdapter,
        )

        data_ingest = getattr(settings, "data_ingest", settings)
        return PolygonHistoricalBarVendorAdapter(
            api_key=str(getattr(data_ingest, "polygon_api_key", "")),
            symbol_by_instrument_id=symbol_by_instrument_id,
            base_url=str(getattr(data_ingest, "polygon_base_url", "https://api.polygon.io")),
            max_concurrent=int(getattr(data_ingest, "polygon_max_concurrent", 4)),
            min_request_interval_seconds=float(
                getattr(data_ingest, "polygon_min_request_interval_seconds", 0.0)
            ),
            timeout_seconds=float(getattr(data_ingest, "polygon_timeout_seconds", 30.0)),
        )
    raise ValueError(f"unsupported intraday historical vendor: {vendor}")


def import_result_payload(result: IntradayImportResult) -> dict[str, object]:
    """Return JSON-safe import result payload."""
    dataset = result.dataset
    return {
        "dataset": {
            "dataset_id": str(dataset.dataset_id),
            "layer": dataset.layer.value,
            "vendor": dataset.vendor,
            "bar_seconds": dataset.bar_seconds,
            "start_at": dataset.start_at.isoformat(),
            "end_at": dataset.end_at.isoformat(),
            "as_of": dataset.as_of.isoformat(),
            "available_at": dataset.available_at.isoformat(),
            "schema_hash": dataset.schema_hash,
            "source_uri": dataset.source_uri,
            "row_count": dataset.row_count,
            "quality": dataset.quality.value,
        },
        "rows_stored": result.rows_stored,
        "validation": validation_payload(result.validation),
    }

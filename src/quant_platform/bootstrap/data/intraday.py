"""Intraday data command composition helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.results import ResultPresentation, UseCaseResult, UseCaseStatus

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.application.operator.requests import IntradayCommandRequest
    from quant_platform.config import PlatformSettings


def _json_result(
    payload: dict[str, object],
    *,
    passed: bool,
) -> UseCaseResult[dict[str, object]]:
    """Return a JSON-rendered result, blocked with exit code 2 when not passed."""
    return UseCaseResult(
        status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
        payload=payload,
        exit_code=0 if passed else 2,
        presentation=ResultPresentation.JSON,
    )


async def run_intraday_command(
    settings: PlatformSettings,
    *,
    request: IntradayCommandRequest,
    contracts: Mapping[uuid.UUID, dict[str, object]],
    vendor_files: tuple[tuple[str, Path], ...],
) -> UseCaseResult[dict[str, object]]:
    """Dispatch intraday data import, validation, fetch, and quorum commands."""
    from quant_platform.application.operator.requests import (
        IntradayFetchRequest,
        IntradayImportRequest,
        IntradayQuorumRequest,
        IntradayValidateRequest,
    )
    from quant_platform.infrastructure.v2.postgres import build_dataset_catalog
    from quant_platform.services.data_service.intraday import (
        FileHistoricalBarVendorAdapter,
        build_intraday_vendor_adapter,
        compute_intraday_quorum_evidence,
        import_result_payload,
        import_vendor_bar_batch,
        load_vendor_bar_batch_from_file,
        validate_vendor_bar_batch,
        validation_payload,
        write_vendor_bar_batch_to_file,
    )
    from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore

    lookup = _instrument_lookup_from_contracts(contracts)
    instrument_ids = set(contracts.keys())
    if isinstance(request, IntradayValidateRequest):
        batch = load_vendor_bar_batch_from_file(
            request.input,
            vendor=request.vendor,
            instrument_lookup=lookup,
            as_of=request.as_of,
        )
        report = validate_vendor_bar_batch(batch, expected_instruments=instrument_ids)
        return _json_result(validation_payload(report), passed=report.passed)
    if isinstance(request, IntradayImportRequest):
        batch = load_vendor_bar_batch_from_file(
            request.input,
            vendor=request.vendor,
            instrument_lookup=lookup,
            as_of=request.as_of,
        )
        catalog = build_dataset_catalog(settings.storage.postgres_dsn)
        result = await import_vendor_bar_batch(
            batch,
            store=ParquetBarStore(settings.storage.object_store_root),
            catalog=catalog,
            expected_instruments=instrument_ids,
        )
        passed = result.validation.passed or request.allow_quarantined
        return _json_result(import_result_payload(result), passed=passed)
    if isinstance(request, IntradayFetchRequest):
        adapter = build_intraday_vendor_adapter(
            vendor=request.vendor,
            settings=settings,
            symbol_by_instrument_id=_symbol_by_instrument_from_contracts(contracts),
        )
        batch = await adapter.fetch_bars(
            list(contracts.keys()),
            request.start if request.start.tzinfo else request.start.replace(tzinfo=UTC),
            request.end if request.end.tzinfo else request.end.replace(tzinfo=UTC),
            60,
            as_of=request.as_of if request.as_of.tzinfo else request.as_of.replace(tzinfo=UTC),
        )
        frozen_path = None
        if request.output_file is not None:
            frozen_path = write_vendor_bar_batch_to_file(
                batch,
                request.output_file,
                symbol_by_instrument_id=_symbol_by_instrument_from_contracts(contracts),
            )
        catalog = build_dataset_catalog(settings.storage.postgres_dsn)
        result = await import_vendor_bar_batch(
            batch,
            store=ParquetBarStore(settings.storage.object_store_root),
            catalog=catalog,
            expected_instruments=instrument_ids,
        )
        payload = import_result_payload(result)
        if frozen_path is not None:
            payload["frozen_output_file"] = str(frozen_path)
        passed = result.validation.passed or request.allow_quarantined
        return _json_result(payload, passed=passed)
    if isinstance(request, IntradayQuorumRequest):
        batches = {}
        for vendor, file_path in vendor_files:
            adapter = FileHistoricalBarVendorAdapter(
                file_path,
                vendor=vendor,
                instrument_lookup=lookup,
            )
            batches[vendor] = await adapter.fetch_bars(
                list(contracts.keys()),
                datetime.min.replace(tzinfo=UTC),
                datetime.max.replace(tzinfo=UTC),
                60,
                as_of=request.as_of,
            )
        evidence = compute_intraday_quorum_evidence(
            batches,
            as_of=request.as_of,
            required_vendor_count=request.required_vendor_count,
            max_disagreement_bps=request.max_disagreement_bps,
        )
        catalog = build_dataset_catalog(settings.storage.postgres_dsn)
        await catalog.record_quorum_evidence(evidence)
        return _json_result(
            {
                "evidence_id": str(evidence.evidence_id),
                "dataset_kind": evidence.dataset_kind,
                "passed": evidence.passed,
                "vendors": list(evidence.vendors),
                "required_vendor_count": evidence.required_vendor_count,
                "max_disagreement_bps": float(evidence.max_disagreement_bps),
                "details": evidence.details,
            },
            passed=evidence.passed,
        )
    raise OperatorUsageError(f"unknown intraday request: {type(request).__name__}")


def _instrument_lookup_from_contracts(
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[str, uuid.UUID]:
    lookup: dict[str, uuid.UUID] = {}
    for instrument_id, spec in contracts.items():
        lookup[str(instrument_id)] = instrument_id
        symbol = spec.get("symbol")
        if symbol:
            lookup[str(symbol).upper()] = instrument_id
    return lookup


def _symbol_by_instrument_from_contracts(
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[uuid.UUID, str]:
    symbols: dict[uuid.UUID, str] = {}
    for instrument_id, spec in contracts.items():
        symbol = spec.get("symbol")
        if symbol:
            symbols[instrument_id] = str(symbol).upper()
    return symbols


__all__ = ["run_intraday_command"]

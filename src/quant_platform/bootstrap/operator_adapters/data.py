"""Data operator adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.operator.cli_inputs import (
    load_instrument_contracts,
    parse_vendor_file,
)

if TYPE_CHECKING:
    from quant_platform.application.data import (
        ComputeFeaturesRequest,
        DataHealthRequest,
        IngestRequest,
        MaintainDataRequest,
        ReprocessCorporateActionsRequest,
    )
    from quant_platform.application.operator.requests import IntradayCommandRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


class DataAdapters:
    """Concrete data adapters backed by bootstrap data operations."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    async def compute_features(self, request: ComputeFeaturesRequest) -> None:
        from quant_platform.bootstrap.data import compute_features

        contracts = (
            load_instrument_contracts(request.contracts_file) if request.contracts_file else {}
        )
        await compute_features(self._settings, instrument_contracts=contracts or None)

    async def ingest(self, request: IngestRequest) -> UseCaseResult[None]:
        from quant_platform.bootstrap.data import ingest_bars

        return await ingest_bars(
            settings=self._settings,
            start=request.start,
            end=request.end,
            instrument_contracts=load_instrument_contracts(request.contracts_file),
            bar_seconds=request.bar_seconds,
            source=request.source,
        )

    async def maintain(self, request: MaintainDataRequest) -> None:
        from quant_platform.bootstrap.data import maintain_data

        contracts = (
            load_instrument_contracts(request.contracts_file) if request.contracts_file else {}
        )
        await maintain_data(
            self._settings,
            interval_seconds=request.interval_seconds,
            backfill_start=request.backfill_start,
            backfill_end=request.backfill_end,
            instrument_contracts=contracts or None,
        )

    async def reprocess_ca(self, request: ReprocessCorporateActionsRequest) -> None:
        from quant_platform.bootstrap.data import reprocess_corporate_actions

        await reprocess_corporate_actions(self._settings, instrument_id=request.instrument_id)

    async def intraday(self, request: IntradayCommandRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.application.operator.requests import IntradayQuorumRequest
        from quant_platform.bootstrap.data import run_intraday_command

        contracts = load_instrument_contracts(request.contracts_file)
        vendor_files = (
            tuple(parse_vendor_file(vendor_file) for vendor_file in request.vendor_file)
            if isinstance(request, IntradayQuorumRequest)
            else ()
        )
        return await run_intraday_command(
            self._settings,
            request=request,
            contracts=contracts,
            vendor_files=vendor_files,
        )

    async def health(self, request: DataHealthRequest) -> tuple[dict[str, object], bool]:
        from quant_platform.bootstrap.data import data_health_payload_for_contracts

        return await data_health_payload_for_contracts(
            self._settings,
            contracts=load_instrument_contracts(request.contracts_file),
            start=request.start,
            end=request.end,
            bar_seconds=request.bar_seconds,
        )


__all__ = ["DataAdapters"]

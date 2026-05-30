"""Minimal data-maintenance scheduler for ingest, liquidity, and feature jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.research import FeatureRequest
from quant_platform.core.domain.signals.feature_inputs import (
    BARS_EOD_INPUT,
    CLOSE_SERIES_INPUT,
    FeatureInputContext,
)
from quant_platform.services.data_service.ingest.daily_ingest import refresh_liquidity_from_store
from quant_platform.telemetry.feature_metrics import emit_feature_distribution_metrics

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from quant_platform.application.features.registry import FeatureFamilyRegistry
    from quant_platform.core.contracts import (
        FeatureRepository,
        HistoricalDataStore,
        MarketDataProvider,
    )
    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.data_service.feeds.failover_bar_fetcher import BarFetcher
    from quant_platform.services.data_service.reference.universe_manager import UniverseManager

log = structlog.get_logger(__name__)
DEFAULT_FEATURE_SET_VERSION = "1.1.0"


@dataclass
class DataMaintenanceResult:
    bars_ingested: int = 0
    liquidity_profiles_updated: int = 0
    features_stored: int = 0
    feature_data: dict[uuid.UUID, dict[str, float]] = field(default_factory=dict)


class DataMaintenanceScheduler:
    """Runs lightweight ingest + liquidity + feature jobs on a cadence.

    Feature jobs dispatch through the typed :class:`FeatureFamilyRegistry`: the
    scheduler resolves the family for the requested ``feature_set_version``,
    packages bar/close history into a :class:`FeatureRequest`, runs the registry
    computer, and persists the returned vectors.
    """

    def __init__(
        self,
        *,
        instruments: list[Instrument],
        bar_store: HistoricalDataStore,
        universe_manager: UniverseManager,
        feature_repo: FeatureRepository,
        market_data_provider: MarketDataProvider | None = None,
        bar_fetcher: BarFetcher | None = None,
        bar_seconds: int = 86400,
        lookback_days: int = 380,
        liquidity_lookback_days: int = 21,
        ingest_lookback_days: int = 7,
        artifact_uri: str = "",
        feature_registry: FeatureFamilyRegistry | None = None,
    ) -> None:
        self._instruments = list(instruments)
        self._bar_store = bar_store
        self._universe_manager = universe_manager
        self._feature_repo = feature_repo
        self._market_data_provider = market_data_provider
        # Optional vendor bar fetcher (Tiingo/Polygon). When set, the daily ingest
        # pulls fresh EOD bars for stale names from the vendor — this is what keeps
        # a multi-day paper soak current without a live IB feed. Opt-in: only
        # non-None when QP__DATA_INGEST__BAR_FETCH_FALLBACK_CHAIN is configured, so
        # IB-live and existing behavior are unchanged.
        self._bar_fetcher = bar_fetcher
        self._bar_seconds = bar_seconds
        self._lookback_days = lookback_days
        self._liquidity_lookback_days = liquidity_lookback_days
        self._ingest_lookback_days = ingest_lookback_days
        if artifact_uri:
            self._artifact_uri = artifact_uri
        else:
            root_value = getattr(bar_store, "_root", None)
            self._artifact_uri = Path(root_value).resolve().as_uri() if root_value else ""
        self._feature_registry = feature_registry

    async def run_once(
        self,
        *,
        strategy_run_id: uuid.UUID,
        as_of: datetime,
        feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    ) -> DataMaintenanceResult:
        result = DataMaintenanceResult()
        result.bars_ingested = await self._ingest_latest_bars(as_of)
        result.liquidity_profiles_updated = await refresh_liquidity_from_store(
            instruments=self._instruments,
            bar_store=self._bar_store,
            universe_manager=self._universe_manager,
            as_of=as_of,
            lookback_days=self._liquidity_lookback_days,
        )

        result.feature_data = await self._run_feature_job(
            strategy_run_id=strategy_run_id,
            as_of=as_of,
            feature_set_version=feature_set_version,
        )
        result.features_stored = len(result.feature_data)

        log.info(
            "data_maintenance.run_once",
            bars_ingested=result.bars_ingested,
            liquidity_profiles_updated=result.liquidity_profiles_updated,
            features_stored=result.features_stored,
            instruments=len(self._instruments),
            as_of=str(as_of),
        )
        return result

    async def _run_feature_job(
        self,
        *,
        strategy_run_id: uuid.UUID,
        as_of: datetime,
        feature_set_version: str,
    ) -> dict[uuid.UUID, dict[str, float]]:
        """Compute and persist one feature family/version through the registry."""
        if self._feature_registry is None:
            return {}
        family = self._feature_registry.family_for_version(feature_set_version)
        if family is None:
            log.debug(
                "data_maintenance.feature_job_skipped",
                feature_set_version=feature_set_version,
                reason="no_feature_family_registered",
            )
            return {}

        plugin = self._feature_registry.get(
            feature_family=family,
            feature_set_version=feature_set_version,
        )
        input_key = plugin.required_inputs[0] if plugin.required_inputs else BARS_EOD_INPUT
        history: Mapping[uuid.UUID, object]
        if input_key == CLOSE_SERIES_INPUT:
            history = await self._load_close_history(as_of)
        else:
            history = await self._load_bar_history(as_of)
        if not history:
            return {}

        request = FeatureRequest(
            feature_set_version=feature_set_version,
            instruments=tuple(history),
            start=as_of - timedelta(days=self._lookback_days),
            end=as_of,
            as_of=as_of,
            strategy_run_id=strategy_run_id,
            artifact_uri=self._artifact_uri,
            context=FeatureInputContext(
                available_inputs=(input_key,),
                payloads={input_key: history},
            ),
        )
        feature_result = await self._feature_registry.compute(
            feature_family=family,
            request=request,
        )
        if not feature_result.passed:
            log.warning(
                "data_maintenance.feature_job_blocked",
                feature_set_version=feature_set_version,
                diagnostics=dict(feature_result.diagnostics),
            )
            return {}

        feature_data: dict[uuid.UUID, dict[str, float]] = {}
        for vector in feature_result.vectors:
            await self._feature_repo.store_vector(vector)
            feature_data[vector.instrument_id] = dict(vector.features)
        emit_feature_distribution_metrics(feature_data, feature_set_version)
        return feature_data

    async def _ingest_latest_bars(self, as_of: datetime) -> int:
        # A configured vendor fetcher takes precedence (the opt-in daily-refresh
        # path that keeps a paper soak current); otherwise the live market-data
        # provider (IB) is used, preserving existing behavior.
        if self._bar_fetcher is not None:
            return await self._ingest_via_vendor(as_of)
        if self._market_data_provider is None:
            return 0
        bars = []
        for inst in self._instruments:
            bar = await self._market_data_provider.get_last_bar(
                inst.instrument_id,
                self._bar_seconds,
            )
            if bar is None or not bar.is_complete:
                continue
            bars.append(bar)
        if not bars:
            return 0
        await self._bar_store.store_bars(bars)
        return len(bars)

    async def _ingest_via_vendor(self, as_of: datetime) -> int:
        """Fetch recent EOD bars for stale names from the configured vendor.

        Self-throttling: only instruments missing a bar on/after the previous
        business day are fetched, so steady state pulls just the newest trading
        day (and nothing once current). The vendor fetcher's own circuit breaker
        bounds rate-limit pressure when many names are stale (e.g. first run after
        a gap), and the bar store dedups, so re-fetch is harmless. Returns the
        number of new bars written.
        """
        if self._bar_fetcher is None:  # pragma: no cover - guarded by caller
            return 0
        cutoff = self._previous_business_day(as_of.date())
        cutoff_dt = datetime(cutoff.year, cutoff.month, cutoff.day, tzinfo=as_of.tzinfo)
        stale: list[Instrument] = []
        for inst in self._instruments:
            recent = await self._bar_store.get_bars(
                inst.instrument_id, self._bar_seconds, cutoff_dt, as_of
            )
            if not recent:
                stale.append(inst)
        if not stale:
            return 0
        start = (as_of - timedelta(days=self._ingest_lookback_days)).date()
        try:
            bars = await self._bar_fetcher(stale, start, as_of.date())
        except Exception as exc:  # noqa: BLE001 - best-effort refresh, never crash the cycle
            log.warning("data_maintenance.vendor_ingest_failed", error=str(exc), stale=len(stale))
            return 0
        if bars:
            await self._bar_store.store_bars(bars)
        log.info(
            "data_maintenance.vendor_ingest",
            stale=len(stale),
            fetched=len(bars),
            cutoff=str(cutoff),
        )
        return len(bars)

    @staticmethod
    def _previous_business_day(d: date) -> date:
        """Most recent weekday strictly before ``d`` (ignores market holidays)."""
        prev = d - timedelta(days=1)
        while prev.weekday() >= 5:  # Saturday=5, Sunday=6
            prev -= timedelta(days=1)
        return prev

    async def _load_close_history(
        self,
        as_of: datetime,
    ) -> dict[uuid.UUID, list[float]]:
        start = as_of - timedelta(days=self._lookback_days)
        bar_data: dict[uuid.UUID, list[float]] = {}
        for inst in self._instruments:
            bars = await self._bar_store.get_bars(
                inst.instrument_id,
                self._bar_seconds,
                start,
                as_of,
            )
            if len(bars) < 21:
                continue
            bar_data[inst.instrument_id] = [float(b.close) for b in bars]
        return bar_data

    async def _load_bar_history(
        self,
        as_of: datetime,
    ) -> dict[uuid.UUID, list[MarketBar]]:
        start = as_of - timedelta(days=self._lookback_days)
        bar_data: dict[uuid.UUID, list[MarketBar]] = {}
        for inst in self._instruments:
            bars = await self._bar_store.get_bars(
                inst.instrument_id,
                self._bar_seconds,
                start,
                as_of,
            )
            if len(bars) < 21:
                continue
            bar_data[inst.instrument_id] = list(bars)
        return bar_data

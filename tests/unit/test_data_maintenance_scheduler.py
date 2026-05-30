"""Unit tests for DataMaintenanceScheduler."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.bootstrap.data.feature_plugins import build_feature_registry
from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
    DataMaintenanceScheduler,
)
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.services.data_service.reference.universe_manager import UniverseManager
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    TEXT_CATALYST_V5_EVENT_FEATURE_SET_VERSION,
    TEXT_CATALYST_V10_ALPHA_FEATURES,
)

_UTC = UTC


def _make_bar(
    instrument_id: uuid.UUID,
    ts: datetime,
    close: Decimal,
    volume: int,
) -> MarketBar:
    low = close - Decimal("1")
    high = close + Decimal("1")
    return MarketBar(
        bar_id=uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{instrument_id}:{ts.isoformat()}",
        ),
        instrument_id=instrument_id,
        timestamp=ts,
        bar_seconds=86400,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=close,
        is_complete=True,
    )


class _InMemoryBarStore:
    def __init__(self) -> None:
        self._bars: dict[uuid.UUID, list[MarketBar]] = {}

    async def store_bars(self, bars: list[MarketBar]) -> None:
        for bar in bars:
            rows = self._bars.setdefault(bar.instrument_id, [])
            if bar.bar_id not in {existing.bar_id for existing in rows}:
                rows.append(bar)
        for inst_id in self._bars:
            self._bars[inst_id].sort(key=lambda b: b.timestamp)

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        return [
            bar
            for bar in self._bars.get(instrument_id, [])
            if bar.bar_seconds == bar_seconds and start <= bar.timestamp <= end
        ]

    async def get_corporate_actions(self, instrument_id: uuid.UUID, since):
        _ = instrument_id
        _ = since
        return []


def _make_scheduler(
    *,
    instrument: Instrument,
    bar_store: _InMemoryBarStore,
    universe: UniverseManager,
    feature_repo: InMemoryFeatureRepository,
) -> DataMaintenanceScheduler:
    return DataMaintenanceScheduler(
        instruments=[instrument],
        bar_store=bar_store,
        universe_manager=universe,
        feature_repo=feature_repo,
        feature_registry=build_feature_registry(feature_repo),
    )


class _FakeVendorFetcher:
    """Records which instruments it was asked for and returns canned bars."""

    def __init__(self, bars_by_inst: dict[uuid.UUID, list[MarketBar]]) -> None:
        self._bars_by_inst = bars_by_inst
        self.calls: list[list[uuid.UUID]] = []

    async def __call__(
        self, instruments: list[Instrument], start: date, end: date
    ) -> list[MarketBar]:
        self.calls.append([i.instrument_id for i in instruments])
        out: list[MarketBar] = []
        for inst in instruments:
            out.extend(self._bars_by_inst.get(inst.instrument_id, []))
        return out


def _inst(symbol: str) -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        exchange="XNAS",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        sector="Information Technology",
    )


@pytest.mark.asyncio
async def test_vendor_ingest_fetches_only_stale_names() -> None:
    fresh, stale = _inst("AAPL"), _inst("MSFT")
    as_of = datetime(2026, 4, 10, 20, 0, tzinfo=_UTC)
    today_mid = datetime(as_of.year, as_of.month, as_of.day, tzinfo=_UTC)
    bar_store = _InMemoryBarStore()
    # fresh has a bar dated today (>= previous business day → current);
    # stale's latest is 30 days back (older than the cutoff → must be refreshed).
    await bar_store.store_bars(
        [_make_bar(fresh.instrument_id, today_mid, Decimal("100"), 1_000_000)]
    )
    await bar_store.store_bars(
        [_make_bar(stale.instrument_id, as_of - timedelta(days=30), Decimal("50"), 500_000)]
    )
    new_bar = _make_bar(stale.instrument_id, today_mid, Decimal("55"), 600_000)
    fetcher = _FakeVendorFetcher({stale.instrument_id: [new_bar]})

    scheduler = DataMaintenanceScheduler(
        instruments=[fresh, stale],
        bar_store=bar_store,
        universe_manager=UniverseManager(ContractMaster([fresh, stale])),
        feature_repo=InMemoryFeatureRepository(),
        bar_fetcher=fetcher,
    )
    written = await scheduler._ingest_via_vendor(as_of)  # noqa: SLF001

    assert fetcher.calls == [[stale.instrument_id]]  # ONLY the stale name was fetched
    assert written == 1
    refreshed = await bar_store.get_bars(stale.instrument_id, 86400, today_mid, as_of)
    assert len(refreshed) == 1  # the new bar landed in the store


@pytest.mark.asyncio
async def test_vendor_ingest_noop_when_all_current() -> None:
    a, b = _inst("AAPL"), _inst("MSFT")
    as_of = datetime(2026, 4, 10, 20, 0, tzinfo=_UTC)
    today_mid = datetime(as_of.year, as_of.month, as_of.day, tzinfo=_UTC)
    bar_store = _InMemoryBarStore()
    for inst in (a, b):
        await bar_store.store_bars(
            [_make_bar(inst.instrument_id, today_mid, Decimal("100"), 1_000)]
        )
    fetcher = _FakeVendorFetcher({})

    scheduler = DataMaintenanceScheduler(
        instruments=[a, b],
        bar_store=bar_store,
        universe_manager=UniverseManager(ContractMaster([a, b])),
        feature_repo=InMemoryFeatureRepository(),
        bar_fetcher=fetcher,
    )
    written = await scheduler._ingest_via_vendor(as_of)  # noqa: SLF001

    assert written == 0
    assert fetcher.calls == []  # nothing stale ⇒ no vendor request


def test_previous_business_day_skips_weekend() -> None:
    # Monday → Friday; Tuesday → Monday; Saturday → Friday.
    assert DataMaintenanceScheduler._previous_business_day(date(2026, 4, 13)) == date(2026, 4, 10)
    assert DataMaintenanceScheduler._previous_business_day(date(2026, 4, 14)) == date(2026, 4, 13)
    assert DataMaintenanceScheduler._previous_business_day(date(2026, 4, 11)) == date(2026, 4, 10)


@pytest.mark.asyncio
async def test_scheduler_refreshes_liquidity_and_computes_features() -> None:
    instrument = Instrument(
        instrument_id=uuid.uuid4(),
        symbol="AAPL",
        exchange="XNAS",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        sector="Information Technology",
    )
    master = ContractMaster([instrument])
    bar_store = _InMemoryBarStore()
    universe = UniverseManager(master)
    feature_repo = InMemoryFeatureRepository()
    as_of = datetime(2026, 4, 10, 20, 0, tzinfo=_UTC)

    bars = [
        _make_bar(
            instrument.instrument_id,
            as_of - timedelta(days=90 - i),
            Decimal("100") + Decimal(i) * Decimal("0.2"),
            1_000_000 + i * 1000,
        )
        for i in range(90)
    ]
    await bar_store.store_bars(bars)

    scheduler = _make_scheduler(
        instrument=instrument,
        bar_store=bar_store,
        universe=universe,
        feature_repo=feature_repo,
    )
    result = await scheduler.run_once(
        strategy_run_id=uuid.uuid4(),
        as_of=as_of,
    )
    assert result.liquidity_profiles_updated == 1
    assert result.features_stored == 1
    assert instrument.instrument_id in result.feature_data
    profile = universe.get_profile(instrument.instrument_id)
    assert profile is not None
    assert profile.adv_usd_20d > 0


@pytest.mark.asyncio
async def test_scheduler_emits_feature_distribution_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = Instrument(
        instrument_id=uuid.uuid4(),
        symbol="AAPL",
        exchange="XNAS",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        sector="Information Technology",
    )
    master = ContractMaster([instrument])
    bar_store = _InMemoryBarStore()
    universe = UniverseManager(master)
    feature_repo = InMemoryFeatureRepository()
    as_of = datetime(2026, 4, 10, 20, 0, tzinfo=_UTC)

    bars = [
        _make_bar(
            instrument.instrument_id,
            as_of - timedelta(days=280 - i),
            Decimal("100") + Decimal(i) * Decimal("0.2"),
            1_000_000 + i * 1000,
        )
        for i in range(280)
    ]
    await bar_store.store_bars(bars)

    emitted: list[tuple[set[uuid.UUID], str]] = []

    def _spy(merged: dict[uuid.UUID, dict[str, float]], feature_set_version: str) -> None:
        emitted.append((set(merged), feature_set_version))

    monkeypatch.setattr(
        "quant_platform.services.data_service.maintenance.maintenance_scheduler."
        "emit_feature_distribution_metrics",
        _spy,
    )

    scheduler = _make_scheduler(
        instrument=instrument,
        bar_store=bar_store,
        universe=universe,
        feature_repo=feature_repo,
    )
    result = await scheduler.run_once(
        strategy_run_id=uuid.uuid4(),
        as_of=as_of,
        feature_set_version=PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    )

    assert result.features_stored == 1
    assert emitted == [({instrument.instrument_id}, PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION)]


@pytest.mark.asyncio
async def test_scheduler_routes_paper_alpha_catalyst_v10_feature_set_to_bar_history() -> None:
    instrument = Instrument(
        instrument_id=uuid.uuid4(),
        symbol="MSFT",
        exchange="XNAS",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        sector="Information Technology",
    )
    master = ContractMaster([instrument])
    bar_store = _InMemoryBarStore()
    universe = UniverseManager(master)
    feature_repo = InMemoryFeatureRepository()
    as_of = datetime(2026, 4, 10, 20, 0, tzinfo=_UTC)

    await feature_repo.store_vector(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=instrument.instrument_id,
            strategy_run_id=uuid.uuid4(),
            as_of=as_of - timedelta(days=5),
            available_at=as_of - timedelta(days=5),
            feature_set_version=TEXT_CATALYST_V5_EVENT_FEATURE_SET_VERSION,
            features={
                "operating_quality": 0.5,
                "text_sentiment": -0.5,
                "catalyst_sentiment": -0.4,
                "event_surprise": -0.3,
                "forward_outlook": 0.6,
                "margin_resilience": 0.4,
                "disclosure_specificity": 0.8,
                "risk_pressure": 0.1,
            },
        )
    )
    bars = [
        _make_bar(
            instrument.instrument_id,
            as_of - timedelta(days=280 - i),
            Decimal("100") + Decimal(i) * Decimal("0.2"),
            1_000_000 + i * 1000,
        )
        for i in range(280)
    ]
    await bar_store.store_bars(bars)

    scheduler = _make_scheduler(
        instrument=instrument,
        bar_store=bar_store,
        universe=universe,
        feature_repo=feature_repo,
    )
    result = await scheduler.run_once(
        strategy_run_id=uuid.uuid4(),
        as_of=as_of,
        feature_set_version=PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    )

    assert result.features_stored == 1
    features = result.feature_data[instrument.instrument_id]
    assert all(name in features for name in TEXT_CATALYST_V10_ALPHA_FEATURES)
    assert features["v10_stability_abs_text_specificity_event_surprise_21d"] >= 0.0
    stored = await feature_repo.get_vectors(
        [instrument.instrument_id],
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
        as_of,
    )
    assert len(stored) == 1

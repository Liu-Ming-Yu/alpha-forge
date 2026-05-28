"""build_ingest_bar_fetcher wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.config import DataIngestSettings, PlatformSettings
from quant_platform.services.data_service.feeds.failover_bar_fetcher import FailoverBarFetcher
from quant_platform.services.data_service.feeds.ib_bar_fetcher import IBBarFetcher
from quant_platform.services.data_service.feeds.ingest_bar_fetcher_factory import (
    build_ingest_bar_fetcher,
    build_vendor_bar_fetcher,
)
from quant_platform.services.data_service.feeds.polygon_daily_bar_fetcher import (
    PolygonDailyBarFetcher,
)
from quant_platform.services.data_service.feeds.tiingo_bar_fetcher import TiingoBarFetcher

if TYPE_CHECKING:
    from datetime import date

    from quant_platform.core.domain.market_data import MarketBar


class _StubBroker:
    async def fetch_historical_bars(
        self,
        instrument_id: object,
        bar_seconds: int,
        end_date: date,
        duration: str = "1 D",
        what_to_show: str = "TRADES",
    ) -> list[MarketBar]:
        return []


def test_factory_ib_only() -> None:
    s = PlatformSettings(_env_file=None)
    out = build_ingest_bar_fetcher(s, _StubBroker())
    assert isinstance(out, IBBarFetcher)


def test_factory_tiingo_failover() -> None:
    s = PlatformSettings(
        _env_file=None,
        data_ingest=DataIngestSettings(
            bar_fetch_fallback="tiingo",
            tiingo_api_token="abc",  # pragma: allowlist secret
        ),
    )
    out = build_ingest_bar_fetcher(s, _StubBroker())
    assert isinstance(out, FailoverBarFetcher)


def test_factory_no_historical_returns_none() -> None:
    s = PlatformSettings(_env_file=None)
    out = build_ingest_bar_fetcher(s, object())
    assert out is None


def test_vendor_fetcher_none_when_unconfigured() -> None:
    s = PlatformSettings(_env_file=None)
    assert build_vendor_bar_fetcher(s) is None


def test_vendor_fetcher_single_vendor_no_failover() -> None:
    s = PlatformSettings(
        _env_file=None,
        data_ingest=DataIngestSettings(
            bar_fetch_fallback_chain=["polygon"],
            polygon_api_key="pk",  # pragma: allowlist secret
        ),
    )
    out = build_vendor_bar_fetcher(s)
    assert isinstance(out, PolygonDailyBarFetcher)


def test_vendor_fetcher_multi_vendor_failover() -> None:
    s = PlatformSettings(
        _env_file=None,
        data_ingest=DataIngestSettings(
            bar_fetch_fallback_chain=["tiingo", "polygon"],
            tiingo_api_token="tk",  # pragma: allowlist secret
            polygon_api_key="pk",  # pragma: allowlist secret
        ),
    )
    out = build_vendor_bar_fetcher(s)
    assert isinstance(out, FailoverBarFetcher)


def test_vendor_fetcher_single_fallback_field() -> None:
    s = PlatformSettings(
        _env_file=None,
        data_ingest=DataIngestSettings(
            bar_fetch_fallback="tiingo",
            tiingo_api_token="tk",  # pragma: allowlist secret
        ),
    )
    out = build_vendor_bar_fetcher(s)
    assert isinstance(out, TiingoBarFetcher)

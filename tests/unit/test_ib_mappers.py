"""Unit tests for IBKR adapter mapping helpers."""

from __future__ import annotations

import sys
import threading
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeContract:
    def __init__(self) -> None:
        self.conId = 0
        self.symbol = ""
        self.exchange = ""
        self.currency = "USD"
        self.secType = "STK"
        self.primaryExchange = ""


class _FakeOrder:
    def __init__(self) -> None:
        self.action = ""
        self.totalQuantity = 0
        self.orderRef = ""
        self.transmit = False
        self.eTradeOnly = True
        self.firmQuoteOnly = True
        self.orderType = ""
        self.lmtPrice = 0.0
        self.tif = ""


class _FakeSyncWrapper:
    def __init__(self) -> None:
        self._account_values: dict[str, str] = {}
        self._account_done: Any = None
        self._account_active_req_id: int | None = None
        self._positions: list[Any] = []
        self._positions_done: Any = None
        self._positions_generation = 0
        self._positions_expected_generation = 0


class _FakeAccountClient:
    def __init__(self, wrapper: _FakeSyncWrapper) -> None:
        self.wrapper = wrapper
        self.account_cancelled: list[int] = []

    def reqAccountSummary(self, req_id: int, group: str, tags: str) -> None:
        self.wrapper._account_done.set_result(
            {
                "NetLiquidation": "1000",
                "SettledCash": "900",
            }
        )

    def cancelAccountSummary(self, req_id: int) -> None:
        self.account_cancelled.append(req_id)


class _FakePositionClient:
    def __init__(self, wrapper: _FakeSyncWrapper, rows: list[Any]) -> None:
        self.wrapper = wrapper
        self.rows = rows
        self.positions_cancelled = 0

    def reqPositions(self) -> None:
        self.wrapper._positions_done.set_result(self.rows)

    def cancelPositions(self) -> None:
        self.positions_cancelled += 1


class _FakeHistoricalWrapper:
    def __init__(self) -> None:
        self._lifecycle_lock = threading.Lock()
        self._hist_futures: dict[int, Any] = {}
        self._hist_data: dict[int, Any] = {}


class _FakeHistoricalClient:
    def __init__(self, wrapper: _FakeHistoricalWrapper, rows: list[Any] | None) -> None:
        self.wrapper = wrapper
        self.rows = rows
        self.requests: list[tuple[Any, ...]] = []

    def reqHistoricalData(
        self,
        req_id: int,
        contract: Any,
        end_date_time: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: int,
        format_date: int,
        keep_up_to_date: bool,
        chart_options: list[Any],
    ) -> None:
        self.requests.append(
            (
                req_id,
                contract,
                end_date_time,
                duration,
                bar_size,
                what_to_show,
                use_rth,
                format_date,
                keep_up_to_date,
                chart_options,
            )
        )
        if self.rows is not None:
            future = self.wrapper._hist_futures.pop(req_id)
            self.wrapper._hist_data.pop(req_id)
            future.set_result(self.rows)


class _FakeNewsWrapper:
    def __init__(self) -> None:
        self._lifecycle_lock = threading.Lock()
        self._historical_news: dict[int, Any] = {}
        self._historical_news_futures: dict[int, Any] = {}
        self._news_article_futures: dict[int, Any] = {}


class _FakeNewsClient:
    def __init__(self, wrapper: _FakeNewsWrapper) -> None:
        self.wrapper = wrapper
        self.historical_requests: list[tuple[Any, ...]] = []
        self.article_requests: list[tuple[Any, ...]] = []

    def reqHistoricalNews(
        self,
        req_id: int,
        con_id: int,
        provider_codes: str,
        start_date_time: str,
        end_date_time: str,
        total_results: int,
        historical_news_options: list[Any],
    ) -> None:
        self.historical_requests.append(
            (
                req_id,
                con_id,
                provider_codes,
                start_date_time,
                end_date_time,
                total_results,
                historical_news_options,
            )
        )
        future = self.wrapper._historical_news_futures.pop(req_id)
        self.wrapper._historical_news.pop(req_id)
        future.set_result(
            [
                ("2026-04-29 14:30:00", "BRFG", "BRFG$1", "AAPL headline"),
                ("2026-04-28 14:30:00", "BRFG", "BRFG$old", "Old headline"),
            ]
        )

    def reqNewsArticle(
        self,
        req_id: int,
        provider_code: str,
        article_id: str,
        news_article_options: list[Any],
    ) -> None:
        self.article_requests.append((req_id, provider_code, article_id, news_article_options))
        future = self.wrapper._news_article_futures.pop(req_id)
        future.set_result((0, "<p>Article body</p>"))


def _install_ibapi_stubs() -> None:
    try:
        __import__("ibapi.contract")
        __import__("ibapi.order")
        return
    except Exception:
        pass

    fake_contract_mod = MagicMock()
    fake_contract_mod.Contract = _FakeContract
    fake_order_mod = MagicMock()
    fake_order_mod.Order = _FakeOrder
    for name, mod in [
        ("ibapi", MagicMock()),
        ("ibapi.contract", fake_contract_mod),
        ("ibapi.order", fake_order_mod),
    ]:
        sys.modules.setdefault(name, mod)


_install_ibapi_stubs()

from quant_platform.core.domain.orders import (  # noqa: E402
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.lifecycle import (  # noqa: E402
    BrokerFillEvent,
    BrokerOrderCancelled,
    BrokerOrderCompleted,
    BrokerOrderRejected,
    BrokerUnmatchedFill,
)
from quant_platform.core.exceptions import BrokerSubmissionError  # noqa: E402
from quant_platform.services.execution_service.ib.ib_account_mapper import (  # noqa: E402
    account_snapshot_from_values,
    position_snapshot_from_values,
)
from quant_platform.services.execution_service.ib.ib_account_sync import (  # noqa: E402
    sync_account_snapshot,
    sync_position_snapshots,
)
from quant_platform.services.execution_service.ib.ib_contract_mapper import (  # noqa: E402
    build_con_id_mapping,
    contract_con_id,
    resolve_contract,
    validate_instrument_mappings,
)
from quant_platform.services.execution_service.ib.ib_historical_data_sync import (  # noqa: E402
    fetch_raw_historical_bars,
)
from quant_platform.services.execution_service.ib.ib_lifecycle_mapper import (  # noqa: E402
    broker_order_from_open_order,
    fill_event_from_pending,
    order_lifecycle_event_from_status,
    parse_execution_time,
    unmatched_fill_event_from_pending,
)
from quant_platform.services.execution_service.ib.ib_market_data_mapper import (  # noqa: E402
    bar_size_string,
    market_bar_from_raw,
    parse_bar_timestamp,
)
from quant_platform.services.execution_service.ib.ib_news import (  # noqa: E402
    IBNewsRuntime,
    parse_tws_news_timestamp,
)
from quant_platform.services.execution_service.ib.ib_news_sync import (  # noqa: E402
    fetch_raw_historical_news,
    fetch_raw_news_article,
)
from quant_platform.services.execution_service.ib.ib_order_mapper import (  # noqa: E402
    build_ib_order,
)


def test_contract_mapper_builds_forward_and_reverse_contract_identity() -> None:
    instrument_id = uuid.uuid4()
    contracts: dict[uuid.UUID, dict[str, Any]] = {
        instrument_id: {
            "symbol": "MSFT",
            "exchange": "SMART",
            "currency": "USD",
            "sec_type": "STK",
            "primary_exchange": "NASDAQ",
            "con_id": 12345,
        }
    }

    reverse = build_con_id_mapping(contracts)
    contract = resolve_contract(contracts, instrument_id)

    assert reverse == {12345: instrument_id}
    assert contract.symbol == "MSFT"
    assert contract.exchange == "SMART"
    assert contract.primaryExchange == "NASDAQ"
    assert contract_con_id(contract) == 12345
    assert validate_instrument_mappings(contracts) == []


def test_contract_mapper_fails_closed_for_bad_mapping() -> None:
    instrument_id = uuid.uuid4()

    with pytest.raises(BrokerSubmissionError, match="no IB contract mapping"):
        resolve_contract({}, instrument_id)

    with pytest.raises(BrokerSubmissionError, match="missing symbol"):
        resolve_contract({instrument_id: {"exchange": "SMART"}}, instrument_id)

    warnings = validate_instrument_mappings(
        {instrument_id: {"symbol": "AAPL", "exchange": "SMART"}}
    )
    assert "has no con_id" in warnings[0]


def test_order_mapper_translates_limit_order_and_smart_routing_flags() -> None:
    order = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("101.25"),
        time_in_force=TimeInForce.IOC,
        created_at=datetime.now(tz=UTC),
    )

    ib_order = build_ib_order(order)

    assert ib_order.action == "BUY"
    assert ib_order.totalQuantity == 10
    assert ib_order.orderRef == str(order.order_id)
    assert ib_order.transmit is True
    if hasattr(ib_order, "eTradeOnly"):
        assert ib_order.eTradeOnly is False
    if hasattr(ib_order, "firmQuoteOnly"):
        assert ib_order.firmQuoteOnly is False
    assert ib_order.orderType == "LMT"
    assert ib_order.lmtPrice == pytest.approx(101.25)
    assert ib_order.tif == "IOC"


def test_order_mapper_translates_sell_moc_order() -> None:
    order = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.SELL,
        quantity=5,
        order_type=OrderType.MOC,
        time_in_force=TimeInForce.DAY,
        created_at=datetime.now(tz=UTC),
    )

    ib_order = build_ib_order(order)

    assert ib_order.action == "SELL"
    assert ib_order.orderType == "MOC"
    assert ib_order.tif == "DAY"


def test_market_data_mapper_translates_bar_size_and_timestamps() -> None:
    assert bar_size_string(60) == "1 min"
    assert bar_size_string(86400) == "1 day"

    assert parse_bar_timestamp("20260102", 86400) == datetime(
        2026,
        1,
        2,
        tzinfo=UTC,
    )
    assert parse_bar_timestamp("20260102  15:30:00 US/Eastern", 60) == datetime(
        2026,
        1,
        2,
        15,
        30,
        tzinfo=UTC,
    )


def test_market_data_mapper_builds_domain_bar_from_raw_ib_tuple() -> None:
    instrument_id = uuid.uuid4()

    bar = market_bar_from_raw(
        instrument_id=instrument_id,
        bar_seconds=60,
        raw=("20260102  15:30:00 US/Eastern", 101.5, 100.5, 102.0, 101.0, -50),
    )

    assert bar.instrument_id == instrument_id
    assert bar.timestamp == datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    assert bar.open == Decimal("101.5")
    assert bar.high == Decimal("101.5")
    assert bar.low == Decimal("101.0")
    assert bar.close == Decimal("101.0")
    assert bar.volume == 0
    assert bar.is_complete is True


@pytest.mark.asyncio
async def test_historical_data_sync_resolves_raw_ib_bars() -> None:
    wrapper = _FakeHistoricalWrapper()
    contract = object()
    rows = [("20260102", 1.0, 2.0, 0.5, 1.5, 100)]
    client = _FakeHistoricalClient(wrapper, rows=rows)

    result = await fetch_raw_historical_bars(
        client=client,
        wrapper=wrapper,
        timeout=1,
        req_id=17,
        contract=contract,
        end_date_time="",
        duration="1 D",
        bar_size="1 day",
        what_to_show="TRADES",
    )

    assert result == rows
    assert client.requests == [(17, contract, "", "1 D", "1 day", "TRADES", 1, 1, False, [])]
    assert 17 not in wrapper._hist_futures
    assert 17 not in wrapper._hist_data


@pytest.mark.asyncio
async def test_historical_data_sync_cleans_up_on_timeout() -> None:
    wrapper = _FakeHistoricalWrapper()
    client = _FakeHistoricalClient(wrapper, rows=None)

    with pytest.raises(TimeoutError):
        await fetch_raw_historical_bars(
            client=client,
            wrapper=wrapper,
            timeout=0.001,
            req_id=18,
            contract=object(),
            end_date_time="",
            duration="1 D",
            bar_size="1 day",
            what_to_show="TRADES",
        )

    assert 18 not in wrapper._hist_futures
    assert 18 not in wrapper._hist_data


@pytest.mark.asyncio
async def test_news_sync_resolves_headlines_and_articles() -> None:
    wrapper = _FakeNewsWrapper()
    client = _FakeNewsClient(wrapper)

    headlines = await fetch_raw_historical_news(
        client=client,
        wrapper=wrapper,
        timeout=1,
        req_id=120,
        con_id=265598,
        provider_codes="BRFG",
        start_date_time="",
        end_date_time="2026-04-30 00:00:00",
        total_results=10,
    )
    article = await fetch_raw_news_article(
        client=client,
        wrapper=wrapper,
        timeout=1,
        req_id=121,
        provider_code="BRFG",
        article_id="BRFG$1",
    )

    assert headlines[0] == ("2026-04-29 14:30:00", "BRFG", "BRFG$1", "AAPL headline")
    assert article == (0, "<p>Article body</p>")
    assert client.historical_requests == [(120, 265598, "BRFG", "", "2026-04-30 00:00:00", 10, [])]
    assert client.article_requests == [(121, "BRFG", "BRFG$1", [])]


@pytest.mark.asyncio
async def test_ib_news_runtime_filters_window_and_fetches_article_body() -> None:
    wrapper = _FakeNewsWrapper()
    client = _FakeNewsClient(wrapper)
    instrument_id = uuid.uuid4()
    runtime = IBNewsRuntime(
        client=client,
        wrapper=wrapper,
        timeout=1,
        instrument_contracts={
            instrument_id: {
                "symbol": "AAPL",
                "con_id": 265598,
            }
        },
        require_connected=lambda: None,
    )

    articles = await runtime.fetch_historical_news(
        instrument_id=instrument_id,
        start=datetime(2026, 4, 29, tzinfo=UTC),
        end=datetime(2026, 4, 30, tzinfo=UTC),
        provider_codes=("BRFG", "DJNL"),
        total_results=25,
        include_article_text=True,
    )

    assert len(articles) == 1
    assert articles[0].instrument_id == instrument_id
    assert articles[0].provider_code == "BRFG"
    assert articles[0].article_text == "<p>Article body</p>"
    assert articles[0].article_status == "ready"
    assert client.historical_requests[0][2] == "BRFG+DJNL"
    # IB's reqHistoricalNews API has version-dependent quirks across
    # the (start, end) shape; the runtime now requests with both
    # bounds empty (= "give me the N most recent") and applies the
    # caller's window via a local filter. See ib_news.py for the
    # comment block explaining why.
    assert client.historical_requests[0][3] == ""
    assert client.historical_requests[0][4] == ""


def test_parse_tws_news_timestamp_accepts_common_formats() -> None:
    assert parse_tws_news_timestamp("2026-04-29 14:30:00") == datetime(
        2026,
        4,
        29,
        14,
        30,
        tzinfo=UTC,
    )
    assert parse_tws_news_timestamp("20260429 14:30:00") == datetime(
        2026,
        4,
        29,
        14,
        30,
        tzinfo=UTC,
    )


def test_account_mapper_builds_account_snapshot_from_ib_values() -> None:
    snapshot_id = uuid.uuid4()
    position = position_snapshot_from_values(
        snapshot_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        quantity=7,
        average_cost=Decimal("12.50"),
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    snapshot = account_snapshot_from_values(
        snapshot_id=snapshot_id,
        as_of=datetime(2026, 1, 2, 15, 30, tzinfo=UTC),
        values={
            "NetLiquidation": "12345.67",
            "TotalCashValue": "1000",
            "SettledCash": "900",
        },
        positions=[position],
    )

    assert snapshot.snapshot_id == snapshot_id
    assert snapshot.settled_cash == Decimal("900")
    assert snapshot.available_cash == Decimal("900")
    assert snapshot.net_asset_value == Decimal("12345.67")
    assert snapshot.positions == (position,)
    assert snapshot.source == "broker"


def test_position_mapper_builds_market_value_from_average_cost() -> None:
    instrument_id = uuid.uuid4()
    as_of = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)

    position = position_snapshot_from_values(
        snapshot_id=uuid.uuid4(),
        instrument_id=instrument_id,
        quantity=3,
        average_cost=Decimal("101.25"),
        as_of=as_of,
    )

    assert position.instrument_id == instrument_id
    assert position.quantity == 3
    assert position.average_cost == Decimal("101.25")
    assert position.market_price == Decimal("101.25")
    assert position.market_value == Decimal("303.75")
    assert position.unrealised_pnl == Decimal("0")
    assert position.as_of == as_of
    assert position.source == "broker"


@pytest.mark.asyncio
async def test_account_sync_builds_snapshot_and_cancels_summary_request() -> None:
    wrapper = _FakeSyncWrapper()
    client = _FakeAccountClient(wrapper)
    position = position_snapshot_from_values(
        snapshot_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        quantity=2,
        average_cost=Decimal("25"),
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    async def sync_positions() -> list[Any]:
        return [position]

    snapshot = await sync_account_snapshot(
        client=client,
        wrapper=wrapper,
        timeout=1,
        sync_positions=sync_positions,
    )

    assert snapshot.settled_cash == Decimal("900")
    assert snapshot.net_asset_value == Decimal("1000")
    assert snapshot.positions == (position,)
    assert client.account_cancelled == [9001]


@pytest.mark.asyncio
async def test_position_sync_maps_known_rows_and_skips_unsafe_rows() -> None:
    wrapper = _FakeSyncWrapper()
    known_instrument = uuid.uuid4()
    known_contract = _FakeContract()
    known_contract.conId = 11
    known_contract.symbol = "GOOD"
    unknown_contract = _FakeContract()
    unknown_contract.conId = 12
    unknown_contract.symbol = "UNKNOWN"
    bad_price_contract = _FakeContract()
    bad_price_contract.conId = 13
    bad_price_contract.symbol = "BADPX"
    client = _FakePositionClient(
        wrapper,
        rows=[
            ("DU1", known_contract, Decimal("3"), Decimal("101.25")),
            ("DU1", unknown_contract, Decimal("4"), Decimal("50")),
            ("DU1", bad_price_contract, Decimal("5"), Decimal("0")),
            ("DU1", known_contract, Decimal("-1"), Decimal("101.25")),
        ],
    )

    snapshots = await sync_position_snapshots(
        client=client,
        wrapper=wrapper,
        timeout=1,
        resolve_instrument_id=lambda con_id: known_instrument if con_id == 11 else None,
    )

    assert len(snapshots) == 1
    assert snapshots[0].instrument_id == known_instrument
    assert snapshots[0].quantity == 3
    assert snapshots[0].average_cost == Decimal("101.25")
    assert client.positions_cancelled == 1


def test_lifecycle_mapper_builds_open_order_projection() -> None:
    order_id = uuid.uuid4()
    observed_at = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)

    order = broker_order_from_open_order(
        order_ref=str(order_id),
        status="PreSubmitted",
        broker_order_id="42",
        observed_at=observed_at,
    )
    placeholder = broker_order_from_open_order(
        order_ref="",
        status="Inactive",
        broker_order_id="43",
        observed_at=observed_at,
    )

    assert order.order_id == order_id
    assert order.status == OrderStatus.SUBMITTED
    assert order.last_updated_at == observed_at
    assert order.broker_order_id == "42"
    assert placeholder.order_id != order_id
    assert placeholder.status == OrderStatus.REJECTED


def test_lifecycle_mapper_builds_terminal_order_events() -> None:
    order_id = uuid.uuid4()
    occurred_at = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)

    completed = order_lifecycle_event_from_status(
        order_id=order_id,
        broker_order_id="7",
        status="Filled",
        remaining=Decimal("0"),
        occurred_at=occurred_at,
    )
    cancelled = order_lifecycle_event_from_status(
        order_id=order_id,
        broker_order_id="7",
        status="Cancelled",
        remaining=Decimal("10"),
        occurred_at=occurred_at,
    )
    rejected = order_lifecycle_event_from_status(
        order_id=order_id,
        broker_order_id="7",
        status="Inactive",
        remaining=Decimal("10"),
        occurred_at=occurred_at,
    )
    non_terminal = order_lifecycle_event_from_status(
        order_id=order_id,
        broker_order_id="7",
        status="Submitted",
        remaining=Decimal("10"),
        occurred_at=occurred_at,
    )

    assert isinstance(completed, BrokerOrderCompleted)
    assert completed.occurred_at == occurred_at
    assert isinstance(cancelled, BrokerOrderCancelled)
    assert cancelled.reason == "broker cancelled"
    assert isinstance(rejected, BrokerOrderRejected)
    assert rejected.reason == "broker rejected (inactive)"
    assert non_terminal is None


def test_lifecycle_mapper_parses_execution_time_with_fallback() -> None:
    fallback = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)

    assert parse_execution_time("20260102  14:59:58", fallback=fallback) == datetime(
        2026,
        1,
        2,
        14,
        59,
        58,
        tzinfo=UTC,
    )
    assert parse_execution_time("bad timestamp", fallback=fallback) == fallback


def test_lifecycle_mapper_builds_fill_and_unmatched_fill_events() -> None:
    internal_id = uuid.uuid4()
    instrument_id = uuid.uuid4()
    executed_at = datetime(2026, 1, 2, 15, 29, tzinfo=UTC)
    received_at = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    pending = {
        "internal_id": internal_id,
        "ib_order_id": 77,
        "exec_id": "abc",
        "shares": 12,
        "price": Decimal("101.25"),
        "side": "BOT",
        "time": executed_at,
    }

    fill_event = fill_event_from_pending(
        pending=pending,
        instrument_id=instrument_id,
        commission=Decimal("1.23"),
        currency="USD",
        received_at=received_at,
        fill_id=uuid.UUID("00000000-0000-0000-0000-000000000077"),
    )
    unmatched = unmatched_fill_event_from_pending(
        pending=pending,
        con_id=12345,
        occurred_at=received_at,
    )

    assert isinstance(fill_event, BrokerFillEvent)
    assert fill_event.is_complete is False
    assert fill_event.fill.order_id == internal_id
    assert fill_event.fill.instrument_id == instrument_id
    assert fill_event.fill.side == OrderSide.BUY
    assert fill_event.fill.quantity == 12
    assert fill_event.fill.executed_at == executed_at
    assert fill_event.fill.received_at == received_at
    assert isinstance(unmatched, BrokerUnmatchedFill)
    assert unmatched.ib_order_id == 77
    assert unmatched.exec_id == "abc"
    assert unmatched.con_id == 12345

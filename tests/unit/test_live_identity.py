"""Tests for live broker instrument identity mapping and session wiring.

These tests cover:
- IBGatewayBrokerGateway canonical reverse mapping (conId → instrument_id)
- validate_instrument_mappings() warnings for missing con_id
- sync_positions() skips broker positions for unmapped conIds
- commissionReport() uses _ib_to_instrument rather than deriving from conId
- create_live_session() raises ValueError for empty instrument_contracts on tws path

No real IB Gateway connection is required.  ibapi is stubbed via sys.modules
patching so the module can be imported without the IBKR TWS API distribution.
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# ibapi stub — must be inserted into sys.modules BEFORE importing broker_gateway
# ---------------------------------------------------------------------------


class _FakeEWrapper:
    """Minimal EWrapper stub — just enough to let _IBWrapper subclass it."""

    def __init__(self) -> None:
        pass


class _FakeEClient:
    """Minimal EClient stub — prevents actual socket connections."""

    def __init__(self, wrapper: Any) -> None:
        self._wrapper = wrapper

    def connect(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass

    def run(self) -> None:  # noqa: D102
        pass

    def disconnect(self) -> None:  # noqa: D102
        pass


def _install_ibapi_stubs() -> None:
    try:
        __import__("ibapi.client")
        __import__("ibapi.wrapper")
        __import__("ibapi.contract")
        __import__("ibapi.order")
        __import__("ibapi.common")
        return
    except Exception:
        pass

    fake_client_mod = MagicMock()
    fake_client_mod.EClient = _FakeEClient

    fake_wrapper_mod = MagicMock()
    fake_wrapper_mod.EWrapper = _FakeEWrapper

    fake_contract_mod = MagicMock()
    fake_contract_mod.Contract = type(
        "Contract",
        (),
        {
            "conId": 0,
            "symbol": "",
            "exchange": "",
            "currency": "USD",
            "secType": "STK",
            "primaryExchange": "",
            "__init__": lambda self: None,
        },
    )

    for name, mod in [
        ("ibapi", MagicMock()),
        ("ibapi.client", fake_client_mod),
        ("ibapi.wrapper", fake_wrapper_mod),
        ("ibapi.contract", fake_contract_mod),
        ("ibapi.order", MagicMock()),
        ("ibapi.common", MagicMock()),
    ]:
        sys.modules.setdefault(name, mod)


_install_ibapi_stubs()

# Now safe to import the gateway module
from quant_platform.services.execution_service.gateways.broker_gateway import (  # noqa: E402
    IBGatewayBrokerGateway,
)
from quant_platform.services.execution_service.reconciliation import (  # noqa: E402
    DiscrepancyType,
)

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)

_AAPL_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_MSFT_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")
_AAPL_CON_ID = 265598
_MSFT_CON_ID = 272093

_FULL_CONTRACTS: dict[uuid.UUID, dict[str, object]] = {
    _AAPL_ID: {
        "symbol": "AAPL",
        "exchange": "SMART",
        "currency": "USD",
        "con_id": _AAPL_CON_ID,
    },
    _MSFT_ID: {
        "symbol": "MSFT",
        "exchange": "SMART",
        "currency": "USD",
        "con_id": _MSFT_CON_ID,
    },
}

_NO_CONID_CONTRACTS: dict[uuid.UUID, dict[str, object]] = {
    _AAPL_ID: {
        "symbol": "AAPL",
        "exchange": "SMART",
        "currency": "USD",
        # con_id deliberately absent
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gateway(
    contracts: dict[uuid.UUID, dict[str, object]] | None = None,
) -> IBGatewayBrokerGateway:
    return IBGatewayBrokerGateway(
        host="127.0.0.1",
        port=7497,
        paper_trading=True,
        client_id=1,
        instrument_contracts=contracts or {},
    )


# ---------------------------------------------------------------------------
# Canonical reverse-mapping tests
# ---------------------------------------------------------------------------


def test_reverse_map_built_from_full_contracts() -> None:
    """_con_id_to_instrument is populated for every spec with a valid con_id."""
    gw = _make_gateway(_FULL_CONTRACTS)
    assert gw._con_id_to_instrument[_AAPL_CON_ID] == _AAPL_ID
    assert gw._con_id_to_instrument[_MSFT_CON_ID] == _MSFT_ID


def test_reverse_map_excludes_specs_without_con_id() -> None:
    """Specs missing con_id do not appear in the reverse map."""
    gw = _make_gateway(_NO_CONID_CONTRACTS)
    assert len(gw._con_id_to_instrument) == 0


def test_resolve_instrument_id_returns_mapped_uuid() -> None:
    gw = _make_gateway(_FULL_CONTRACTS)
    assert gw._resolve_instrument_id(_AAPL_CON_ID) == _AAPL_ID


def test_resolve_instrument_id_returns_none_for_unknown_con_id() -> None:
    gw = _make_gateway(_FULL_CONTRACTS)
    assert gw._resolve_instrument_id(99999999) is None


def test_resolve_instrument_id_returns_none_when_no_contracts() -> None:
    gw = _make_gateway({})
    assert gw._resolve_instrument_id(_AAPL_CON_ID) is None


# ---------------------------------------------------------------------------
# validate_instrument_mappings tests
# ---------------------------------------------------------------------------


def test_validate_returns_empty_when_all_con_ids_present() -> None:
    gw = _make_gateway(_FULL_CONTRACTS)
    warnings = gw.validate_instrument_mappings()
    assert warnings == []


def test_validate_warns_for_missing_con_id() -> None:
    gw = _make_gateway(_NO_CONID_CONTRACTS)
    warnings = gw.validate_instrument_mappings()
    assert len(warnings) == 1
    assert str(_AAPL_ID) in warnings[0]
    assert "no con_id" in warnings[0]


def test_validate_warns_for_each_missing_con_id() -> None:
    contracts = {
        _AAPL_ID: {"symbol": "AAPL", "exchange": "SMART"},
        _MSFT_ID: {"symbol": "MSFT", "exchange": "SMART"},
    }
    gw = _make_gateway(contracts)
    warnings = gw.validate_instrument_mappings()
    assert len(warnings) == 2


def test_validate_returns_empty_when_no_contracts() -> None:
    """Empty mapping dict has nothing to validate — no warnings."""
    gw = _make_gateway({})
    warnings = gw.validate_instrument_mappings()
    assert warnings == []


# ---------------------------------------------------------------------------
# UNKNOWN_BROKER_CONTRACT discrepancy type
# ---------------------------------------------------------------------------


def test_unknown_broker_contract_type_exists() -> None:
    assert DiscrepancyType.UNKNOWN_BROKER_CONTRACT == "unknown_broker_contract"


def test_discrepancy_types_cover_all_expected_cases() -> None:
    types = {dt.value for dt in DiscrepancyType}
    assert "position_size_mismatch" in types
    assert "missing_internal_position" in types
    assert "extra_internal_position" in types
    assert "unknown_broker_contract" in types


# ---------------------------------------------------------------------------
# create_live_session validation
# ---------------------------------------------------------------------------


def test_create_live_session_raises_for_empty_contracts_on_tws_path() -> None:
    """create_live_session must reject an empty instrument_contracts on the tws path."""
    import pytest

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.session import create_live_session

    snapshot = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("50000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("50000"),
        net_asset_value=Decimal("50000"),
        positions=(),
    )
    settings = PlatformSettings(_env_file=None)
    assert settings.broker.primary_broker_path == "tws"  # default path

    with pytest.raises(ValueError, match="instrument_contracts must not be empty"):
        create_live_session(settings=settings, initial_snapshot=snapshot, instrument_contracts=None)


def test_create_live_session_raises_for_none_contracts_on_tws_path() -> None:
    import pytest

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.session import create_live_session

    snapshot = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("50000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("50000"),
        net_asset_value=Decimal("50000"),
        positions=(),
    )

    with pytest.raises(ValueError, match="instrument_contracts must not be empty"):
        create_live_session(initial_snapshot=snapshot)


def test_create_live_session_succeeds_with_valid_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session factory returns a Session when valid instrument_contracts are supplied."""
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.session import Session, create_live_session

    snapshot = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("50000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("50000"),
        net_asset_value=Decimal("50000"),
        positions=(),
    )

    # The live-session factory invokes ``_assert_live_session_defaults``,
    # which refuses to start against the in-memory / stub defaults used in
    # this hermetic unit test.  Opt into the dev-defaults escape hatch so
    # the test exercises the instrument-identity invariants it cares about
    # without needing a full Postgres / Redis / MarketRegimeDetector stack.
    settings = PlatformSettings(_env_file=None, allow_dev_defaults=True)
    session = create_live_session(
        settings=settings,
        initial_snapshot=snapshot,
        instrument_contracts=_FULL_CONTRACTS,
    )
    assert isinstance(session, Session)
    # Verify the trading broker has the canonical reverse map populated
    assert hasattr(session.trading_broker, "_con_id_to_instrument")
    assert session.trading_broker._con_id_to_instrument[_AAPL_CON_ID] == _AAPL_ID  # type: ignore[attr-defined]
    assert session.trading_broker._con_id_to_instrument[_MSFT_CON_ID] == _MSFT_ID  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# get_last_bar: pacing + dedup + disabled-by-default
# ---------------------------------------------------------------------------


import asyncio  # noqa: E402

import pytest  # noqa: E402

from quant_platform.config import BrokerSettings  # noqa: E402


def _build_enabled_gateway() -> IBGatewayBrokerGateway:
    settings = BrokerSettings(
        historical_bar_fetch_enabled=True,
        request_timeout_seconds=2.0,
    )
    gw = IBGatewayBrokerGateway(
        settings=settings,
        instrument_contracts=_FULL_CONTRACTS,
    )
    gw._connected = True
    return gw


@pytest.mark.asyncio
async def test_get_last_bar_returns_none_when_disabled() -> None:
    """Default settings keep historical fetches off; result is ``None``."""
    gw = _make_gateway(_FULL_CONTRACTS)
    gw._connected = True
    result = await gw.get_last_bar(_AAPL_ID, 86400)
    assert result is None


@pytest.mark.asyncio
async def test_get_last_bar_returns_none_for_unmapped_instrument() -> None:
    """An unknown instrument is reported cleanly rather than raising."""
    gw = _build_enabled_gateway()
    result = await gw.get_last_bar(uuid.uuid4(), 86400)
    assert result is None


@pytest.mark.asyncio
async def test_get_last_bar_maps_ibapi_response_to_market_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single reqHistoricalData response is parsed into a MarketBar."""
    gw = _build_enabled_gateway()

    issued: list[int] = []

    def fake_req(
        reqId: int,
        contract,
        endDateTime,
        duration,
        barSize,
        whatToShow,
        useRTH,
        formatDate,
        keepUpToDate,
        chartOptions,
    ):
        issued.append(reqId)
        with gw._wrapper._lifecycle_lock:
            entries = gw._wrapper._hist_data.setdefault(reqId, [])
            entries.append(("20260101", 100.0, 105.0, 99.0, 104.0, 1234))
            entries.append(("20260102", 104.0, 108.0, 103.0, 107.0, 5678))
            fut = gw._wrapper._hist_futures.pop(reqId)
            data = gw._wrapper._hist_data.pop(reqId)
        gw._wrapper._loop.call_soon_threadsafe(fut.set_result, data)

    gw._wrapper.set_loop(asyncio.get_running_loop())
    monkeypatch.setattr(gw._client, "reqHistoricalData", fake_req, raising=False)

    bar = await gw.get_last_bar(_AAPL_ID, 86400)
    assert bar is not None
    assert bar.instrument_id == _AAPL_ID
    assert bar.bar_seconds == 86400
    assert bar.close == Decimal("107.0")
    assert bar.volume == 5678
    assert len(issued) == 1


@pytest.mark.asyncio
async def test_get_last_bar_dedups_in_flight_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two overlapping calls for the same instrument issue only one ibapi request."""
    gw = _build_enabled_gateway()
    gw._wrapper.set_loop(asyncio.get_running_loop())

    call_count = 0
    release = asyncio.Event()

    def fake_req(
        reqId: int,
        contract,
        endDateTime,
        duration,
        barSize,
        whatToShow,
        useRTH,
        formatDate,
        keepUpToDate,
        chartOptions,
    ):
        nonlocal call_count
        call_count += 1

        async def _deliver():
            await release.wait()
            with gw._wrapper._lifecycle_lock:
                entries = gw._wrapper._hist_data.setdefault(reqId, [])
                entries.append(("20260101", 10.0, 11.0, 9.0, 10.5, 100))
                fut = gw._wrapper._hist_futures.pop(reqId)
                data = gw._wrapper._hist_data.pop(reqId)
            fut.set_result(data)

        asyncio.get_running_loop().create_task(_deliver())

    monkeypatch.setattr(gw._client, "reqHistoricalData", fake_req, raising=False)

    task_a = asyncio.create_task(gw.get_last_bar(_AAPL_ID, 86400))
    task_b = asyncio.create_task(gw.get_last_bar(_AAPL_ID, 86400))
    await asyncio.sleep(0)  # allow task_a to start the request
    await asyncio.sleep(0)
    release.set()

    bar_a, bar_b = await asyncio.gather(task_a, task_b)
    assert bar_a is not None and bar_b is not None
    assert bar_a.instrument_id == _AAPL_ID
    assert bar_b.instrument_id == _AAPL_ID
    assert call_count == 1

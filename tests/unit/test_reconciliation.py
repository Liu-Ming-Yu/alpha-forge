"""Unit tests for ReconciliationEngine.

Covers:
- One ReconciliationCompleted event returned per call (not one per discrepancy)
- No discrepancies → clean result
- Auto-corrected small mismatch
- Operator-action-required large mismatch
- Missing internal position → auto-corrected
- Extra internal position → operator action required
- Audit records written per discrepancy (not per cycle summary)
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from quant_platform.core.contracts import BrokerHealth, BrokerHealthStatus
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.core.events import DomainEvent, ReconciliationCompleted
from quant_platform.services.execution_service.reconciliation import (
    DiscrepancyResolution,
    DiscrepancyType,
    ReconciliationEngine,
)

if TYPE_CHECKING:
    from quant_platform.core.domain.orders import BrokerOrder

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_INSTRUMENT_A = uuid.uuid4()
_INSTRUMENT_B = uuid.uuid4()


class _FixedClock:
    def now(self) -> datetime:
        return _NOW

    def today(self) -> date:
        return _NOW.date()


def _pos(
    instrument_id: uuid.UUID, quantity: int, price: Decimal = Decimal("100")
) -> PositionSnapshot:
    return PositionSnapshot(
        snapshot_id=uuid.uuid4(),
        instrument_id=instrument_id,
        quantity=quantity,
        average_cost=price,
        market_price=price,
        market_value=Decimal(str(quantity)) * price,
        unrealised_pnl=Decimal("0"),
        as_of=_NOW,
    )


def _account(positions: tuple[PositionSnapshot, ...] = ()) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("10000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("10000"),
        net_asset_value=Decimal("10000"),
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubBroker:
    def __init__(
        self,
        positions: list[PositionSnapshot] | None = None,
        account: AccountSnapshot | None = None,
    ) -> None:
        self._positions = positions or []
        self._account = account or _account()

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def health_check(self) -> BrokerHealth:
        return BrokerHealth(
            status=BrokerHealthStatus.CONNECTED,
            latency_ms=1,
            last_heartbeat_at=_NOW,
        )

    async def sync_account(self) -> AccountSnapshot:
        return self._account

    async def sync_positions(self) -> list[PositionSnapshot]:
        return self._positions

    async def place_order(self, order: Any) -> Any: ...
    async def cancel_order(self, broker_order_id: str) -> None: ...
    async def fetch_open_orders(self) -> list[BrokerOrder]:
        return []


class _StubPositionRepo:
    def __init__(self, snapshot: AccountSnapshot | None = None) -> None:
        self._snapshot = snapshot
        self.saved: list[AccountSnapshot] = []

    async def save_snapshot(self, snapshot: AccountSnapshot) -> None:
        self.saved.append(snapshot)

    async def get_latest_snapshot(self) -> AccountSnapshot | None:
        return self._snapshot

    async def get_snapshot_at(self, as_of: datetime) -> AccountSnapshot | None:
        return self._snapshot


class _StubAuditSink:
    def __init__(self) -> None:
        self.records: list[tuple[DomainEvent, dict[str, Any]]] = []

    async def record(self, event: DomainEvent, context: dict[str, Any]) -> None:
        self.records.append((event, context))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReconciliationEngine:
    @pytest.mark.asyncio
    async def test_no_discrepancies_clean_result(self) -> None:
        pos_a = _pos(_INSTRUMENT_A, 100)
        broker = _StubBroker(positions=[pos_a], account=_account((pos_a,)))
        repo = _StubPositionRepo(snapshot=_account((pos_a,)))
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        run_id = uuid.uuid4()
        result = await engine.reconcile(run_id)

        assert isinstance(result, ReconciliationCompleted)
        assert result.discrepancies_found == 0
        assert result.discrepancies_resolved == 0
        assert not result.requires_operator_action
        # No discrepancies → audit should not have been called
        assert len(audit.records) == 0

    @pytest.mark.asyncio
    async def test_returns_exactly_one_event(self) -> None:
        """reconcile() must return exactly one ReconciliationCompleted, not one per discrepancy."""
        pos_broker_a = _pos(_INSTRUMENT_A, 100)
        pos_broker_b = _pos(_INSTRUMENT_B, 50)
        # Internal state only has A; B is missing → one discrepancy
        broker = _StubBroker(
            positions=[pos_broker_a, pos_broker_b],
            account=_account((pos_broker_a, pos_broker_b)),
        )
        repo = _StubPositionRepo(snapshot=_account((pos_broker_a,)))
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        result = await engine.reconcile(uuid.uuid4())

        # Must be a single ReconciliationCompleted value, not a list
        assert isinstance(result, ReconciliationCompleted)
        assert result.discrepancies_found == 1

    @pytest.mark.asyncio
    async def test_missing_internal_position_auto_corrected(self) -> None:
        pos = _pos(_INSTRUMENT_A, 100)
        broker = _StubBroker(positions=[pos], account=_account((pos,)))
        repo = _StubPositionRepo(snapshot=_account(()))  # internal has no positions
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        result = await engine.reconcile(uuid.uuid4())

        assert result.discrepancies_found == 1
        assert result.discrepancies_resolved == 1
        assert not result.requires_operator_action
        # One audit entry for the one discrepancy
        assert len(audit.records) == 1
        _, ctx = audit.records[0]
        assert ctx["type"] == DiscrepancyType.MISSING_INTERNAL_POSITION

    @pytest.mark.asyncio
    async def test_large_mismatch_requires_operator_action(self) -> None:
        pos_broker = _pos(_INSTRUMENT_A, 100)
        pos_internal = _pos(_INSTRUMENT_A, 50)  # delta = 50 > threshold
        broker = _StubBroker(positions=[pos_broker], account=_account((pos_broker,)))
        repo = _StubPositionRepo(snapshot=_account((pos_internal,)))
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        result = await engine.reconcile(uuid.uuid4())

        assert result.requires_operator_action
        assert result.discrepancies_found == 1
        assert result.discrepancies_resolved == 0
        _, ctx = audit.records[0]
        assert ctx["resolution"] == DiscrepancyResolution.OPERATOR_ACTION_REQUIRED

    @pytest.mark.asyncio
    async def test_small_mismatch_auto_corrected(self) -> None:
        pos_broker = _pos(_INSTRUMENT_A, 100)
        pos_internal = _pos(_INSTRUMENT_A, 99)  # delta = 1 ≤ threshold
        broker = _StubBroker(positions=[pos_broker], account=_account((pos_broker,)))
        repo = _StubPositionRepo(snapshot=_account((pos_internal,)))
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        result = await engine.reconcile(uuid.uuid4())

        assert not result.requires_operator_action
        assert result.discrepancies_resolved == 1
        _, ctx = audit.records[0]
        assert ctx["resolution"] == DiscrepancyResolution.AUTO_CORRECTED

    @pytest.mark.asyncio
    async def test_extra_internal_position_requires_operator_action(self) -> None:
        pos_internal = _pos(_INSTRUMENT_A, 100)
        broker = _StubBroker(positions=[], account=_account(()))  # broker has nothing
        repo = _StubPositionRepo(snapshot=_account((pos_internal,)))
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        result = await engine.reconcile(uuid.uuid4())

        assert result.requires_operator_action
        _, ctx = audit.records[0]
        assert ctx["type"] == DiscrepancyType.EXTRA_INTERNAL_POSITION

    @pytest.mark.asyncio
    async def test_broker_snapshot_persisted(self) -> None:
        """The broker-authoritative snapshot must be persisted regardless of discrepancies."""
        broker = _StubBroker(positions=[], account=_account(()))
        repo = _StubPositionRepo(snapshot=None)
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        await engine.reconcile(uuid.uuid4())

        assert len(repo.saved) == 1


# ---------------------------------------------------------------------------
# Stream 1 — Cash drift detection and ledger sync
# ---------------------------------------------------------------------------


class _StubLedger:
    """Minimal CashLedger stand-in for reconciliation tests."""

    def __init__(self, settled: Decimal) -> None:
        self.settled_cash = settled
        self.reset_calls: list[AccountSnapshot] = []

    def reset_from_snapshot(self, snapshot: AccountSnapshot) -> None:
        self.settled_cash = snapshot.settled_cash
        self.reset_calls.append(snapshot)


class TestCashDriftDetection:
    @pytest.mark.asyncio
    async def test_no_drift_returns_none_cash_drift_usd(self) -> None:
        """When ledger and broker agree within threshold, cash_drift_usd is near-zero."""
        broker_account = _account()  # settled_cash=10000
        broker = _StubBroker(positions=[], account=broker_account)
        repo = _StubPositionRepo(snapshot=None)
        audit = _StubAuditSink()
        ledger = _StubLedger(Decimal("10000"))  # same as broker

        engine = ReconciliationEngine(
            broker,
            repo,
            audit,
            _FixedClock(),
            ledger=ledger,
            cash_drift_threshold=Decimal("1.00"),
        )
        result = await engine.reconcile(uuid.uuid4())

        assert result.cash_drift_usd == Decimal("0")
        assert engine.last_cash_drift_event is None
        # Ledger should still be synced even on no-drift path.
        assert len(ledger.reset_calls) == 1

    @pytest.mark.asyncio
    async def test_drift_exceeds_threshold_resets_ledger(self) -> None:
        """When drift > threshold, ledger is reset and last_cash_drift_event is populated."""
        broker_account = _account()  # settled_cash=10000
        broker = _StubBroker(positions=[], account=broker_account)
        repo = _StubPositionRepo(snapshot=None)
        audit = _StubAuditSink()
        # Ledger thinks it has $10050 — $50 more than broker
        ledger = _StubLedger(Decimal("10050"))

        engine = ReconciliationEngine(
            broker,
            repo,
            audit,
            _FixedClock(),
            ledger=ledger,
            cash_drift_threshold=Decimal("1.00"),
        )
        result = await engine.reconcile(uuid.uuid4())

        assert result.cash_drift_usd == Decimal("50")
        assert engine.last_cash_drift_event is not None
        drift_ev = engine.last_cash_drift_event
        assert drift_ev.drift_usd == Decimal("50")
        assert drift_ev.ledger_settled_cash == Decimal("10050")
        assert drift_ev.broker_settled_cash == Decimal("10000")
        # Ledger must be reset to broker truth.
        assert ledger.settled_cash == Decimal("10000")
        # Audit must record the drift event.
        assert any(isinstance(ev, type(drift_ev)) for ev, _ in audit.records)

    @pytest.mark.asyncio
    async def test_drift_within_threshold_no_event(self) -> None:
        """Drift under threshold: no CashDriftDetected event, ledger still synced."""

        broker_account = _account()  # settled_cash=10000
        broker = _StubBroker(positions=[], account=broker_account)
        repo = _StubPositionRepo(snapshot=None)
        audit = _StubAuditSink()
        ledger = _StubLedger(Decimal("10000.50"))  # 50 cents — under $1 threshold

        engine = ReconciliationEngine(
            broker,
            repo,
            audit,
            _FixedClock(),
            ledger=ledger,
            cash_drift_threshold=Decimal("1.00"),
        )
        result = await engine.reconcile(uuid.uuid4())

        assert result.cash_drift_usd == Decimal("0.50")
        assert engine.last_cash_drift_event is None
        # Ledger should still be synced.
        assert ledger.settled_cash == Decimal("10000")

    @pytest.mark.asyncio
    async def test_no_ledger_cash_drift_usd_is_none(self) -> None:
        """Without a ledger, cash_drift_usd is None."""
        broker = _StubBroker(positions=[], account=_account())
        repo = _StubPositionRepo(snapshot=None)
        audit = _StubAuditSink()

        engine = ReconciliationEngine(broker, repo, audit, _FixedClock())
        result = await engine.reconcile(uuid.uuid4())

        assert result.cash_drift_usd is None
        assert engine.last_cash_drift_event is None

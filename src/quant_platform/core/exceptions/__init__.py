"""Contract-level exceptions raised across service boundaries.

These exceptions are defined in `core` so that any service adapter or
controller can raise or catch them without importing from a sibling service.

Hierarchy:
    QuantPlatformError
    ├── CashError
    │   ├── InsufficientCashError
    │   └── DuplicateReservationError
    ├── RiskError
    │   └── RiskLimitBreachedError
    ├── BrokerError
    │   ├── BrokerUnavailableError
    │   ├── BrokerAckTimeoutError
    │   ├── BrokerSubmissionError
    │   └── BrokerOrderNotFoundError
    ├── DataError
    │   └── DataStalenessError
    ├── ReconciliationError
    └── SettlementError
        └── PrematureSettlementError
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class QuantPlatformError(Exception):
    """Root exception for all platform-level errors."""


# ---------------------------------------------------------------------------
# Cash / reservation errors
# ---------------------------------------------------------------------------


class CashError(QuantPlatformError):
    """Base for cash accounting errors."""


class InsufficientCashError(CashError):
    """Raised when available settled cash is insufficient for the requested order.

    The message must include available_cash and required_cash so callers can
    build a TradeDecision without re-deriving the amounts.
    """


class DuplicateReservationError(CashError):
    """Raised when an ACTIVE reservation for the same order_id already exists.

    Signals a programming error: the caller should have checked for an existing
    reservation before calling reserve_cash().
    """


# ---------------------------------------------------------------------------
# Risk errors
# ---------------------------------------------------------------------------


class RiskError(QuantPlatformError):
    """Base for risk-policy errors."""


class RiskLimitBreachedError(RiskError):
    """Raised when a hard risk limit would be violated by the proposed action.

    The message must identify which limit was breached and the relevant values.
    """


# ---------------------------------------------------------------------------
# Broker errors
# ---------------------------------------------------------------------------


class BrokerError(QuantPlatformError):
    """Base for broker adapter errors."""


class BrokerUnavailableError(BrokerError):
    """Raised when the broker gateway is unreachable or not yet connected.

    Callers should implement exponential-backoff retry for this error.
    The kill switch must NOT be activated automatically on this error alone;
    wait for the configured retry budget to be exhausted first.
    """


class BrokerAckTimeoutError(BrokerUnavailableError):
    """Raised when an order was transmitted but no broker ack arrived in time.

    This is intentionally a BrokerUnavailableError subclass, but callers must
    treat it as an uncertain submission: the broker may still
    have a live order and local cash/order state must not be released until
    reconciliation resolves the broker state.
    """

    def __init__(
        self,
        message: str,
        *,
        order_id: object,
        broker_order_id: str,
    ) -> None:
        super().__init__(message)
        self.order_id = order_id
        self.broker_order_id = broker_order_id


class BrokerSubmissionError(BrokerError):
    """Raised on a non-retryable broker rejection of an order submission.

    Contrast with BrokerUnavailableError: this means the broker received the
    order and explicitly rejected it (wrong fields, no permission, etc.).
    The CashReservation for the rejected order must be released by the caller.
    """


class BrokerOrderNotFoundError(BrokerError):
    """Raised when cancel_order() targets an order unknown to the broker.

    The caller should verify the order's internal status before treating this
    as an anomaly — the order may have already been filled or expired.
    """


class InstrumentMappingError(BrokerError):
    """Raised when one or more instrument contracts lack a valid con_id.

    Live sessions require every mapped instrument to have a numeric con_id > 0
    so that broker-reported positions and fills can be matched to internal
    instrument_ids.  Sessions with incomplete mappings must not proceed.
    """


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------


class DataError(QuantPlatformError):
    """Base for data pipeline errors."""


class DataStalenessError(DataError):
    """Raised when cached market data exceeds the configured maximum age.

    Callers should abort signal generation and log at ERROR level so the
    operator is alerted.  The cycle must not submit orders based on stale prices.
    """

    def __init__(
        self,
        message: str,
        *,
        instrument_id: object | None = None,
        bar_timestamp: object | None = None,
        max_age_minutes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.instrument_id = instrument_id
        self.bar_timestamp = bar_timestamp
        self.max_age_minutes = max_age_minutes


# ---------------------------------------------------------------------------
# Reconciliation errors
# ---------------------------------------------------------------------------


class ReconciliationError(QuantPlatformError):
    """Raised when reconciliation cannot complete or produces an unresolvable state.

    When this is raised, the kill switch should be activated and no new orders
    should be submitted until an operator clears the condition.
    """


# ---------------------------------------------------------------------------
# Settlement errors
# ---------------------------------------------------------------------------


class SettlementError(QuantPlatformError):
    """Base for settlement calendar and lot errors."""


class PrematureSettlementError(SettlementError):
    """Raised when settle_lot() is called before the settlement date has arrived.

    Indicates either a clock error or a caller logic bug.
    """


class InfrastructureError(QuantPlatformError):
    """Base for infrastructure-layer errors."""


class DistributedLockError(InfrastructureError):
    """Raised when a distributed lock operation fails or the lease is lost.

    Callers using ``async with lock:`` must catch this in a try/except block.
    A lost lease means the critical section may have been entered by another
    worker — any mutations made during that section are suspect and the caller
    must treat its local state as potentially inconsistent.
    """


class ResearchError(QuantPlatformError):
    """Base for research / signal-generation errors."""


class FeatureValidationError(ResearchError):
    """Raised when a computed feature value is outside acceptable bounds.

    Prevents corrupted features from flowing into position-sizing or live
    order generation.  Callers must abort the cycle, not silently clamp.
    """


class LookAheadBiasError(ResearchError):
    """Raised when a feature is accessed before its declared available_at date.

    Indicates a look-ahead bias violation in backtest code; the backtest
    result must be discarded and the pipeline corrected.
    """


class ComplianceError(QuantPlatformError):
    """Base for pre-trade compliance violations."""


class ComplianceViolationError(ComplianceError):
    """Raised when a BLOCK-severity pre-trade compliance rule is triggered.

    Carries the list of violations so the caller can log them and choose
    whether to activate the kill switch.
    """

    def __init__(self, message: str, violations: Sequence[object] | None = None) -> None:
        super().__init__(message)
        self.violations: tuple[object, ...] = tuple(violations) if violations else ()

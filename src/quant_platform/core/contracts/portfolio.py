"""Portfolio / risk contracts: regime detection, construction, risk, cash.

These are the decision-side contracts that convert scored signals into
approved orders while respecting risk limits and the settled-cash
invariant of a cash account.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.contracts.common import TradeDecision
    from quant_platform.core.domain.orders import FillEvent, OrderIntent
    from quant_platform.core.domain.portfolio import (
        OptimizerResult,
        PortfolioRiskModel,
        PortfolioTarget,
        RiskLimits,
        RiskSnapshot,
    )
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.settlement import CashReservation, SettlementLot
    from quant_platform.core.domain.signals import RegimeLabel, RegimeState, SignalScore


@runtime_checkable
class RegimeDetector(Protocol):
    """Classify the current market environment into a RegimeState.

    Must never:
        Emit order instructions or adjust positions directly.
    """

    async def detect(self, as_of: datetime) -> RegimeState:
        """Return the current regime classification."""
        ...


@runtime_checkable
class PortfolioConstructor(Protocol):
    """Convert ranked signals into investable portfolio targets.

    Must never:
        Access the broker gateway or emit orders.
        Modify the input signals list.

    Research-to-production parity requirement:
        The same implementation is used in both backtest and live runs.
    """

    def build_targets(
        self,
        signals: list[SignalScore],
        regime: RegimeState,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> PortfolioTarget:
        """Return a PortfolioTarget satisfying all risk and cash constraints.

        Args:
            signals: Ranked cross-sectional scores from the signal service.
            regime: Current market regime used to select risk parameters.
            account: Current account state including settled cash and positions.
            limits: Hard risk constraints for the current session.

        Returns:
            A PortfolioTarget with weights that sum to <= 1.0 and respect all
            single-name, sector, and cash-buffer limits.

        Failure semantics:
            Raises PortfolioConstructionError if no valid target can be built
            (e.g. no settled cash, all names excluded by constraints).
        """
        ...


@runtime_checkable
class RegimeScaleProvider(Protocol):
    """Expose regime capital scaling without depending on a concrete constructor."""

    def scale_for_regime(self, label: RegimeLabel) -> Decimal:
        """Return the capital scale used for a regime label."""
        ...


@runtime_checkable
class RiskPolicy(Protocol):
    """Validate portfolio targets and individual orders against risk constraints.

    Two distinct entry points serve two distinct callers:
    - evaluate(): portfolio-level check called by BuildPortfolioController.
    - check_order_limits(): per-order check called by PreTradeGate before
      each individual order submission.

    Must never:
        Modify the target; only approve or reject it with an explanation.

    Research-to-production parity requirement:
        The same implementation is used in both backtest and live runs.
    """

    def evaluate(
        self,
        target: PortfolioTarget,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> TradeDecision:
        """Validate a full PortfolioTarget against portfolio-level risk limits.

        Args:
            target: The proposed portfolio target to evaluate.
            account: Current account state.
            limits: Hard risk constraints.

        Returns:
            TradeDecision.approved=True if all portfolio-level checks pass.
        """
        ...

    def check_order_limits(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> TradeDecision:
        """Validate a single OrderIntent against per-order risk limits.

        Called by PreTradeGate for each order before submission.  Checks a
        narrower set of limits than evaluate() — those that can be violated by
        a single order independent of portfolio composition, specifically:
        - single-name notional vs. NAV × max_single_name_weight

        Cash-buffer enforcement is intentionally excluded.  That responsibility
        belongs exclusively to CashConstraintEngine.can_open_order(), which
        reserves notional × (1 + buffer) and runs before this method in the
        PreTradeGate pipeline.

        Args:
            intent: The order to evaluate.
            account: Current account state including NAV.
            limits: Hard risk constraints.

        Returns:
            TradeDecision.approved=True if all per-order checks pass.
        """
        ...


@runtime_checkable
class CashConstraintEngine(Protocol):
    """Hard-gate tradability for a cash account using settled funds and reservations.

    This contract is the enforcement point for the most critical cash-account
    invariant: no order leaves the system without confirmed settled cash.

    Must never:
        Approve a buy order when available settled cash < required amount.
        Allow a reservation to persist after the order lifecycle terminates.
        Return stale account data; always operate on the snapshot passed in.

    Research-to-production parity requirement:
        The same implementation is used in both backtest and live runs.
    """

    @property
    def settled_cash(self) -> Decimal:
        """Total settled cash currently tracked by the engine."""
        ...

    def can_open_order(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
    ) -> TradeDecision:
        """Check whether settled cash is sufficient to fund this order.

        Args:
            intent: The proposed order to evaluate.
            account: Current account state, including reserved_cash.

        Returns:
            TradeDecision with approved=True if settled cash (minus existing
            reservations) is sufficient, else approved=False with reason.
        """
        ...

    def reserve_cash(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
    ) -> CashReservation:
        """Earmark settled cash for a pending order.

        Must only be called after can_open_order returns approved=True.
        The reservation must be released when the order terminates.

        Args:
            intent: The order for which cash is being reserved.
            account: Current account state.

        Returns:
            A new ACTIVE CashReservation reducing account.available_cash.

        Failure semantics:
            Raises InsufficientCashError if available cash dropped between the
            eligibility check and reservation creation (TOCTOU guard).
        """
        ...

    def release_reservation(
        self,
        reservation_id: uuid.UUID,
        reason: str,
    ) -> None:
        """Release a cash reservation, restoring available_cash.

        Must be called on order fill, cancellation, rejection, or expiry.
        Idempotent: calling on an already-released reservation is a no-op.

        Args:
            reservation_id: FK to CashReservation.reservation_id.
            reason: Human-readable release reason for the audit trail.
        """
        ...

    def project_settlement(
        self,
        fills: list[FillEvent],
    ) -> list[SettlementLot]:
        """Compute projected settlement lots from a list of sell fills.

        Args:
            fills: Sell FillEvents for which settlement should be projected.

        Returns:
            SettlementLots with trade_date and settlement_date populated
            using the SettlementCalendar.
        """
        ...

    def apply_fill(
        self,
        fill: FillEvent,
        is_order_complete: bool = False,
    ) -> None:
        """Apply a broker fill to the cash engine's in-memory state."""
        ...

    def cancel_order(self, order_id: uuid.UUID, reason: str) -> None:
        """Release order-linked reservations after cancellation or rejection."""
        ...

    def settle_lot(self, lot: SettlementLot) -> None:
        """Move a sell settlement lot into settled cash once eligible."""
        ...

    def reset_from_snapshot(self, snapshot: AccountSnapshot) -> None:
        """Reset state from a broker-authoritative account snapshot."""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Account-level optimizer for V2 central portfolio construction."""

    def optimize(
        self,
        target: PortfolioTarget,
        account: AccountSnapshot,
        limits: RiskLimits,
        risk_model: PortfolioRiskModel,
    ) -> OptimizerResult:
        """Return a target/risk snapshot that satisfies all hard constraints."""
        ...


@runtime_checkable
class PortfolioRiskModelRepository(Protocol):
    """Durable risk-model and risk-snapshot persistence."""

    async def latest_risk_model(self, *, as_of: datetime) -> PortfolioRiskModel | None:
        """Return the latest covariance/factor/stress model available at ``as_of``."""
        ...

    async def save_risk_snapshot(self, snapshot: RiskSnapshot) -> None:
        """Persist a risk snapshot produced by the optimizer/pretrade pass."""
        ...

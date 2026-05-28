"""Concrete portfolio-service controller implementations.

ApproveOrdersControllerImpl:
    For each OrderIntent, runs the PreTradeGate, creates a CashReservation for
    approved buys, and emits OrderApproved/OrderRejected events.

BuildPortfolioControllerImpl:
    Delegates portfolio construction to a PortfolioConstructor, validates
    the result with RiskPolicy, and emits PortfolioTargetBuilt on success.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import OrderSide
from quant_platform.core.events import (
    OrderApproved,
    OrderRejected,
    PortfolioTargetBuilt,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import (
        CashConstraintEngine,
        EventBus,
        OrderRepository,
        PortfolioConstructor,
        RiskPolicy,
    )
    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.signals import RegimeState, SignalScore
    from quant_platform.services.portfolio_service.pretrade_gate import PreTradeGate

log = structlog.get_logger(__name__)


class ApproveOrdersControllerImpl:
    """Run cash-gate and risk checks, reserving cash only for approved buys.

    Args:
        gate: PreTradeGate instance combining cash, risk, and execution checks.
        cash_engine: CashConstraintEngine for creating reservations.
        order_repo: OrderRepository for persisting intents.
        event_bus: EventBus for emitting approval/rejection events.
        limits: Active risk limits for this session.
    """

    def __init__(
        self,
        gate: PreTradeGate,
        cash_engine: CashConstraintEngine,
        order_repo: OrderRepository,
        event_bus: EventBus,
        limits: RiskLimits,
    ) -> None:
        self._gate = gate
        self._cash = cash_engine
        self._repo = order_repo
        self._bus = event_bus
        self._limits = limits

    async def approve(
        self,
        intents: list[OrderIntent],
        account: AccountSnapshot,
    ) -> tuple[list[OrderIntent], list[OrderIntent]]:
        """Run pre-trade checks and create buy-side reservations where needed.

        Returns:
            (approved_intents, rejected_intents). Approved buy intents include
            a cash reservation; approved sells keep ``cash_reservation_id=None``.
        """
        approved: list[OrderIntent] = []
        rejected: list[OrderIntent] = []
        now_ts = account.as_of

        for intent in intents:
            decision = self._gate.evaluate(intent, account, self._limits)

            if not decision.passed:
                rejected.append(intent)
                log.info(
                    "approve_orders.rejected",
                    order_id=str(intent.order_id),
                    reason=decision.reason,
                )
                await self._bus.publish(
                    OrderRejected(
                        event_id=uuid.uuid4(),
                        occurred_at=now_ts,
                        order_id=intent.order_id,
                        reason=decision.reason,
                    )
                )
                continue

            reservation_id: uuid.UUID | None = None
            if intent.side == OrderSide.BUY:
                try:
                    reservation = self._cash.reserve_cash(intent, account)
                except Exception as exc:
                    rejected.append(intent)
                    log.warning(
                        "approve_orders.reservation_failed",
                        order_id=str(intent.order_id),
                        side=intent.side.value,
                        error=str(exc),
                    )
                    await self._bus.publish(
                        OrderRejected(
                            event_id=uuid.uuid4(),
                            occurred_at=now_ts,
                            order_id=intent.order_id,
                            reason=str(exc),
                        )
                    )
                    continue
                reservation_id = reservation.reservation_id

            approved_intent = replace(intent, cash_reservation_id=reservation_id)
            approved.append(approved_intent)
            await self._repo.save_intent(approved_intent)
            await self._bus.publish(
                OrderApproved(
                    event_id=uuid.uuid4(),
                    occurred_at=now_ts,
                    order_id=intent.order_id,
                    reservation_id=reservation_id,
                )
            )
            log.info(
                "approve_orders.approved",
                order_id=str(intent.order_id),
                side=intent.side.value,
                reservation_id=str(reservation_id) if reservation_id is not None else None,
            )

        return approved, rejected


class BuildPortfolioControllerImpl:
    """Build and risk-validate a PortfolioTarget from signal scores.

    Orchestrates the portfolio construction pipeline:
    1. Calls PortfolioConstructor.build_targets() to produce candidate weights.
    2. Validates the result with RiskPolicy.evaluate().
    3. Emits PortfolioTargetBuilt on success; returns None on risk failure.

    Args:
        constructor: PortfolioConstructor that converts signals to weights.
        risk_policy: RiskPolicy for portfolio-level validation.
        event_bus: EventBus for PortfolioTargetBuilt events.
    """

    def __init__(
        self,
        constructor: PortfolioConstructor,
        risk_policy: RiskPolicy,
        event_bus: EventBus,
    ) -> None:
        self._constructor = constructor
        self._risk = risk_policy
        self._bus = event_bus

    async def build(
        self,
        signals: list[SignalScore],
        regime: RegimeState,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> PortfolioTarget | None:
        """Build and risk-validate a portfolio target.

        Args:
            signals: Ranked cross-sectional scores from the signal service.
            regime: Current market regime classification.
            account: Current account state used for risk validation.
            limits: Hard risk constraints for the current session.

        Returns:
            A validated PortfolioTarget, or None if the target fails risk
            policy checks.
        """
        target = self._constructor.build_targets(signals, regime, account, limits)

        decision = self._risk.evaluate(target, account, limits)
        if not decision.approved:
            log.warning(
                "portfolio.risk_rejected",
                reason=decision.reason,
                target_id=str(target.target_id),
            )
            return None

        await self._bus.publish(
            PortfolioTargetBuilt(
                event_id=uuid.uuid4(),
                occurred_at=account.as_of,
                target_id=target.target_id,
                strategy_run_id=target.strategy_run_id,
                regime_id=target.regime_id,
            )
        )
        log.info(
            "portfolio.target_built",
            target_id=str(target.target_id),
            n_names=len(target.weights),
            regime=regime.regime_label.value,
        )
        return target

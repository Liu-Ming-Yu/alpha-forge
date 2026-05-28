"""Pre-trade eligibility gate: the last line of defence before order submission.

The PreTradeGate aggregates the outputs of CashConstraintEngine, RiskPolicy,
an optional liquidity checker, and ExecutionPolicy to produce a single
go/no-go decision for each OrderIntent.

This is not the same as ApproveOrdersController (which orchestrates the
workflow).  The gate is a pure domain-layer check that can be called
synchronously in both backtest simulation and live execution.

Invariants:
- An order may only pass the gate if all configured checks pass.
- The gate never modifies the order; it only inspects it.
- The gate never creates or releases reservations; that is the caller's
  responsibility after a PASS decision.
- A PASS decision is valid only for the AccountSnapshot it was evaluated
  against.  Callers must not cache or reuse decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from quant_platform.core.contracts import (
    CashConstraintEngine,
    ExecutionPolicy,
    RiskPolicy,
    TradeDecision,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.core.domain.portfolio import RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


class GateOutcome(StrEnum):
    PASS = "pass"
    FAIL_CASH = "fail_cash"
    FAIL_RISK = "fail_risk"
    FAIL_LIQUIDITY = "fail_liquidity"
    FAIL_EXECUTION_POLICY = "fail_execution_policy"


@dataclass(frozen=True)
class GateDecision:
    """Result of a pre-trade gate evaluation.

    Args:
        outcome: One of the GateOutcome values.
        cash_decision: Result of the CashConstraintEngine check.
        risk_decision: Result of the RiskPolicy check.
        execution_decision: Result of the ExecutionPolicy check.
        reason: Human-readable summary of the first failing check,
            or "all checks passed" on PASS.
    """

    outcome: GateOutcome
    cash_decision: TradeDecision
    risk_decision: TradeDecision
    liquidity_decision: TradeDecision
    execution_decision: TradeDecision
    reason: str

    @property
    def passed(self) -> bool:
        """True if all three checks approved the order."""
        return self.outcome == GateOutcome.PASS


class PreTradeGate:
    """Synchronous pre-trade eligibility gate.

    Evaluates CashConstraintEngine → RiskPolicy → LiquidityChecker (optional)
    → ExecutionPolicy in order, short-circuiting on the first failure.

    Args:
        cash_engine: CashConstraintEngine implementation.
        risk_policy: RiskPolicy implementation.
        execution_policy: ExecutionPolicy implementation.
        liquidity_checker: Optional ADV/liquidity guard.

    Must never:
        Create cash reservations (caller does that after a PASS).
        Submit orders (caller does that after reservation is created).
        Be bypassed by any code path that submits orders.
    """

    def __init__(
        self,
        cash_engine: CashConstraintEngine,
        risk_policy: RiskPolicy,
        execution_policy: ExecutionPolicy,
        liquidity_checker: Callable[[OrderIntent, AccountSnapshot, RiskLimits], TradeDecision]
        | None = None,
    ) -> None:
        self._cash = cash_engine
        self._risk = risk_policy
        self._execution = execution_policy
        self._liquidity_checker = liquidity_checker

    def evaluate(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> GateDecision:
        """Run all pre-trade checks for a single OrderIntent.

        Checks are run in order: cash → risk → liquidity (optional) → execution policy.
        The first failure short-circuits the remaining checks, which are
        returned as approved=False with reason "not evaluated".

        Args:
            intent: The order to evaluate.
            account: Current account state (must be up-to-date).
            limits: Active risk limits for the current session.

        Returns:
            GateDecision with the aggregate outcome and individual check results.
        """
        not_evaluated = TradeDecision(
            approved=False,
            reason="not evaluated (prior check failed)",
            available_cash=Decimal("0"),
            required_cash=Decimal("0"),
        )

        # Step 1: cash check
        cash_decision = self._cash.can_open_order(intent, account)
        if not cash_decision.approved:
            return GateDecision(
                outcome=GateOutcome.FAIL_CASH,
                cash_decision=cash_decision,
                risk_decision=not_evaluated,
                liquidity_decision=not_evaluated,
                execution_decision=not_evaluated,
                reason=f"cash gate: {cash_decision.reason}",
            )

        # Step 2: per-order risk check (single-name notional, cash-buffer post-order).
        # Note: portfolio-level risk (sector weights, gross exposure) is checked
        # upstream by BuildPortfolioController via RiskPolicy.evaluate().  This
        # step enforces only the per-order limits that can be violated by a single
        # order independently of the rest of the portfolio.
        risk_decision = self._risk.check_order_limits(intent, account, limits)
        if not risk_decision.approved:
            return GateDecision(
                outcome=GateOutcome.FAIL_RISK,
                cash_decision=cash_decision,
                risk_decision=risk_decision,
                liquidity_decision=not_evaluated,
                execution_decision=not_evaluated,
                reason=f"risk limit: {risk_decision.reason}",
            )

        # Step 3: optional liquidity check (ADV participation)
        liquidity_decision = TradeDecision(
            approved=True,
            reason="not configured",
            available_cash=cash_decision.available_cash,
            required_cash=cash_decision.required_cash,
        )
        if self._liquidity_checker is not None:
            liquidity_decision = self._liquidity_checker(intent, account, limits)
            if not liquidity_decision.approved:
                return GateDecision(
                    outcome=GateOutcome.FAIL_LIQUIDITY,
                    cash_decision=cash_decision,
                    risk_decision=risk_decision,
                    liquidity_decision=liquidity_decision,
                    execution_decision=not_evaluated,
                    reason=f"liquidity gate: {liquidity_decision.reason}",
                )

        # Step 4: execution policy check (kill switch + throttle)
        execution_decision = self._execution.can_submit(intent)
        if not execution_decision.approved:
            return GateDecision(
                outcome=GateOutcome.FAIL_EXECUTION_POLICY,
                cash_decision=cash_decision,
                risk_decision=risk_decision,
                liquidity_decision=liquidity_decision,
                execution_decision=execution_decision,
                reason=f"execution policy: {execution_decision.reason}",
            )

        return GateDecision(
            outcome=GateOutcome.PASS,
            cash_decision=cash_decision,
            risk_decision=risk_decision,
            liquidity_decision=liquidity_decision,
            execution_decision=execution_decision,
            reason="all checks passed",
        )

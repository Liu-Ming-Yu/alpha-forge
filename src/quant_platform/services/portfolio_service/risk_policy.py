"""Standard risk policy implementation."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.contracts import TradeDecision
from quant_platform.services.portfolio_service.risk_policy_checks import (
    check_cash_buffer,
    check_etf_group_weights,
    check_gross_exposure,
    check_order_notional,
    check_sector_weights,
    check_single_name_weights,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot

log = structlog.get_logger(__name__)


class StandardRiskPolicy:
    """Concrete RiskPolicy implementation enforcing all RiskLimits constraints."""

    def __init__(
        self,
        sector_map: dict[uuid.UUID, str] | None = None,
        *,
        etf_groups: dict[str, set[uuid.UUID]] | None = None,
        etf_group_cap_multiplier: float = 1.5,
    ) -> None:
        self._sector_map: dict[uuid.UUID, str] = sector_map or {}
        self._etf_groups: dict[str, set[uuid.UUID]] = etf_groups or {}
        self._etf_group_cap_multiplier = Decimal(str(etf_group_cap_multiplier))

    def evaluate(
        self,
        target: PortfolioTarget,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> TradeDecision:
        """Validate a PortfolioTarget against all portfolio-level risk limits."""
        weights = dict(target.weights)

        if decision := check_gross_exposure(weights, account, limits):
            return decision

        if decision := check_single_name_weights(weights, account, limits):
            return decision

        if self._etf_groups:
            decision = check_etf_group_weights(
                weights,
                account,
                limits,
                etf_groups=self._etf_groups,
                cap_multiplier=self._etf_group_cap_multiplier,
            )
            if decision is not None:
                return decision

        if self._sector_map:
            sector_result = check_sector_weights(weights, account, limits, self._sector_map)
            if sector_result.missing_instrument_ids:
                log.warning(
                    "risk_policy.missing_sector",
                    instrument_ids=[str(i) for i in sector_result.missing_instrument_ids],
                    note="sector-weight check skipped for these instruments",
                )
            if sector_result.decision is not None:
                return sector_result.decision

        if decision := check_cash_buffer(target.cash_target_weight, account, limits):
            return decision

        return TradeDecision(
            approved=True,
            reason="all portfolio risk checks passed",
            available_cash=account.available_cash,
            required_cash=Decimal("0"),
        )

    def check_order_limits(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> TradeDecision:
        """Validate a single OrderIntent against per-order risk limits."""
        return check_order_notional(intent, account, limits)

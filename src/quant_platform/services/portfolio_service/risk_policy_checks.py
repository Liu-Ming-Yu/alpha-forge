"""Pure risk-policy checks used by ``StandardRiskPolicy``."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.contracts import TradeDecision
from quant_platform.core.domain.orders import OrderIntent, OrderSide

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from quant_platform.core.domain.portfolio import RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


@dataclass(frozen=True)
class SectorCheckResult:
    """Sector risk result plus instruments with missing sector metadata."""

    decision: TradeDecision | None
    missing_instrument_ids: tuple[uuid.UUID, ...]


def check_gross_exposure(
    weights: Mapping[uuid.UUID, Decimal],
    account: AccountSnapshot,
    limits: RiskLimits,
) -> TradeDecision | None:
    gross = sum(weights.values(), Decimal("0"))
    if gross <= limits.max_gross_exposure:
        return None
    return TradeDecision(
        approved=False,
        reason=f"gross exposure {gross:.4f} exceeds limit {limits.max_gross_exposure:.4f}",
        available_cash=account.available_cash,
        required_cash=Decimal("0"),
    )


def check_single_name_weights(
    weights: Mapping[uuid.UUID, Decimal],
    account: AccountSnapshot,
    limits: RiskLimits,
) -> TradeDecision | None:
    for instrument_id, weight in weights.items():
        if weight > limits.max_single_name_weight:
            return TradeDecision(
                approved=False,
                reason=(
                    f"single-name weight {weight:.4f} for {instrument_id} "
                    f"exceeds limit {limits.max_single_name_weight:.4f}"
                ),
                available_cash=account.available_cash,
                required_cash=Decimal("0"),
            )
    return None


def check_etf_group_weights(
    weights: Mapping[uuid.UUID, Decimal],
    account: AccountSnapshot,
    limits: RiskLimits,
    *,
    etf_groups: Mapping[str, set[uuid.UUID]],
    cap_multiplier: Decimal,
) -> TradeDecision | None:
    group_cap = limits.max_single_name_weight * cap_multiplier
    for group_name, group_ids in etf_groups.items():
        group_weight = sum(weights.get(iid, Decimal("0")) for iid in group_ids)
        if group_weight > group_cap:
            return TradeDecision(
                approved=False,
                reason=(
                    f"ETF correlation group '{group_name}' combined weight "
                    f"{group_weight:.4f} exceeds group cap {group_cap:.4f} "
                    f"({limits.max_single_name_weight:.4f} x {cap_multiplier:.2f})"
                ),
                available_cash=account.available_cash,
                required_cash=Decimal("0"),
            )
    return None


def check_sector_weights(
    weights: Mapping[uuid.UUID, Decimal],
    account: AccountSnapshot,
    limits: RiskLimits,
    sector_map: Mapping[uuid.UUID, str],
) -> SectorCheckResult:
    sector_weights: dict[str, Decimal] = {}
    missing_sectors: list[uuid.UUID] = []
    for instrument_id, weight in weights.items():
        sector = sector_map.get(instrument_id)
        if sector is None:
            missing_sectors.append(instrument_id)
        else:
            sector_weights[sector] = sector_weights.get(sector, Decimal("0")) + weight

    for sector, agg_weight in sector_weights.items():
        if agg_weight > limits.max_sector_weight:
            return SectorCheckResult(
                decision=TradeDecision(
                    approved=False,
                    reason=(
                        f"sector '{sector}' aggregate weight {agg_weight:.4f} "
                        f"exceeds limit {limits.max_sector_weight:.4f}"
                    ),
                    available_cash=account.available_cash,
                    required_cash=Decimal("0"),
                ),
                missing_instrument_ids=tuple(missing_sectors),
            )
    return SectorCheckResult(decision=None, missing_instrument_ids=tuple(missing_sectors))


def check_cash_buffer(
    cash_target_weight: Decimal,
    account: AccountSnapshot,
    limits: RiskLimits,
) -> TradeDecision | None:
    if cash_target_weight >= limits.min_cash_buffer:
        return None
    return TradeDecision(
        approved=False,
        reason=(
            f"cash target weight {cash_target_weight:.4f} is below "
            f"minimum cash buffer {limits.min_cash_buffer:.4f}"
        ),
        available_cash=account.available_cash,
        required_cash=Decimal("0"),
    )


def check_order_notional(
    intent: OrderIntent,
    account: AccountSnapshot,
    limits: RiskLimits,
) -> TradeDecision:
    if intent.side == OrderSide.SELL:
        return TradeDecision(
            approved=True,
            reason="sell orders skip per-order risk limits",
            available_cash=account.available_cash,
            required_cash=Decimal("0"),
        )

    price = intent.limit_price
    if price is None:
        pos = next(
            (p for p in account.positions if p.instrument_id == intent.instrument_id),
            None,
        )
        price = pos.market_price if pos else None

    notional = Decimal("0")
    if price is not None:
        notional = Decimal(str(intent.quantity)) * price
        max_allowed_notional = account.net_asset_value * limits.max_single_name_weight
        if notional > max_allowed_notional:
            return TradeDecision(
                approved=False,
                reason=(
                    f"order notional {notional:.2f} exceeds single-name limit "
                    f"{max_allowed_notional:.2f} "
                    f"(NAV={account.net_asset_value:.2f} x "
                    f"max_single_name_weight={limits.max_single_name_weight:.4f})"
                ),
                available_cash=account.available_cash,
                required_cash=notional,
            )

    return TradeDecision(
        approved=True,
        reason="per-order risk checks passed",
        available_cash=account.available_cash,
        required_cash=notional,
    )

"""Session component builders that do not require broker-specific wiring."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.contracts import TradeDecision
from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.services.data_service.reference.universe_manager import (
    LiquidityProfile,
    UniverseManager,
)
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.core.domain.portfolio import RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


def _spec_float(value: object, *, name: str) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"{name} must be numeric")


def build_default_portfolio_constructor(
    settings: PlatformSettings,
) -> LongOnlyPortfolioConstructor:
    """Build the default portfolio constructor from VolSizingSettings."""
    vol_settings = settings.vol_sizing
    if vol_settings.enabled:
        return VolTargetedPortfolioConstructor(
            vol_target=vol_settings.vol_target_annualized,
            min_vol_floor=vol_settings.min_vol_floor,
        )
    return LongOnlyPortfolioConstructor()


def build_contract_master(
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None,
) -> ContractMaster:
    instruments: list[Instrument] = []
    for instrument_id, spec in (instrument_contracts or {}).items():
        symbol = str(spec.get("symbol", f"INST_{str(instrument_id)[:8]}")).upper()
        exchange = str(spec.get("exchange", "SMART")).upper()
        currency = str(spec.get("currency", "USD")).upper()
        lot_size = int(str(spec.get("lot_size", 1)))
        sector_value = spec.get("sector")
        sector = str(sector_value) if sector_value else None
        instruments.append(
            Instrument(
                instrument_id=instrument_id,
                symbol=symbol,
                exchange=exchange,
                asset_class=_asset_class_from_contract_spec(spec),
                currency=currency,
                lot_size=lot_size,
                active=bool(spec.get("active", True)),
                sector=sector,
            )
        )
    return ContractMaster(instruments)


def seed_universe_liquidity(
    universe_manager: UniverseManager,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None,
    as_of: datetime,
) -> None:
    profiles: list[LiquidityProfile] = []
    for instrument_id, spec in (instrument_contracts or {}).items():
        adv_shares = spec.get("adv_shares_20d")
        last_close = spec.get("last_close")
        if adv_shares is None or last_close is None:
            continue
        adv_shares_f = _spec_float(adv_shares, name="adv_shares_20d")
        last_close_dec = Decimal(str(last_close))
        profiles.append(
            LiquidityProfile(
                instrument_id=instrument_id,
                adv_shares_20d=adv_shares_f,
                adv_usd_20d=adv_shares_f * float(last_close_dec),
                last_close=last_close_dec,
                computed_at=as_of,
            )
        )
    if profiles:
        universe_manager.update_liquidity(profiles)


def build_liquidity_checker(
    universe_manager: UniverseManager,
    *,
    allow_missing_profile: bool,
) -> Callable[[OrderIntent, AccountSnapshot, RiskLimits], TradeDecision]:
    def _check(
        intent: OrderIntent,
        _account: AccountSnapshot,
        _limits: RiskLimits,
    ) -> TradeDecision:
        ok, reason = universe_manager.check_participation(
            intent.instrument_id,
            intent.quantity,
            allow_missing_profile=allow_missing_profile,
        )
        return TradeDecision(
            approved=ok,
            reason=reason,
            available_cash=Decimal("0"),
            required_cash=Decimal("0"),
        )

    return _check


def _asset_class_from_contract_spec(spec: dict[str, object]) -> AssetClass:
    raw = str(spec.get("asset_class", "equity")).strip().lower()
    if raw == AssetClass.ETF.value:
        return AssetClass.ETF
    if raw == AssetClass.FUND.value:
        return AssetClass.FUND
    return AssetClass.EQUITY

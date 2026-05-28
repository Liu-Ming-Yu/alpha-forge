"""Universe management — liquid instrument selection and liquidity screening.

Filters the contract master down to a tradable universe based on:
- Minimum average daily volume (ADV) in USD
- Asset class (equity, ETF)
- Active status
- Sector availability for risk enforcement

Also provides an ADV participation check for pre-trade order sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from quant_platform.config import LiquiditySettings
from quant_platform.core.domain.instruments import AssetClass, Instrument

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.services.data_service.reference.contract_master import ContractMaster

log = structlog.get_logger(__name__)

GICS_SECTORS = frozenset(
    {
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
        "Utilities",
    }
)

ETF_ROTATION_UNIVERSE = frozenset(
    {
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
    }
)


@dataclass(frozen=True)
class LiquidityProfile:
    """Liquidity characteristics for one instrument.

    Args:
        instrument_id: FK to Instrument.
        adv_shares_20d: 20-day average daily volume in shares.
        adv_usd_20d: 20-day ADV in USD (shares x VWAP).
        last_close: Most recent close price.
        computed_at: When this profile was last computed.
    """

    instrument_id: uuid.UUID
    adv_shares_20d: float
    adv_usd_20d: float
    last_close: Decimal
    computed_at: datetime


class UniverseManager:
    """Filters and screens the tradable instrument universe.

    Args:
        contract_master: Source of registered instruments.
        settings: Liquidity configuration (ADV thresholds, participation caps).
    """

    def __init__(
        self,
        contract_master: ContractMaster,
        settings: LiquiditySettings | None = None,
    ) -> None:
        self._master = contract_master
        self._settings = settings or LiquiditySettings()
        self._profiles: dict[uuid.UUID, LiquidityProfile] = {}

    def update_liquidity(self, profiles: list[LiquidityProfile]) -> None:
        """Ingest fresh liquidity profiles (called after daily bar processing)."""
        for p in profiles:
            self._profiles[p.instrument_id] = p
        log.info("universe_manager.liquidity_updated", count=len(profiles))

    def get_tradable_universe(
        self,
        asset_classes: set[AssetClass] | None = None,
        require_sector: bool = True,
    ) -> list[Instrument]:
        """Return instruments passing all liquidity and data-quality screens.

        Filters applied:
        1. Active in contract master
        2. Matching asset class (if specified)
        3. ADV >= min_adv_usd
        4. Sector populated (if require_sector=True)
        """
        allowed_classes = asset_classes or {AssetClass.EQUITY, AssetClass.ETF}
        candidates = self._master.list_active()

        result: list[Instrument] = []
        for inst in candidates:
            if inst.asset_class not in allowed_classes:
                continue

            profile = self._profiles.get(inst.instrument_id)
            if profile is None:
                continue
            if profile.adv_usd_20d < self._settings.min_adv_usd:
                continue

            if require_sector and inst.sector is None:
                continue

            result.append(inst)

        log.info(
            "universe_manager.screened",
            total_active=len(candidates),
            passed=len(result),
        )
        return result

    def check_participation(
        self,
        instrument_id: uuid.UUID,
        order_shares: int,
        *,
        allow_missing_profile: bool = True,
    ) -> tuple[bool, str]:
        """Check whether order size respects the ADV participation cap.

        Returns (ok, reason).  ok=True if order_shares <= adv_shares_20d * participation_pct.
        """
        profile = self._profiles.get(instrument_id)
        if profile is None:
            if allow_missing_profile:
                return True, "liquidity profile unavailable; check skipped"
            return False, "no liquidity profile available"

        max_shares = int(profile.adv_shares_20d * self._settings.adv_participation_pct)
        if max_shares < 1:
            return False, f"ADV too low: {profile.adv_shares_20d:.0f} shares/day"

        if order_shares > max_shares:
            return False, (
                f"order {order_shares} shares exceeds "
                f"{self._settings.adv_participation_pct:.0%} of 20d ADV "
                f"({profile.adv_shares_20d:.0f} shares, max {max_shares})"
            )

        return True, "within participation limit"

    def get_profile(self, instrument_id: uuid.UUID) -> LiquidityProfile | None:
        """Return a stored liquidity profile, if one exists."""
        return self._profiles.get(instrument_id)

    def get_etf_rotation_instruments(self) -> list[Instrument]:
        """Return the fixed ETF rotation sleeve instruments."""
        result = []
        for symbol in ETF_ROTATION_UNIVERSE:
            inst = self._master.get_by_symbol(symbol)
            if inst is not None and inst.active:
                result.append(inst)
        return result

    def sector_map(self) -> dict[uuid.UUID, str]:
        """Delegate to contract master sector map for risk policy use."""
        return self._master.sector_map()

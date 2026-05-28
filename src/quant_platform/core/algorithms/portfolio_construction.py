"""Core portfolio construction implementations shared by live and research."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
from quant_platform.core.domain.signals import RegimeLabel, RegimeState, SignalScore

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot

log = structlog.get_logger(__name__)

DEFAULT_REGIME_SCALE = Decimal("0.75")
_DEFAULT_REGIME_SCALES: dict[RegimeLabel, Decimal] = {
    RegimeLabel.RISK_ON: Decimal("1.0"),
    RegimeLabel.UNKNOWN: DEFAULT_REGIME_SCALE,
    RegimeLabel.TRANSITION: DEFAULT_REGIME_SCALE,
    RegimeLabel.RISK_OFF: Decimal("0.5"),
    RegimeLabel.CRISIS: Decimal("0.0"),
}


class SimpleRegimeDetector:
    """Stub regime detector that always returns RISK_ON with full confidence."""

    _VERSION = "0.0.1-stub"
    is_simple_regime_detector = True

    async def detect(self, as_of: datetime) -> RegimeState:
        return RegimeState(
            regime_id=uuid.uuid4(),
            as_of=as_of,
            regime_label=RegimeLabel.RISK_ON,
            confidence=1.0,
            detector_version=self._VERSION,
            supporting_features={},
        )


class LongOnlyPortfolioConstructor:
    """Equal-weight, long-only portfolio constructor with regime scaling."""

    def __init__(
        self,
        top_n: int = 10,
        min_score_threshold: float = 0.0,
        sector_map: dict[uuid.UUID, str] | None = None,
        regime_scales: dict[RegimeLabel, Decimal] | None = None,
        no_trade_band: float = 0.0,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if no_trade_band < 0.0:
            raise ValueError("no_trade_band must be >= 0")
        self._top_n = top_n
        self._min_score = min_score_threshold
        self._sector_map = sector_map or {}
        self._scales = regime_scales if regime_scales is not None else dict(_DEFAULT_REGIME_SCALES)
        self._no_trade_band = no_trade_band

    def scale_for_regime(self, label: RegimeLabel) -> Decimal:
        """Return the capital scale used for a regime label."""
        return self._scales.get(label, DEFAULT_REGIME_SCALE)

    def build_targets(
        self,
        signals: list[SignalScore],
        regime: RegimeState,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> PortfolioTarget:
        """Build a long-only PortfolioTarget from cross-sectional signal scores."""
        notes: list[str] = []
        strategy_run_id = signals[0].strategy_run_id if signals else uuid.uuid4()
        scale = self.scale_for_regime(regime.regime_label)

        if scale == Decimal("0"):
            notes.append(f"regime={regime.regime_label.value}: fully de-risked to cash")
            log.info("portfolio.crisis_regime", regime=regime.regime_label.value)
            return self._cash_target(strategy_run_id, regime, notes)

        max_invest = min(limits.max_gross_exposure, Decimal("1") - limits.min_cash_buffer) * scale
        eligible = sorted(
            (s for s in signals if s.score > self._min_score),
            key=lambda s: s.score,
            reverse=True,
        )
        if not eligible:
            notes.append("no eligible signals above score threshold; holding cash")
            return self._cash_target(strategy_run_id, regime, notes)

        selected = eligible[: self._top_n]
        selected, weight_per_name = self._apply_name_cap(
            selected=selected,
            max_invest=max_invest,
            limits=limits,
            notes=notes,
        )
        if not selected:
            return self._cash_target(strategy_run_id, regime, notes)

        selected = self._apply_sector_cap(
            selected=selected,
            weight_per_name=weight_per_name,
            limits=limits,
            notes=notes,
        )
        weights = {s.instrument_id: weight_per_name for s in selected}
        if self._no_trade_band > 0.0:
            weights = self._apply_no_trade_band(weights, account, notes)
        invested = sum(weights.values(), Decimal("0"))
        cash_target = Decimal("1") - invested

        log.info(
            "portfolio.built",
            n_names=len(weights),
            regime=regime.regime_label.value,
            regime_scale=str(scale),
            weight_per_name=str(weight_per_name),
            invested=str(invested),
            cash_target=str(cash_target),
        )
        return PortfolioTarget(
            target_id=uuid.uuid4(),
            strategy_run_id=strategy_run_id,
            as_of=regime.as_of,
            regime_id=regime.regime_id,
            weights=weights,
            cash_target_weight=cash_target,
            construction_notes=notes,
        )

    def _apply_name_cap(
        self,
        *,
        selected: list[SignalScore],
        max_invest: Decimal,
        limits: RiskLimits,
        notes: list[str],
    ) -> tuple[list[SignalScore], Decimal]:
        weight_per_name = max_invest / Decimal(str(len(selected)))
        if weight_per_name <= limits.max_single_name_weight:
            return selected, weight_per_name

        n_max = int(max_invest / limits.max_single_name_weight)
        if n_max < 1:
            notes.append("max_single_name_weight too restrictive given max_invest; holding cash")
            return [], Decimal("0")

        notes.append(
            f"single-name cap active: reduced to {n_max} names at weight "
            f"{limits.max_single_name_weight}"
        )
        return selected[:n_max], limits.max_single_name_weight

    def _apply_no_trade_band(
        self,
        weights: dict[uuid.UUID, Decimal],
        account: AccountSnapshot,
        notes: list[str],
    ) -> dict[uuid.UUID, Decimal]:
        """Hold current weights when the desired change is below the band.

        Cost-aware hysteresis mirroring the campaign evaluation path, so live
        turnover matches what the research eligibility gate measured.  Names
        whose target differs from the current weight by less than the band are
        left untouched; small score wiggles no longer generate trades.
        """
        nav = account.net_asset_value
        if nav <= 0:
            return weights
        current = {pos.instrument_id: pos.market_value / nav for pos in account.positions}
        band = Decimal(str(self._no_trade_band))
        held: dict[uuid.UUID, Decimal] = {}
        for instrument_id in set(weights) | set(current):
            target_weight = weights.get(instrument_id, Decimal("0"))
            current_weight = current.get(instrument_id, Decimal("0"))
            if abs(target_weight - current_weight) < band:
                if current_weight > 0:
                    held[instrument_id] = current_weight
            elif target_weight > 0:
                held[instrument_id] = target_weight
        carried = sum(1 for instrument_id in held if instrument_id not in weights)
        if carried:
            notes.append(f"no-trade band carried {carried} sub-band position(s)")
        return held

    def _apply_sector_cap(
        self,
        *,
        selected: list[SignalScore],
        weight_per_name: Decimal,
        limits: RiskLimits,
        notes: list[str],
    ) -> list[SignalScore]:
        if not self._sector_map:
            return selected

        sector_alloc: dict[str, Decimal] = {}
        kept: list[SignalScore] = []
        for score in selected:
            sector = self._sector_map.get(score.instrument_id, "UNKNOWN")
            already = sector_alloc.get(sector, Decimal("0"))
            if already + weight_per_name > limits.max_sector_weight:
                notes.append(f"sector '{sector}' cap reached; excluded {score.instrument_id}")
                continue
            sector_alloc[sector] = already + weight_per_name
            kept.append(score)
        return kept

    def _cash_target(
        self,
        strategy_run_id: uuid.UUID,
        regime: RegimeState,
        notes: list[str],
    ) -> PortfolioTarget:
        return PortfolioTarget(
            target_id=uuid.uuid4(),
            strategy_run_id=strategy_run_id,
            as_of=regime.as_of,
            regime_id=regime.regime_id,
            weights={},
            cash_target_weight=Decimal("1"),
            construction_notes=notes,
        )


__all__ = [
    "DEFAULT_REGIME_SCALE",
    "LongOnlyPortfolioConstructor",
    "SimpleRegimeDetector",
]

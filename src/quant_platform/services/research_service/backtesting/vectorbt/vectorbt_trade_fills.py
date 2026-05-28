"""VectorBT backtest fill-artifact builders."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from ..artifacts.backtest_artifacts import (
    BacktestFillArtifact,
)

if TYPE_CHECKING:
    from datetime import datetime


def participation_pct_for_shares(shares: int, adv_shares: float) -> float:
    return min((shares / adv_shares) * 100.0 if adv_shares > 0 else 0.0, 100.0)


def build_vectorbt_fill_artifact(
    *,
    ts: datetime,
    instrument_id: uuid.UUID,
    side: str,
    shares: int,
    price: Decimal,
    slippage_frac: Decimal,
    commission: Decimal,
    adv_shares: float,
    spread_bps: float,
    slippage_bps: float,
) -> BacktestFillArtifact:
    adjusted_price = (
        price * (Decimal("1") + slippage_frac)
        if side == "buy"
        else price * (Decimal("1") - slippage_frac)
    )
    return BacktestFillArtifact(
        cycle_ts=ts,
        order_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=side,
        quantity=shares,
        requested_quantity=shares,
        filled_quantity=shares,
        fill_ratio=1.0,
        raw_fill_price=price,
        adjusted_fill_price=adjusted_price,
        commission=commission,
        adv_shares_20d=adv_shares,
        participation_pct=participation_pct_for_shares(shares, adv_shares),
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        slippage_cost=Decimal(shares) * price * slippage_frac,
        implementation_shortfall_bps=slippage_bps,
        is_complete=True,
    )


__all__ = ["build_vectorbt_fill_artifact", "participation_pct_for_shares"]

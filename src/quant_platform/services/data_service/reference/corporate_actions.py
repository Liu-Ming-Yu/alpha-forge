"""Shared read-time corporate-action adjustment engine.

Both ``InMemoryBarStore`` and ``ParquetBarStore`` must return corporate-
action-adjusted bars per the ``HistoricalDataStore`` contract.  Before
Phase 1.3 (parity and data completeness plan) the in-memory store returned
raw bars, causing tests that used it to silently disagree with production
behaviour.  This module hosts the single ``apply_adjustments`` function
that both stores use.

Supported actions
-----------------
SPLIT    — multiply pre-ex_date open/high/low/close/vwap by (1/ratio)
            and multiply pre-ex_date volume by ratio.
DIVIDEND — subtract ``cash_amount`` from pre-ex_date open/high/low/close
            (and vwap if present).  The adjusted bar is only emitted when
            the adjusted low remains strictly positive; otherwise the
            original bar is retained.

Adjustments are applied iteratively in reverse chronological order so that
composed events (split then split, split then dividend, …) compose cleanly.
Bars on or after ``ex_date`` are assumed to already be in post-action units
and are passed through unchanged; pre-ex_date bars are adjusted.
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from quant_platform.core.domain.instruments import (
    CorporateAction,
    CorporateActionType,
)
from quant_platform.core.domain.market_data import MarketBar

log = structlog.get_logger(__name__)

# Minimum allowed price tick after dividend adjustment.
_MIN_TICK = Decimal("0.01")


def apply_adjustments(
    bars: list[MarketBar],
    actions: list[CorporateAction],
) -> list[MarketBar]:
    """Apply corporate-action adjustments to a list of bars at read time.

    The input ``bars`` list is not mutated; a new list of adjusted
    ``MarketBar`` objects is returned.  Ordering of the output follows the
    input ordering (callers must sort afterwards if they require a
    deterministic order).

    Ordering invariant: for actions on the same ex_date, SPLIT adjustments
    are applied before DIVIDEND adjustments.  This matches the economic
    convention where shares are split first and the dividend is then
    subtracted from the post-split price.
    """
    adjusted = list(bars)
    # Sort: latest ex_date first; within the same date SPLIT before DIVIDEND.
    # "SPLIT" > "DIVIDEND" alphabetically, so reverse=True on the tuple
    # yields (ex_date DESC, action_type DESC) = SPLIT before DIVIDEND.
    for action in sorted(
        actions,
        key=lambda a: (a.ex_date, a.action_type.value),
        reverse=True,
    ):
        if action.action_type == CorporateActionType.SPLIT and action.ratio != Decimal("1"):
            inv_ratio = Decimal("1") / action.ratio
            new_bars: list[MarketBar] = []
            for bar in adjusted:
                if bar.timestamp.date() < action.ex_date:
                    bar = MarketBar(
                        bar_id=bar.bar_id,
                        instrument_id=bar.instrument_id,
                        timestamp=bar.timestamp,
                        bar_seconds=bar.bar_seconds,
                        open=bar.open * inv_ratio,
                        high=bar.high * inv_ratio,
                        low=bar.low * inv_ratio,
                        close=bar.close * inv_ratio,
                        volume=int(bar.volume * float(action.ratio)),
                        vwap=bar.vwap * inv_ratio if bar.vwap else None,
                        is_complete=bar.is_complete,
                    )
                new_bars.append(bar)
            adjusted = new_bars

        elif action.action_type == CorporateActionType.DIVIDEND and action.cash_amount > 0:
            new_bars = []
            for bar in adjusted:
                if bar.timestamp.date() < action.ex_date:
                    adj_close = bar.close - action.cash_amount
                    adj_open = bar.open - action.cash_amount
                    adj_high = bar.high - action.cash_amount
                    adj_low = bar.low - action.cash_amount
                    if adj_low <= 0:
                        # Clamp to minimum tick rather than dropping the bar.
                        # Dropping bars would introduce look-ahead bias.
                        log.warning(
                            "corporate_actions.dividend_clamped",
                            instrument_id=str(bar.instrument_id),
                            timestamp=bar.timestamp.isoformat(),
                            adj_low=str(adj_low),
                            cash_amount=str(action.cash_amount),
                            clamped_to=str(_MIN_TICK),
                        )
                        adj_low = _MIN_TICK
                        adj_open = max(adj_open, _MIN_TICK)
                        adj_high = max(adj_high, _MIN_TICK)
                        adj_close = max(adj_close, _MIN_TICK)
                    bar = MarketBar(
                        bar_id=bar.bar_id,
                        instrument_id=bar.instrument_id,
                        timestamp=bar.timestamp,
                        bar_seconds=bar.bar_seconds,
                        open=adj_open,
                        high=adj_high,
                        low=adj_low,
                        close=adj_close,
                        volume=bar.volume,
                        vwap=max(bar.vwap - action.cash_amount, _MIN_TICK) if bar.vwap else None,
                        is_complete=bar.is_complete,
                    )
                new_bars.append(bar)
            adjusted = new_bars

    return adjusted

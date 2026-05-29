"""Path-dependent long-only campaign portfolio evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.portfolio.costs import LinearTurnoverCost
from quant_platform.services.research_service.campaigns.portfolio.targets import (
    apply_no_trade_band,
    apply_position_change_cap,
    apply_turnover_cap,
    clean_weights,
    enforce_weight_limits,
    raw_long_only_target,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
    PortfolioEvaluation,
)
from quant_platform.services.research_service.reports.statistics import (
    spearman_ic as _spearman,
)
from quant_platform.services.research_service.sampling.samples import has_realized_returns

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.services.research_service.campaigns.portfolio.costs import TradingCostModel
    from quant_platform.services.research_service.campaigns.portfolio.selection import (
        SelectionStrategy,
    )
    from quant_platform.services.research_service.campaigns.portfolio.weighting import (
        WeightingScheme,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def evaluate_long_only_portfolio(
    scored: Sequence[tuple[SupervisedAlphaSample, float]],
    *,
    slippage_bps_per_turnover: float,
    config: CampaignPortfolioConfig,
    previous_weights: Mapping[uuid.UUID, float] | None = None,
    exposure_scale: float = 1.0,
    cost_model: TradingCostModel | None = None,
    weighting: WeightingScheme | None = None,
    selection: SelectionStrategy | None = None,
) -> PortfolioEvaluation:
    """Build path-dependent long-only daily returns from scored samples.

    Constraint precedence on rebalance days
    ---------------------------------------
    Targets pass through, in order:

    1. ``raw_long_only_target`` (top-N equal-weighted, capped at
       ``max_single_name_weight`` and ``max_gross_exposure``).
    2. ``enforce_weight_limits`` (level cap, idempotent).
    3. ``apply_no_trade_band`` (hold sub-band moves).
    4. ``apply_position_change_cap`` (per-name |Δ| cap).
    5. ``enforce_weight_limits`` again — this re-imposes the gross cap if the
       per-name moves above lifted the book above it.
    6. ``apply_turnover_cap`` (sum |Δ| cap).
    7. ``enforce_weight_limits`` once more for the same reason after turnover
       scaling.

    The two soft-vs-hard tensions worth knowing:

    * Steps 5 and 7 honor the *gross-cap* over ``max_position_change``: a
      uniform scale-down that brings the book back inside the gross cap can
      shrink a name by more than ``max_position_change`` would otherwise
      allow. Gross is the harder risk constraint and wins.
    * Realised ``turnover`` is recomputed from the final post-cap state with
      ``_turnover(last_weights, today_weights)``. ``apply_turnover_cap``
      returns ``max_daily_turnover`` when it scales, but the subsequent
      ``enforce_weight_limits`` step can shrink that further, so trusting the
      scaler's return value would overcharge slippage.

    Cost
    ----
    ``cost_model`` prices each rebalance from its per-name trade vector. When it
    is ``None`` the evaluator builds a :class:`LinearTurnoverCost` from
    ``slippage_bps_per_turnover`` — the behavior-preserving default that charges
    exactly ``turnover * bps / 1e4`` as the inlined code did. A convex model
    (e.g. :class:`QuadraticImpactCost`, Arm K) is a drop-in swap; nothing else in
    the construction path changes.

    Weighting
    ---------
    ``weighting`` sizes the selected top-N names. ``None`` uses
    :class:`EqualWeight` (the prior behavior, bit-identical). An inverse-vol
    scheme (:class:`InverseVolWeight`, Arm L) is a drop-in swap — it changes only
    the size distribution, never which names are held, and shares the same gross
    budget, so the selected set is unchanged versus the equal-weight arm.

    Selection
    ---------
    ``selection`` chooses which names to hold. ``None`` uses
    :class:`TopNSelection` (the fresh top-N, bit-identical). A buffered strategy
    (:class:`BufferedTopKSelection`, Arm M) keeps slipping incumbents instead of
    churning them — changing the held book (returns + turnover) but not the
    reported IC, which is measured over the full scored cross-section each day,
    not the held names. The current book (``last_weights``) is passed as the
    incumbent set on every rebalance.
    """
    by_day: dict[datetime, list[tuple[SupervisedAlphaSample, float]]] = {}
    for row, score in scored:
        by_day.setdefault(row.as_of, []).append((row, float(score)))

    returns: list[float] = []
    ics: list[tuple[str, float]] = []
    turnovers: list[float] = []
    diagnostics: list[dict[str, object]] = []
    last_weights = enforce_weight_limits(
        clean_weights(previous_weights or {}),
        config=config,
    )
    # ``cost_model=None`` rebuilds the behavior-preserving linear model from the
    # flat ``slippage_bps_per_turnover`` so pre-seam callers are bit-identical
    # (LinearTurnoverCost computes exactly ``turnover * bps / 1e4``). An explicit
    # model (e.g. QuadraticImpactCost, Arm K) prices each rebalance from its
    # per-name trade vector instead.
    active_cost_model = (
        cost_model if cost_model is not None else LinearTurnoverCost(slippage_bps_per_turnover)
    )
    if exposure_scale < 0.0:
        raise ValueError("exposure_scale must be >= 0")
    rebalance_interval = max(1, int(config.rebalance_interval_days))

    # Return-accounting mode. ``forward_return`` is a multi-day label and
    # must not be compounded as a daily realized P&L. When every scored
    # sample carries ``realized_return_1d`` (a one-day simple realized
    # return), the evaluator marks the held book to market with that field
    # — the canonical correct behavior. Otherwise it falls back to using
    # ``forward_return`` to preserve compatibility with the existing test
    # suite and any caller that has not yet migrated. Legacy mode carries
    # the well-known label-as-P&L bug and must not gate production.
    # See ``docs/architecture/adr-003-return-accounting-separation.md``
    # for the rationale.
    realized_mode = has_realized_returns(scored)

    for day_index, (as_of, rows) in enumerate(sorted(by_day.items())):
        labels = [row.forward_return for row, _ in rows]
        scores = [score for _, score in rows]
        ics.append((as_of.date().isoformat(), _spearman(scores, labels)))

        # Rebalance only on interval days; carry weights forward (zero turnover)
        # between them.  The no-trade band suppresses sub-band churn on the
        # rebalance days themselves.
        is_rebalance = (day_index % rebalance_interval) == 0
        if is_rebalance:
            # ``last_weights`` is the prior rebalance's book here (reassigned at
            # the end of the loop), so it is the correct current-holdings set for
            # a buffered selection's incumbent test.
            target = raw_long_only_target(
                rows,
                config=config,
                weighting=weighting,
                selection=selection,
                current_holdings=frozenset(last_weights),
            )
            target = enforce_weight_limits(target, config=config)
            target = {
                instrument_id: weight * exposure_scale for instrument_id, weight in target.items()
            }
            target = enforce_weight_limits(target, config=config)
            after_band = apply_no_trade_band(
                current=last_weights,
                target=target,
                band=float(config.no_trade_band),
            )
            after_position_cap = apply_position_change_cap(
                current=last_weights,
                target=after_band,
                max_position_change=float(config.max_position_change),
            )
            after_risk_cap = enforce_weight_limits(after_position_cap, config=config)
            today_weights, turnover = apply_turnover_cap(
                current=last_weights,
                target=after_risk_cap,
                max_daily_turnover=float(config.max_daily_turnover),
            )
            today_weights = enforce_weight_limits(today_weights, config=config)
            turnover = _turnover(last_weights, today_weights)
            target_count = len(target)
        else:
            # Non-rebalance day: carry weights forward unchanged. ``last_weights``
            # has already been through ``enforce_weight_limits`` on entry
            # (line above the loop) or on the previous rebalance iteration, so
            # re-enforcing here would either be a no-op or — worse — silently
            # trade on a non-rebalance day if ``config`` tightened mid-run.
            today_weights = dict(last_weights)
            turnover = 0.0
            target_count = len(last_weights)

        max_position_change = max(
            (
                abs(today_weights.get(instrument_id, 0.0) - last_weights.get(instrument_id, 0.0))
                for instrument_id in set(today_weights) | set(last_weights)
            ),
            default=0.0,
        )

        if realized_mode:
            row_returns: dict[uuid.UUID, float] = {
                row.instrument_id: float(row.realized_return_1d or 0.0) for row, _ in rows
            }
        else:
            row_returns = {row.instrument_id: float(row.forward_return) for row, _ in rows}
        gross_return = sum(
            weight * row_returns.get(instrument_id, 0.0)
            for instrument_id, weight in today_weights.items()
        )
        # Per-name trade vector (today − prior); ``last_weights`` is still the
        # prior day's book here (reassigned at the end of the loop). The cost
        # model prices the rebalance from it. On non-rebalance days
        # ``today_weights == last_weights`` so the vector is all-zero -> 0 cost,
        # matching the prior ``turnover * bps`` behavior (turnover is 0 too).
        trades = {
            instrument_id: today_weights.get(instrument_id, 0.0)
            - last_weights.get(instrument_id, 0.0)
            for instrument_id in set(today_weights) | set(last_weights)
        }
        slippage_cost = active_cost_model.cost(trades)
        returns.append(gross_return - slippage_cost)
        turnovers.append(turnover)
        diagnostics.append(
            _day_diagnostics(
                as_of=as_of,
                weights=today_weights,
                turnover=turnover,
                gross_return=gross_return,
                slippage_cost=slippage_cost,
                exposure_scale=exposure_scale,
                max_position_change=max_position_change,
                selected_count=sum(1 for _, score in rows if score > 0.0),
                target_count=target_count,
                rebalanced=is_rebalance,
            )
        )
        last_weights = today_weights

    return PortfolioEvaluation(
        daily_returns=tuple(returns),
        daily_ics=tuple(ics),
        daily_turnover=tuple(turnovers),
        final_weights=last_weights,
        day_diagnostics=tuple(diagnostics),
    )


def _turnover(
    prior: Mapping[uuid.UUID, float],
    current: Mapping[uuid.UUID, float],
) -> float:
    keys = set(prior) | set(current)
    return sum(abs(float(current.get(key, 0.0)) - float(prior.get(key, 0.0))) for key in keys)


def _day_diagnostics(
    *,
    as_of: datetime,
    weights: Mapping[uuid.UUID, float],
    turnover: float,
    gross_return: float,
    slippage_cost: float,
    exposure_scale: float,
    max_position_change: float,
    selected_count: int,
    target_count: int,
    rebalanced: bool,
) -> dict[str, object]:
    gross = sum(abs(weight) for weight in weights.values())
    net = sum(weights.values())
    sorted_weights = sorted((abs(weight) for weight in weights.values()), reverse=True)
    top5 = sum(sorted_weights[:5])
    hhi = sum(weight * weight for weight in sorted_weights)
    return {
        "date": as_of.date().isoformat(),
        "gross_exposure": float(gross),
        "net_exposure": float(net),
        "cash": float(max(0.0, 1.0 - gross)),
        "position_count": float(len(weights)),
        "turnover": float(turnover),
        "max_position_change": float(max_position_change),
        "max_name_weight": float(max(sorted_weights, default=0.0)),
        "top5_concentration": float(top5),
        "hhi": float(hhi),
        "gross_return": float(gross_return),
        "slippage_cost": float(slippage_cost),
        "exposure_scale": float(exposure_scale),
        "positive_score_count": float(selected_count),
        "target_position_count": float(target_count),
        "rebalanced": bool(rebalanced),
    }


__all__ = ["evaluate_long_only_portfolio"]

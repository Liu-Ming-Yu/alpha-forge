"""Portfolio diagnostics for governed research campaign artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    equity_curve,
    max_drawdown,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.campaigns.portfolio.construction import (
        CampaignPortfolioConfig,
        FoldVolatilityScale,
    )

_HIGHLIGHT_FOLDS = (4, 6, 8)


def portfolio_config_payload(config: CampaignPortfolioConfig | None) -> dict[str, object]:
    """Serialize effective portfolio construction defaults and overrides."""
    if config is None:
        return {}
    return config.to_payload()


def fold_portfolio_diagnostics(
    *,
    fold_index: int,
    day_diagnostics: Sequence[Mapping[str, object]],
    daily_returns: Sequence[float],
    volatility_scale: FoldVolatilityScale,
) -> dict[str, object]:
    """Summarize exposure and concentration metrics for one fold."""
    return {
        "fold_index": int(fold_index),
        "days": len(day_diagnostics),
        "avg_gross_exposure": _mean_field(day_diagnostics, "gross_exposure"),
        "max_gross_exposure": _max_field(day_diagnostics, "gross_exposure"),
        "avg_net_exposure": _mean_field(day_diagnostics, "net_exposure"),
        "max_net_exposure": _max_field(day_diagnostics, "net_exposure"),
        "avg_cash": _mean_field(day_diagnostics, "cash"),
        "min_cash": _min_field(day_diagnostics, "cash"),
        "avg_position_count": _mean_field(day_diagnostics, "position_count"),
        "max_position_count": _max_field(day_diagnostics, "position_count"),
        "avg_turnover": _mean_field(day_diagnostics, "turnover"),
        "max_turnover": _max_field(day_diagnostics, "turnover"),
        "max_position_change": _max_field(day_diagnostics, "max_position_change"),
        "max_name_weight": _max_field(day_diagnostics, "max_name_weight"),
        "avg_top5_concentration": _mean_field(day_diagnostics, "top5_concentration"),
        "max_top5_concentration": _max_field(day_diagnostics, "top5_concentration"),
        "avg_hhi": _mean_field(day_diagnostics, "hhi"),
        "max_hhi": _max_field(day_diagnostics, "hhi"),
        "volatility_scale": volatility_scale.to_payload(),
        "daily": [dict(row) for row in day_diagnostics],
        "fold_total_return": _compound_return(daily_returns),
        "fold_max_drawdown": max_drawdown(daily_returns),
    }


def drawdown_diagnostics_payload(
    fold_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build campaign drawdown diagnostics with fold 4/6/8 highlights."""
    folds: list[dict[str, object]] = [
        {
            "fold_index": _as_int(row.get("fold_index", 0)),
            "test_start": row.get("test_start"),
            "test_end": row.get("test_end"),
            "max_drawdown": _as_float(row.get("max_drawdown", 0.0)),
            "total_return": _as_float(row.get("total_return", 0.0)),
            "slippage_adjusted_sharpe": _as_float(row.get("slippage_adjusted_sharpe", 0.0)),
            "turnover_avg": _as_float(row.get("turnover_avg", 0.0)),
        }
        for row in fold_rows
    ]
    highlighted = [
        row for row in folds if _as_int(row.get("fold_index", 0)) in set(_HIGHLIGHT_FOLDS)
    ]
    worst = min(folds, key=lambda row: _as_float(row.get("max_drawdown")), default=None)
    return {
        "folds": folds,
        "highlight_fold_indices": list(_HIGHLIGHT_FOLDS),
        "highlighted_folds": highlighted,
        "worst_fold": worst,
    }


def portfolio_diagnostics_payload(
    fold_portfolio_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build portfolio diagnostics payload from per-fold summaries."""
    rows = [dict(row) for row in fold_portfolio_rows]
    return {
        "folds": rows,
        "aggregate": {
            "avg_gross_exposure": _mean_field(rows, "avg_gross_exposure"),
            "max_gross_exposure": _max_field(rows, "max_gross_exposure"),
            "avg_net_exposure": _mean_field(rows, "avg_net_exposure"),
            "max_net_exposure": _max_field(rows, "max_net_exposure"),
            "avg_cash": _mean_field(rows, "avg_cash"),
            "min_cash": _min_field(rows, "min_cash"),
            "avg_turnover": _mean_field(rows, "avg_turnover"),
            "max_turnover": _max_field(rows, "max_turnover"),
            "max_position_change": _max_field(rows, "max_position_change"),
            "max_name_weight": _max_field(rows, "max_name_weight"),
            "avg_top5_concentration": _mean_field(rows, "avg_top5_concentration"),
            "max_top5_concentration": _max_field(rows, "max_top5_concentration"),
            "avg_hhi": _mean_field(rows, "avg_hhi"),
            "max_hhi": _max_field(rows, "max_hhi"),
        },
    }


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = _numeric_field(rows, field)
    return sum(values) / len(values) if values else 0.0


def _max_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = _numeric_field(rows, field)
    return max(values) if values else 0.0


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = _numeric_field(rows, field)
    return min(values) if values else 0.0


def _numeric_field(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _maybe_float(row.get(field, 0.0))
        if value is None:
            continue
        values.append(value)
    return values


def _as_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(cast("Any", value or 0.0))
    except (TypeError, ValueError, OverflowError):
        return default


def _maybe_float(value: object) -> float | None:
    try:
        return float(cast("Any", value or 0.0))
    except (TypeError, ValueError, OverflowError):
        return None


def _as_int(value: object) -> int:
    try:
        return int(cast("Any", value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _compound_return(returns: Sequence[float]) -> float:
    return equity_curve(returns)[-1] - 1.0 if returns else 0.0


__all__ = [
    "drawdown_diagnostics_payload",
    "fold_portfolio_diagnostics",
    "portfolio_config_payload",
    "portfolio_diagnostics_payload",
]

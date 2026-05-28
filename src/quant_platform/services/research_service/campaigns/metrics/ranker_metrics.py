"""Pure ranker scoring and walk-forward metric helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.constants import BPS_PER_UNIT
from quant_platform.services.research_service.reports.statistics import bootstrap_mean_ci
from quant_platform.services.research_service.reports.statistics import mean as _mean
from quant_platform.services.research_service.reports.statistics import spearman_ic as _spearman
from quant_platform.services.research_service.sampling.samples import has_realized_returns

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def fit_correlation_weights(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str] | None = None,
    *,
    non_negative: bool = False,
) -> dict[str, float]:
    """Fit IC-proportional feature weights.

    When ``non_negative`` is set, factors with negative in-sample IC are dropped
    (weight 0) rather than shorted.  The platform's classical alpha factors are
    all positive-oriented (higher value = more attractive), so a negative weight
    is always regime-overfit: it shorts a factor with a real positive long-term
    premium (e.g. the low-volatility anomaly) just because it lagged in-sample.
    """
    selected_names = _selected_feature_names(samples, feature_names)
    raw: dict[str, float] = {}
    labels = [row.forward_return for row in samples]
    for name in selected_names:
        values = [float(row.features.get(name, 0.0)) for row in samples]
        ic = _spearman(values, labels)
        raw[name] = max(0.0, ic) if non_negative else ic
    total = sum(abs(value) for value in raw.values())
    if total <= 0:
        equal = 1.0 / max(1, len(selected_names))
        return {name: equal for name in selected_names}
    return {name: value / total for name, value in raw.items() if value != 0.0}


def equal_weights(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    selected_names = _selected_feature_names(samples, feature_names)
    if not selected_names:
        return {}
    weight = 1.0 / len(selected_names)
    return {name: weight for name in selected_names}


def _selected_feature_names(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str] | None,
) -> list[str]:
    available = {name for row in samples for name in row.features}
    if feature_names is None:
        return sorted(available)
    return sorted({name for name in feature_names if name in available})


def score_features(features: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(float(features.get(name, 0.0)) * weight for name, weight in weights.items())


def daily_metrics(
    scored: Sequence[tuple[SupervisedAlphaSample, float]],
    slippage_bps_per_turnover: float,
    *,
    prev_scores: dict[uuid.UUID, float] | None = None,
) -> tuple[list[float], list[tuple[str, float]], list[float], dict[uuid.UUID, float]]:
    """Daily IC + turnover-aware daily-MtM return + persistent carry weights.

    **Return accounting.** ``forward_return`` is a *label* (the multi-day
    forward return used to score features). It must not be compounded as a
    daily realized P&L. When every scored sample carries
    ``realized_return_1d`` (a one-day simple realized return), this function
    runs in **realized mode**:

    * One rebalance per fold call, at the first ``as_of`` in ``scored``.
    * Hold the rebalance-day weights through the remaining days of the fold.
    * Each day's portfolio return is the weighted sum of holdings'
      ``realized_return_1d`` — a true one-day mark-to-market return that
      compounds correctly via ``equity *= 1 + r``.
    * Turnover slippage is charged only on the rebalance day; carry days
      contribute zero turnover.

    When **any** sample lacks ``realized_return_1d`` the function falls back
    to **legacy mode**: rebalance every day from that day's scores and use
    ``forward_return`` as the daily return. Legacy mode preserves the
    previous behavior for the existing test suite and any external caller
    that has not opted into realized accounting; it has the 21d-label
    compounding bug and must not be used to make production gating
    decisions.

    The IC series is computed in both modes against ``forward_return`` (the
    correct label for IC) at every observed ``as_of``.

    **Held-but-missing instrument semantics.** In realized mode the book is
    decided on the rebalance day from ``rebalance_rows``. If a held name
    has no row on a subsequent carry day (delisted, halted, missing data
    upstream), its ``realized_return_1d`` lookup falls back to 0.0 — the
    book contributes nothing for that name that day. This is the most
    conservative choice (no spurious return, no negative-infinity panic)
    but it does *not* model exit-at-last-close or position-redistribution.
    With the global trading-day calendar in the latest-stack builder this
    case is rare; if it becomes common, surface a per-fold count in
    ``portfolio_diagnostics`` instead of silently absorbing the gap.

    See ``docs/architecture/adr-003-return-accounting-separation.md`` for
    the rationale behind the dual-mode design.
    """
    by_day: dict[datetime, list[tuple[SupervisedAlphaSample, float]]] = {}
    for row, score in scored:
        by_day.setdefault(row.as_of, []).append((row, score))
    returns: list[float] = []
    ics: list[tuple[str, float]] = []
    turnovers: list[float] = []
    bps_per_turnover = slippage_bps_per_turnover / BPS_PER_UNIT

    sorted_days = sorted(by_day.keys())
    if not sorted_days:
        return [], [], [], dict(prev_scores or {})

    # IC is always computed against forward_return — the predictive label.
    for as_of in sorted_days:
        rows = by_day[as_of]
        labels = [row.forward_return for row, _ in rows]
        day_scores = [score for _, score in rows]
        ics.append((as_of.date().isoformat(), _spearman(day_scores, labels)))

    if not has_realized_returns(scored):
        # Legacy path: every day a rebalance, forward_return as daily return.
        # Kept for backward compat with tests + callers that haven't migrated.
        carry_weights: dict[uuid.UUID, float] = dict(prev_scores or {})
        for as_of in sorted_days:
            rows = by_day[as_of]
            day_scores = [score for _, score in rows]
            denom = sum(abs(score) for score in day_scores)
            if denom <= 0:
                today_weights: dict[uuid.UUID, float] = {row.instrument_id: 0.0 for row, _ in rows}
            else:
                today_weights = {row.instrument_id: score / denom for row, score in rows}
            keys = set(carry_weights) | set(today_weights)
            turnover = sum(
                abs(today_weights.get(key, 0.0) - carry_weights.get(key, 0.0)) for key in keys
            )
            turnovers.append(turnover)
            if denom <= 0:
                returns.append(-turnover * bps_per_turnover)
            else:
                weighted = sum(score * row.forward_return for row, score in rows) / denom
                returns.append(weighted - turnover * bps_per_turnover)
            carry_weights = today_weights
        return returns, ics, turnovers, carry_weights

    # Realized mode: rebalance once at the first as_of of the fold,
    # carry weights through subsequent days, mark-to-market with
    # realized_return_1d.
    rebalance_day = sorted_days[0]
    rebalance_rows = by_day[rebalance_day]
    rebalance_scores = [score for _, score in rebalance_rows]
    denom_rebal = sum(abs(score) for score in rebalance_scores)
    if denom_rebal <= 0:
        carry_weights = {row.instrument_id: 0.0 for row, _ in rebalance_rows}
    else:
        carry_weights = {row.instrument_id: score / denom_rebal for row, score in rebalance_rows}
    prev_carry = dict(prev_scores or {})
    rebalance_turnover = sum(
        abs(carry_weights.get(key, 0.0) - prev_carry.get(key, 0.0))
        for key in set(prev_carry) | set(carry_weights)
    )

    for as_of in sorted_days:
        rows = by_day[as_of]
        realized_by_id = {
            row.instrument_id: float(row.realized_return_1d or 0.0) for row, _ in rows
        }
        # Held-but-missing instruments contribute 0 — see docstring.
        gross_return = sum(
            weight * realized_by_id.get(instrument_id, 0.0)
            for instrument_id, weight in carry_weights.items()
        )
        if as_of == rebalance_day:
            turnovers.append(rebalance_turnover)
            returns.append(gross_return - rebalance_turnover * bps_per_turnover)
        else:
            turnovers.append(0.0)
            returns.append(gross_return)
    return returns, ics, turnovers, carry_weights


def feature_stability(fold_weights: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean absolute change in selected weight per feature across folds."""
    if len(fold_weights) < 2:
        return {name: 0.0 for name in (fold_weights[0] if fold_weights else {})}
    feature_names = sorted({name for fold in fold_weights for name in fold})
    changes: dict[str, list[float]] = {name: [] for name in feature_names}
    for prior, current in zip(fold_weights[:-1], fold_weights[1:], strict=True):
        for name in feature_names:
            changes[name].append(abs(current.get(name, 0.0) - prior.get(name, 0.0)))
    return {name: _mean(values) for name, values in changes.items()}


def bootstrap_ic_ci(
    ics: Sequence[float],
    *,
    samples: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a 95 percent bootstrap confidence interval for mean IC."""
    return bootstrap_mean_ci(
        ics,
        seed=seed,
        samples=samples,
        lower_quantile=0.025,
        upper_quantile=0.975,
        round_indices=True,
    )


def attribution_by_metadata(
    scored: Sequence[tuple[SupervisedAlphaSample, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Group average IC and average forward return by metadata key/value."""
    groups: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for sample, score in scored:
        for key, value in sample.metadata:
            buckets = groups.setdefault(key, {})
            buckets.setdefault(value, []).append((score, sample.forward_return))
    result: dict[str, dict[str, dict[str, float]]] = {}
    for key, buckets in groups.items():
        result[key] = {}
        for value, observations in buckets.items():
            scores = [item[0] for item in observations]
            labels = [item[1] for item in observations]
            result[key][value] = {
                "observations": float(len(observations)),
                "mean_score": _mean(scores),
                "mean_forward_return": _mean(labels),
                "ic": _spearman(scores, labels),
            }
    return result


def top_minus_bottom_decile_ic(
    scored: Sequence[tuple[SupervisedAlphaSample, float]],
) -> float:
    """Decile-spread metric: mean forward return of top decile minus bottom."""
    if len(scored) < 10:
        return 0.0
    ordered = sorted(scored, key=lambda item: item[1])
    decile_size = max(1, len(ordered) // 10)
    bottom = ordered[:decile_size]
    top = ordered[-decile_size:]
    return _mean([row.forward_return for row, _ in top]) - _mean(
        [row.forward_return for row, _ in bottom]
    )


__all__ = [
    "attribution_by_metadata",
    "bootstrap_ic_ci",
    "daily_metrics",
    "equal_weights",
    "feature_stability",
    "fit_correlation_weights",
    "score_features",
    "top_minus_bottom_decile_ic",
]

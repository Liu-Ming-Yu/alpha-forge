"""Purged walk-forward campaign evaluation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal, cast

from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    attribution_by_metadata,
    bootstrap_ic_ci,
    daily_metrics,
    equal_weights,
    fit_correlation_weights,
    score_features,
    top_minus_bottom_decile_ic,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    feature_stability as compute_feature_stability,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    compound_return,
    max_drawdown,
    sharpe,
)
from quant_platform.services.research_service.campaigns.portfolio.construction import (
    CampaignPortfolioConfig,
    evaluate_long_only_portfolio,
    fit_fold_volatility_scale,
)
from quant_platform.services.research_service.campaigns.portfolio.diagnostics import (
    drawdown_diagnostics_payload,
    fold_portfolio_diagnostics,
    portfolio_config_payload,
    portfolio_diagnostics_payload,
)
from quant_platform.services.research_service.campaigns.portfolio.streak_risk import (
    FoldStreakRiskConfig,
    FoldStreakRiskScale,
    compute_fold_streak_exposure_scale,
    fold_streak_diagnostics_payload,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
    generate_folds,
)
from quant_platform.services.research_service.reports.statistics import mean as _mean
from quant_platform.services.research_service.reports.statistics import (
    negative_streak as _negative_streak,
)
from quant_platform.services.research_service.sampling.eligibility import eligibility
from quant_platform.services.research_service.sampling.factory_models import (
    AlphaEligibilityThresholds,
    WalkForwardEvidence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


#: Per-fold tag describing which purge mechanism ran for that fold.
#: ``"calendar_days"`` — ``WalkForwardConfig.purge_days`` only (legacy callers
#: that don't carry per-sample label indices).
#: ``"calendar_days_plus_sample_label_index_purge"`` — calendar purge plus
#: the stricter sample-level filter on the global trading-day calendar.
FoldBasis = Literal["calendar_days", "calendar_days_plus_sample_label_index_purge"]


def run_sample_walk_forward(
    *,
    samples: Sequence[SupervisedAlphaSample],
    config: WalkForwardConfig,
    model_version: str,
    feature_set_version: str,
    thresholds: AlphaEligibilityThresholds | None = None,
    slippage_bps_per_turnover: float = 10.0,
    feature_names: Sequence[str] | None = None,
    weight_mode: Literal["ic_weighted", "equal_weight"] = "ic_weighted",
    return_scale: float = 1.0,
    portfolio_config: CampaignPortfolioConfig | None = None,
    fold_streak_risk_config: FoldStreakRiskConfig | None = None,
) -> WalkForwardEvidence:
    """Evaluate a simple learned linear ranker over purged walk-forward folds.

    See ``docs/architecture/adr-003-return-accounting-separation.md`` for
    the realized-vs-legacy mode contract; this driver enforces all-or-
    nothing realized-mode samples so the evaluator cannot silently switch
    modes between folds.

    ``fold_streak_risk_config`` (long-only arms only) enables the
    fold-streak exposure throttle: each fold's exposure scale is
    additionally multiplied by a factor in ``[0, 1]`` computed from the
    prior folds' mean OOS ICs (EWMA smooth throttle + consecutive-
    negative-IC circuit breaker). PIT-safe by construction — the
    current fold's IC is appended to history only AFTER its own
    evaluation completes. Signed-rank arms (no ``portfolio_config``)
    ignore the streak config; they are diagnostic baselines, not
    portfolio candidates, and adding the dial there is noise.
    """
    if not samples:
        raise ValueError("walk-forward requires at least one sample")
    if weight_mode not in {"ic_weighted", "equal_weight"}:
        raise ValueError(f"unsupported walk-forward weight_mode: {weight_mode}")
    if return_scale < 0.0:
        raise ValueError("return_scale must be >= 0")
    # All-or-nothing realized-mode check. Per-fold dispatch in the
    # evaluators (``ranker_metrics.daily_metrics`` and
    # ``portfolio.evaluation.evaluate_long_only_portfolio``) decides
    # realized-vs-legacy mode from sample-field presence; if some folds
    # have ``realized_return_1d`` set and others don't, the same run
    # would silently mix gated and ungated arithmetic across folds.
    # Fail loud here instead.
    realized_count = sum(1 for s in samples if s.realized_return_1d is not None)
    if realized_count not in (0, len(samples)):
        raise ValueError(
            "samples mix realized-mode and legacy-mode rows: "
            f"{realized_count} of {len(samples)} carry realized_return_1d. "
            "Either populate realized_return_1d on every sample (realized "
            "mode) or none (legacy mode); silent per-fold mode switching "
            "would corrupt the metrics."
        )
    # Index-presence is similarly all-or-nothing for the sample-level
    # purge. Same reasoning, narrower scope.
    indexed_count = sum(
        1 for s in samples if s.as_of_index is not None and s.label_end_index is not None
    )
    if indexed_count not in (0, len(samples)):
        raise ValueError(
            "samples mix indexed and unindexed rows: "
            f"{indexed_count} of {len(samples)} carry as_of_index + "
            "label_end_index. Either populate both indices on every "
            "sample or none."
        )
    samples_have_indices = indexed_count == len(samples)
    thresholds = thresholds or AlphaEligibilityThresholds()
    ordered = sorted(samples, key=lambda row: (row.as_of, str(row.instrument_id)))
    start = ordered[0].as_of
    end = ordered[-1].as_of
    folds = generate_folds(start, end, config)

    fold_rows: list[dict[str, object]] = []
    all_daily_returns: list[float] = []
    all_daily_ics: list[tuple[str, float]] = []
    all_daily_turnover: list[float] = []
    all_scored: list[tuple[SupervisedAlphaSample, float]] = []
    selected_weights: dict[str, float] = {}
    fold_weights: list[dict[str, float]] = []
    fold_portfolio_rows: list[dict[str, object]] = []

    # Fold-streak exposure throttle bookkeeping. ``prior_fold_ics`` is
    # appended ONLY after a fold's evaluation completes — never before —
    # so the scale for fold N is computed strictly from folds 0..N-1.
    # ``per_fold_streak_scales`` mirrors the order of ``fold_rows`` so
    # diagnostics line up one-to-one with the folds that actually ran.
    prior_fold_ics: list[float] = []
    per_fold_streak_scales: list[FoldStreakRiskScale] = []

    prev_scores: dict[uuid.UUID, float] | None = None
    prev_portfolio_weights: dict[uuid.UUID, float] | None = None
    for fold in folds:
        train = [row for row in ordered if fold.train_start <= row.as_of < fold.train_end]
        test = [row for row in ordered if fold.test_start <= row.as_of < fold.test_end]
        if not train or not test:
            continue
        # Sample-level purge. When every sample carries ``as_of_index``
        # and ``label_end_index`` on the *global* trading-day calendar
        # (validated all-or-nothing at the top of this function), drop
        # training rows whose forward-return label window reaches into
        # the test period. Stricter than ``WalkForwardConfig.purge_days``
        # because that purge counts calendar days while labels live on
        # trading days; a 21-trading-day label can leak past a
        # 21-calendar-day purge gap. ``fold_basis`` records which mode
        # ran so downstream evidence is self-describing.
        fold_basis: FoldBasis
        if samples_have_indices:
            # test/train are sub-lists of ``samples``, which we already
            # validated as all-indexed — every row's indices are set.
            # ``cast`` is a type-system no-op that lets the comprehension
            # avoid a redundant ``is not None`` check on every row.
            test_start_index = min(cast("int", row.as_of_index) for row in test)
            train = [row for row in train if cast("int", row.label_end_index) < test_start_index]
            if not train:
                continue
            fold_basis = "calendar_days_plus_sample_label_index_purge"
        else:
            fold_basis = "calendar_days"
        if weight_mode == "equal_weight":
            weights = equal_weights(train, feature_names)
        else:
            # Non-negative fit: classical alpha factors are positive-oriented,
            # so a negative weight is always regime-overfit (it would short a
            # factor with a real positive premium). Drop, do not short.
            weights = fit_correlation_weights(train, feature_names, non_negative=True)
        if feature_names is not None and not weights:
            raise ValueError("walk-forward selected no matching features")
        selected_weights = weights
        fold_weights.append(dict(weights))
        scored = [(row, score_features(row.features, weights)) for row in test]
        all_scored.extend(scored)
        volatility_payload: dict[str, float] | None = None
        streak_scale_payload: dict[str, object] | None = None
        if portfolio_config is None:
            daily_returns, daily_ics, daily_turnover, prev_scores = daily_metrics(
                scored,
                slippage_bps_per_turnover,
                prev_scores=prev_scores,
            )
        else:
            train_scored = [(row, score_features(row.features, weights)) for row in train]
            volatility_scale = fit_fold_volatility_scale(
                train_scored,
                config=portfolio_config,
            )
            # Fold-streak throttle. Computed from ``prior_fold_ics`` —
            # which contains ONLY completed prior folds at this point;
            # the current fold's IC is appended later, after evaluation.
            # When the dial is disabled, the scale is unconditionally
            # 1.0 and the effective exposure equals the vol-scale alone.
            if fold_streak_risk_config is None:
                streak_scale_factor = 1.0
            else:
                streak_scale_obj = compute_fold_streak_exposure_scale(
                    tuple(prior_fold_ics),
                    config=fold_streak_risk_config,
                )
                per_fold_streak_scales.append(streak_scale_obj)
                streak_scale_factor = streak_scale_obj.scale
                streak_scale_payload = streak_scale_obj.to_payload()
            effective_exposure_scale = volatility_scale.exposure_scale * streak_scale_factor
            portfolio_eval = evaluate_long_only_portfolio(
                scored,
                slippage_bps_per_turnover=slippage_bps_per_turnover,
                config=portfolio_config,
                previous_weights=prev_portfolio_weights,
                exposure_scale=effective_exposure_scale,
            )
            daily_returns = list(portfolio_eval.daily_returns)
            daily_ics = list(portfolio_eval.daily_ics)
            daily_turnover = list(portfolio_eval.daily_turnover)
            prev_portfolio_weights = portfolio_eval.final_weights
            volatility_payload = volatility_scale.to_payload()
            fold_portfolio_rows.append(
                fold_portfolio_diagnostics(
                    fold_index=fold.fold_index,
                    day_diagnostics=portfolio_eval.day_diagnostics,
                    daily_returns=daily_returns,
                    volatility_scale=volatility_scale,
                )
            )
        if return_scale != 1.0:
            daily_returns = [value * return_scale for value in daily_returns]
            daily_turnover = [value * return_scale for value in daily_turnover]
        all_daily_returns.extend(daily_returns)
        all_daily_ics.extend(daily_ics)
        all_daily_turnover.extend(daily_turnover)
        # Append THIS fold's mean OOS IC to the streak-history AFTER
        # the fold has been evaluated. Done outside the if/else so
        # the history is consistent whether or not the dial is active
        # — keeps the per-fold history identical between Arm D and
        # Arm E so the comparison isolates the dial's effect.
        prior_fold_ics.append(_mean([ic for _, ic in daily_ics]))
        fold_rows.append(
            {
                "fold_index": fold.fold_index,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "train_samples": len(train),
                "test_samples": len(test),
                "mean_ic": _mean([ic for _, ic in daily_ics]),
                "slippage_adjusted_sharpe": sharpe(daily_returns),
                "total_return": compound_return(daily_returns),
                "max_drawdown": max_drawdown(daily_returns),
                "turnover_avg": _mean(daily_turnover),
                "portfolio_mode": portfolio_config.mode
                if portfolio_config is not None
                else "signed-rank",
                "volatility_scale": volatility_payload,
                "fold_streak_scale": streak_scale_payload,
                "fold_basis": fold_basis,
            }
        )

    if not fold_rows:
        raise ValueError("walk-forward produced no folds with train and test samples")

    feature_stability = compute_feature_stability(fold_weights)
    bootstrap_low, bootstrap_high = bootstrap_ic_ci(
        [ic for _, ic in all_daily_ics],
        seed=42,
    )
    attribution = attribution_by_metadata(all_scored)
    decile_spread = top_minus_bottom_decile_ic(all_scored)
    ic_streak_metrics = _ic_streak_metrics(
        fold_rows=fold_rows,
        all_daily_ics=all_daily_ics,
    )

    metrics = {
        "oos_rolling_ic": _mean([ic for _, ic in all_daily_ics[-20:]]),
        "ic_60d": _mean([ic for _, ic in all_daily_ics[-60:]]),
        "max_drawdown": max_drawdown(all_daily_returns),
        "slippage_adjusted_sharpe": sharpe(all_daily_returns),
        "total_return": compound_return(all_daily_returns),
        "folds": float(len(fold_rows)),
        "daily_observations": float(len(all_daily_returns)),
        "turnover_avg": _mean(all_daily_turnover),
        "feature_stability_avg": _mean(list(feature_stability.values())),
        "bootstrap_ic_p05": bootstrap_low,
        "bootstrap_ic_p95": bootstrap_high,
        "top_minus_bottom_decile_ic": decile_spread,
        "slippage_bps_per_turnover": float(slippage_bps_per_turnover),
        "return_scale": float(return_scale),
        **ic_streak_metrics,
    }
    if portfolio_config is not None:
        portfolio_payload = dict(portfolio_diagnostics_payload(fold_portfolio_rows))
        portfolio_config_map = portfolio_config_payload(portfolio_config)
        drawdown_payload = drawdown_diagnostics_payload(fold_rows)
        # Pack the fold-streak dial's aggregate diagnostics into the
        # existing portfolio_diagnostics field — the dial only applies
        # to portfolio arms, and avoiding a new top-level WalkForwardEvidence
        # field keeps the DTO contract stable for other consumers
        # (artifact payloads, tearsheet, eligibility gate).
        portfolio_payload["fold_streak_risk"] = fold_streak_diagnostics_payload(
            fold_streak_risk_config,
            per_fold_streak_scales,
        )
        aggregate_obj = portfolio_payload.get("aggregate", {})
        aggregate = aggregate_obj if isinstance(aggregate_obj, dict) else {}
        streak_payload = portfolio_payload.get("fold_streak_risk", {})
        streak_payload_dict = streak_payload if isinstance(streak_payload, dict) else {}
        metrics_update: dict[str, float] = {
            "portfolio_effective_max_gross_cap": float(portfolio_config.effective_max_gross_cap),
            "portfolio_avg_gross_exposure": float(aggregate.get("avg_gross_exposure", 0.0)),
            "portfolio_max_gross_exposure": float(aggregate.get("max_gross_exposure", 0.0)),
            "portfolio_avg_net_exposure": float(aggregate.get("avg_net_exposure", 0.0)),
            "portfolio_avg_cash": float(aggregate.get("avg_cash", 0.0)),
            "portfolio_max_turnover": float(aggregate.get("max_turnover", 0.0)),
            "portfolio_max_position_change": float(aggregate.get("max_position_change", 0.0)),
            "portfolio_max_name_weight": float(aggregate.get("max_name_weight", 0.0)),
            "portfolio_max_top5_concentration": float(aggregate.get("max_top5_concentration", 0.0)),
            "portfolio_max_hhi": float(aggregate.get("max_hhi", 0.0)),
        }
        if streak_payload_dict.get("applied"):
            metrics_update.update(
                {
                    "fold_streak_scale_avg": float(streak_payload_dict.get("scale_avg", 1.0)),
                    "fold_streak_scale_min": float(streak_payload_dict.get("scale_min", 1.0)),
                    "fold_streak_zero_fold_count": float(
                        streak_payload_dict.get("zero_fold_count", 0)
                    ),
                }
            )
        metrics.update(metrics_update)
    else:
        portfolio_payload = {}
        portfolio_config_map = {}
        drawdown_payload = {}
    eligibility_result = eligibility(metrics, thresholds)
    return WalkForwardEvidence(
        run_id=uuid.uuid4(),
        model_version=model_version,
        feature_set_version=feature_set_version,
        folds=tuple(fold_rows),
        selected_weights=selected_weights,
        daily_returns=tuple(all_daily_returns),
        daily_ics=tuple(all_daily_ics),
        metrics=metrics,
        eligibility=eligibility_result,
        daily_turnover=tuple(all_daily_turnover),
        feature_stability=feature_stability,
        bootstrap_ic_ci=(bootstrap_low, bootstrap_high),
        attribution=attribution,
        slippage_bps_per_turnover=float(slippage_bps_per_turnover),
        portfolio_config=portfolio_config_map,
        portfolio_diagnostics=portfolio_payload,
        drawdown_diagnostics=drawdown_payload,
    )


def _ic_streak_metrics(
    *,
    fold_rows: Sequence[dict[str, object]],
    all_daily_ics: Sequence[tuple[str, float]],
) -> dict[str, float]:
    """Return fold-level and raw daily IC streak diagnostics.

    Daily ICs for multi-day forward-return horizons are overlapping labels, so
    a long calendar-day negative run can be one bad horizon window repeated
    many times. Eligibility gates on the fold-level streak (mostly independent
    across folds thanks to purge + embargo). The raw daily streak is surfaced
    in metrics for dashboards/diagnostics only.

    The legacy ``negative_ic_streak`` alias was removed on 2026-05-25 — its
    semantics silently flipped from daily-streak to fold-streak when this
    function was introduced, and keeping the alias kept the trap alive. New
    consumers must read ``fold_negative_ic_streak`` (gate metric) or
    ``daily_negative_ic_streak`` (raw diagnostic) by name.
    """
    # ``fold_rows`` is typed as ``Sequence[dict[str, object]]`` because the
    # row payload mixes ints, floats, strings, and nested dicts; mypy will not
    # narrow a bare ``row.get(...)`` for ``float()``. Coerce defensively — in
    # the production caller ``mean_ic`` is always a float written by
    # ``_mean(...)``, but the isinstance guard also keeps malformed test
    # fixtures from blowing up the helper.
    fold_ics: list[float] = []
    for row in fold_rows:
        raw = row.get("mean_ic", 0.0)
        fold_ics.append(float(raw) if isinstance(raw, (int, float)) else 0.0)
    daily_ics = [ic for _, ic in all_daily_ics]
    return {
        "fold_negative_ic_streak": float(_negative_streak(fold_ics)),
        "daily_negative_ic_streak": float(_negative_streak(daily_ics)),
    }

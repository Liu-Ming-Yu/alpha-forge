"""Model-comparison helpers for paper research campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.services.research_service.campaigns.portfolio.construction import (
        CampaignPortfolioConfig,
    )
    from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
        WalkForwardConfig,
    )
    from quant_platform.services.research_service.sampling.factory_models import (
        AlphaEligibilityThresholds,
        WalkForwardEvidence,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


CLASSICAL_RESEARCH_FEATURES = (
    "momentum_1m",
    "momentum_3m",
    "momentum_12m_1m",
    "vol_compression",
    "short_term_reversal_5d",
    "trend_quality_63d",
    "distance_to_52w_high",
)


def build_linear_model_comparison(
    *,
    samples: Sequence[SupervisedAlphaSample],
    config: WalkForwardConfig,
    model_version: str,
    feature_set_version: str,
    thresholds: AlphaEligibilityThresholds,
    slippage_bps_per_turnover: float,
    feature_names: Sequence[str] | None = None,
    return_scale: float = 1.0,
    portfolio_config: CampaignPortfolioConfig | None = None,
) -> tuple[WalkForwardEvidence, list[dict[str, object]]]:
    """Run the baseline and primary linear candidates for campaign comparison."""
    rows: list[dict[str, object]] = []
    admitted = set(feature_names) if feature_names is not None else None
    baseline_features = tuple(
        name for name in CLASSICAL_RESEARCH_FEATURES if admitted is None or name in admitted
    )
    try:
        baseline = run_sample_walk_forward(
            samples=samples,
            config=config,
            model_version=f"{model_version}__classical_baseline",
            feature_set_version=feature_set_version,
            thresholds=thresholds,
            slippage_bps_per_turnover=slippage_bps_per_turnover,
            feature_names=baseline_features,
            weight_mode="equal_weight",
            return_scale=return_scale,
            portfolio_config=portfolio_config,
        )
        rows.append(
            _evidence_row(
                candidate="classical_baseline",
                model_type="equal_weight_linear",
                evidence=baseline,
                selected=False,
                feature_names=baseline_features,
            )
        )
    except ValueError as exc:
        rows.append(
            {
                "candidate": "classical_baseline",
                "model_type": "equal_weight_linear",
                "status": "failed",
                "selected": False,
                "reason": str(exc),
            }
        )

    evidence = run_sample_walk_forward(
        samples=samples,
        config=config,
        model_version=model_version,
        feature_set_version=feature_set_version,
        thresholds=thresholds,
        slippage_bps_per_turnover=slippage_bps_per_turnover,
        feature_names=feature_names,
        return_scale=return_scale,
        portfolio_config=portfolio_config,
    )
    rows.append(
        _evidence_row(
            candidate="ic_weighted_linear",
            model_type="linear_ranker_walk_forward",
            evidence=evidence,
            selected=True,
            feature_names=feature_names,
        )
    )
    return evidence, rows


def xgboost_model_comparison_row(
    *,
    manifest_path: object | None,
    manifest: object | None = None,
    search_candidates: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Build the comparison row for optional XGBoost training."""
    if manifest_path is None or manifest is None:
        return {
            "candidate": "xgboost_ranker",
            "model_type": "xgboost_ranker",
            "status": "skipped",
            "selected": False,
            "reason": "--train-xgboost was not set",
        }
    metrics = dict(getattr(manifest, "metrics", {}) or {})
    return {
        "candidate": "xgboost_ranker",
        "model_type": getattr(manifest, "model_type", "xgboost_ranker"),
        "status": "trained",
        "selected": False,
        "model_version": getattr(manifest, "model_version", ""),
        "manifest_path": str(manifest_path),
        "feature_schema_hash": getattr(manifest, "feature_schema_hash", ""),
        "feature_versions": dict(getattr(manifest, "feature_versions", {}) or {}),
        "metrics": {
            "validation_ic": float(metrics.get("validation_ic", 0.0) or 0.0),
            "train_samples": float(metrics.get("train_samples", 0.0) or 0.0),
            "validation_samples": float(metrics.get("validation_samples", 0.0) or 0.0),
        },
        "search_candidates": list(search_candidates),
    }


def _evidence_row(
    *,
    candidate: str,
    model_type: str,
    evidence: WalkForwardEvidence,
    selected: bool,
    feature_names: Sequence[str] | None = None,
) -> dict[str, object]:
    passed = bool(evidence.eligibility["passed"])
    row: dict[str, object] = {
        "candidate": candidate,
        "model_type": model_type,
        "status": "passed" if passed else "failed",
        "selected": selected and passed,
        "model_version": evidence.model_version,
        "metrics": {
            "oos_rolling_ic": float(evidence.metrics["oos_rolling_ic"]),
            "ic_60d": float(evidence.metrics["ic_60d"]),
            "slippage_adjusted_sharpe": float(evidence.metrics["slippage_adjusted_sharpe"]),
            "max_drawdown": float(evidence.metrics["max_drawdown"]),
            "daily_observations": float(evidence.metrics["daily_observations"]),
            "return_scale": float(evidence.metrics.get("return_scale", 1.0)),
            "portfolio_effective_max_gross_cap": float(
                evidence.metrics.get("portfolio_effective_max_gross_cap", 0.0)
            ),
            "portfolio_max_drawdown": float(evidence.metrics.get("max_drawdown", 0.0)),
            "portfolio_max_gross_exposure": float(
                evidence.metrics.get("portfolio_max_gross_exposure", 0.0)
            ),
            "portfolio_max_turnover": float(evidence.metrics.get("portfolio_max_turnover", 0.0)),
        },
        "eligibility_passed": passed,
    }
    if feature_names is not None:
        row["feature_names"] = list(feature_names)
    return row


__all__ = [
    "CLASSICAL_RESEARCH_FEATURES",
    "build_linear_model_comparison",
    "xgboost_model_comparison_row",
]

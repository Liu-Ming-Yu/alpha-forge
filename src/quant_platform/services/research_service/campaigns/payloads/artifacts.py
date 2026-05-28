"""Walk-forward artifact payload builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.metrics.return_metrics import equity_curve

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.services.research_service.sampling.factory_models import (
        WalkForwardEvidence,
    )


def walk_forward_artifact_payloads(
    evidence: WalkForwardEvidence,
) -> dict[str, Mapping[str, object]]:
    """Build standard walk-forward artifact payloads without writing files."""
    payloads: dict[str, Mapping[str, object]] = {
        "fold_metrics.json": {"folds": list(evidence.folds)},
        "eligibility.json": {
            "passed": evidence.eligibility["passed"],
            "checks": evidence.eligibility["checks"],
        },
        "model_manifest.json": {
            "model_type": "linear_ranker_walk_forward",
            "model_version": evidence.model_version,
            "feature_set_version": evidence.feature_set_version,
            "weights": dict(evidence.selected_weights),
            "metrics": dict(evidence.metrics),
            "eligibility_passed": evidence.eligibility["passed"],
        },
        "run_summary.json": {
            "annualised_sharpe": evidence.metrics["slippage_adjusted_sharpe"],
            "total_return": evidence.metrics["total_return"],
            "max_drawdown": evidence.metrics["max_drawdown"],
            "gross_turnover": float(evidence.metrics.get("turnover_avg", 0.0)),
            "equity_curve": equity_curve(evidence.daily_returns),
            "bootstrap_ic_p05": float(evidence.metrics.get("bootstrap_ic_p05", 0.0)),
            "bootstrap_ic_p95": float(evidence.metrics.get("bootstrap_ic_p95", 0.0)),
            "top_minus_bottom_decile_ic": float(
                evidence.metrics.get("top_minus_bottom_decile_ic", 0.0)
            ),
            "feature_stability_avg": float(evidence.metrics.get("feature_stability_avg", 0.0)),
        },
        "execution_quality.json": {
            "aggregate": {
                "fill_rate": 1.0,
                "average_participation_pct": 0.0,
                "max_participation_pct": 0.0,
                "total_commission": "0",
                "total_slippage_cost": "modelled_in_returns",
                "average_implementation_shortfall_bps": float(evidence.slippage_bps_per_turnover)
                * float(evidence.metrics.get("turnover_avg", 0.0)),
                "slippage_bps_per_turnover": float(evidence.slippage_bps_per_turnover),
                "turnover_avg": float(evidence.metrics.get("turnover_avg", 0.0)),
            },
            "orders": [],
        },
        "ic_report.json": {
            "horizons": [1],
            "series": [
                {
                    "factor": "walk_forward_model",
                    "decay": {"1": evidence.metrics["oos_rolling_ic"]},
                }
            ],
            "bootstrap_ic_p05": float(evidence.metrics.get("bootstrap_ic_p05", 0.0)),
            "bootstrap_ic_p95": float(evidence.metrics.get("bootstrap_ic_p95", 0.0)),
            "top_minus_bottom_decile_ic": float(
                evidence.metrics.get("top_minus_bottom_decile_ic", 0.0)
            ),
        },
        "attribution.json": {
            "groups": {
                key: {bucket: dict(values) for bucket, values in groups.items()}
                for key, groups in evidence.attribution.items()
            },
        },
        "feature_stability.json": {
            "per_feature_mean_abs_change": dict(evidence.feature_stability),
            "average": float(evidence.metrics.get("feature_stability_avg", 0.0)),
        },
    }
    if evidence.portfolio_config:
        payloads["portfolio_config.json"] = dict(evidence.portfolio_config)
    if evidence.portfolio_diagnostics:
        payloads["portfolio_diagnostics.json"] = dict(evidence.portfolio_diagnostics)
    if evidence.drawdown_diagnostics:
        payloads["drawdown_diagnostics.json"] = dict(evidence.drawdown_diagnostics)
    return payloads


__all__ = ["walk_forward_artifact_payloads"]

"""Direction diagnostics for governed paper feature repair."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import FeatureExpectedSign
from quant_platform.services.research_service.feature_quality.audit.calculations import (
    feature_rows,
)
from quant_platform.services.research_service.feature_quality.audit.gate_evaluators import (
    evaluate_cost_gate,
    evaluate_economic_gate,
    evaluate_incremental_gate,
    evaluate_leakage_gate,
    evaluate_noise_gate,
    evaluate_stability_gate,
)
from quant_platform.services.research_service.feature_quality.audit.thresholds import (
    FeatureAuditThresholds,
)
from quant_platform.services.research_service.feature_quality.cards import load_feature_definition

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.core.domain.research import FeatureDefinition
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def build_feature_direction_diagnostics(
    *,
    samples: Sequence[SupervisedAlphaSample],
    feature_set_version: str,
    feature_card_dir: Path,
    slippage_bps_per_turnover: float,
    candidate_feature_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Evaluate every sampled feature under positive and negative orientation."""
    all_feature_names = sorted(
        {
            str(name)
            for sample in samples
            for name in sample.features
            if not str(name).startswith("_")
        }
    )
    feature_names = (
        sorted({str(name) for name in candidate_feature_names})
        if candidate_feature_names is not None
        else all_feature_names
    )
    rows = [
        _diagnose_feature(
            samples=samples,
            feature_name=feature_name,
            feature_set_version=feature_set_version,
            feature_card_dir=feature_card_dir,
            all_feature_names=all_feature_names,
            slippage_bps_per_turnover=slippage_bps_per_turnover,
        )
        for feature_name in feature_names
    ]
    return {
        "feature_set_version": feature_set_version,
        "feature_card_dir": str(feature_card_dir),
        "feature_count": len(rows),
        "features": rows,
        "missing_cards": [
            str(row["feature_name"]) for row in rows if bool(row.get("missing_card"))
        ],
    }


def _diagnose_feature(
    *,
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    feature_set_version: str,
    feature_card_dir: Path,
    all_feature_names: Sequence[str],
    slippage_bps_per_turnover: float,
) -> dict[str, object]:
    card_path = feature_card_dir / f"{feature_name}.json"
    if not card_path.is_file():
        return {
            "feature_name": feature_name,
            "missing_card": True,
            "blockers": [f"missing feature card: {card_path}"],
            "recommended_orientation": "none",
            "recommended_passed": False,
            "orientations": {},
        }

    feature = load_feature_definition(card_path)
    baseline = [name for name in all_feature_names if name != feature_name]
    orientations = {
        "positive": _evaluate_orientation(
            feature=dataclasses.replace(
                feature,
                expected_sign=FeatureExpectedSign.POSITIVE,
            ),
            samples=samples,
            feature_set_version=feature_set_version,
            baseline_features=baseline,
            slippage_bps_per_turnover=slippage_bps_per_turnover,
        ),
        "negative": _evaluate_orientation(
            feature=dataclasses.replace(
                feature,
                expected_sign=FeatureExpectedSign.NEGATIVE,
            ),
            samples=samples,
            feature_set_version=feature_set_version,
            baseline_features=baseline,
            slippage_bps_per_turnover=slippage_bps_per_turnover,
        ),
    }
    recommendation = _recommend_orientation(orientations)
    return {
        "feature_name": feature_name,
        "card": str(card_path),
        "card_expected_sign": feature.expected_sign.value,
        "feature_set_version": feature_set_version,
        "recommended_orientation": recommendation["orientation"],
        "recommended_passed": recommendation["passed"],
        "recommendation_reason": recommendation["reason"],
        "orientations": orientations,
    }


def _evaluate_orientation(
    *,
    feature: FeatureDefinition,
    samples: Sequence[SupervisedAlphaSample],
    feature_set_version: str,
    baseline_features: Sequence[str],
    slippage_bps_per_turnover: float,
) -> dict[str, object]:
    rows = feature_rows(samples, feature.name)
    if not rows:
        return {
            "passed": False,
            "failed_gates": ["missing_feature"],
            "blockers": [f"samples do not contain feature {feature.name!r}"],
            "metrics": {},
        }
    thresholds = FeatureAuditThresholds(min_daily_groups=252)
    reports = {
        "noise": evaluate_noise_gate(samples, feature.name, thresholds),
        "leakage": evaluate_leakage_gate(feature, samples, rows, thresholds, 17),
        "ic_stability": evaluate_stability_gate(feature, rows, thresholds, 17),
    }
    reports["economic_logic"] = evaluate_economic_gate(
        feature,
        reports["ic_stability"],
    )
    reports["cost"] = evaluate_cost_gate(
        feature,
        rows,
        thresholds,
        slippage_bps_per_turnover,
    )
    reports["incremental"] = evaluate_incremental_gate(
        feature,
        samples,
        thresholds,
        baseline_features,
    )
    metrics = _selected_metrics(reports)
    failed_gates = [name for name, report in reports.items() if not bool(report["passed"])]
    return {
        "passed": not failed_gates,
        "expected_sign": feature.expected_sign.value,
        "failed_gates": failed_gates,
        "blockers": [
            f"{name}: {blocker}"
            for name, report in reports.items()
            for blocker in _blockers(report)
        ],
        "metrics": metrics,
    }


def _selected_metrics(reports: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for gate_name in ("ic_stability", "cost", "incremental", "noise"):
        raw = reports[gate_name].get("metrics", {})
        if not isinstance(raw, Mapping):
            continue
        for name in (
            "ic_mean",
            "icir",
            "ic_negative_streak",
            "cost_net_mean_return",
            "incremental_delta_ic",
            "noise_mean_daily_unique_ratio",
        ):
            if name in raw:
                metrics[name] = float(raw[name])
    return metrics


def _blockers(report: Mapping[str, object]) -> tuple[str, ...]:
    raw = report.get("blockers", ())
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, Sequence):
        return ()
    return tuple(str(value) for value in raw)


def _recommend_orientation(
    orientations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    passing = [name for name, row in orientations.items() if bool(row.get("passed"))]
    if passing:
        best = max(passing, key=lambda name: _ic_mean(orientations[name]))
        return {
            "orientation": best,
            "passed": True,
            "reason": "orientation passes all diagnostic gates",
        }
    best = max(orientations, key=lambda name: _ic_mean(orientations[name]))
    return {
        "orientation": best,
        "passed": False,
        "reason": "no orientation passes all gates; selected highest signed IC",
    }


def _ic_mean(row: Mapping[str, object]) -> float:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return float("-inf")
    return float(metrics.get("ic_mean", float("-inf")))


__all__ = ["build_feature_direction_diagnostics"]

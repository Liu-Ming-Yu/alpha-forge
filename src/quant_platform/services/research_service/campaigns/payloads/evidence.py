"""Replay and eligibility evidence payload builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.features.admission import ordered_feature_schema_hash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.services.research_service.sampling.factory_models import (
        WalkForwardEvidence,
    )


def replay_context_payload(
    evidence: WalkForwardEvidence,
    *,
    samples_path: Path,
    campaign_context: Mapping[str, object],
) -> dict[str, object]:
    sample_build = mapping_payload(campaign_context.get("sample_build"))
    return {
        "sample_start": sample_build.get("sample_start"),
        "sample_end": sample_build.get("sample_end"),
        "universe": sample_build.get("universe") or sample_build.get("contracts_file"),
        "samples_path": str(samples_path),
        "feature_set_version": evidence.feature_set_version,
        "feature_schema_hash": ordered_feature_schema_hash(tuple(evidence.selected_weights)),
        "model_version": evidence.model_version,
        "date_policy": sample_build.get("date_policy"),
        "horizon_days": sample_build.get("horizon_days"),
        "bar_seconds": sample_build.get("bar_seconds"),
        "max_feature_age_days": sample_build.get("max_feature_age_days"),
    }


def campaign_evidence_payload(
    evidence: WalkForwardEvidence,
    *,
    passed: bool,
    campaign_context: Mapping[str, object],
) -> dict[str, object]:
    sample_build = mapping_payload(campaign_context.get("sample_build"))
    return {
        "sample_start": sample_build.get("sample_start"),
        "sample_end": sample_build.get("sample_end"),
        "universe": sample_build.get("universe") or sample_build.get("contracts_file"),
        "horizon_days": sample_build.get("horizon_days"),
        "bar_seconds": sample_build.get("bar_seconds"),
        "max_feature_age_days": sample_build.get("max_feature_age_days"),
        "feature_set_version": evidence.feature_set_version,
        "feature_schema_hash": ordered_feature_schema_hash(tuple(evidence.selected_weights)),
        "model_version": evidence.model_version,
        "walk_forward_folds": list(evidence.folds),
        "oos_ic_by_fold": [
            {
                "fold_index": fold.get("fold_index"),
                "mean_ic": fold.get("mean_ic"),
            }
            for fold in evidence.folds
        ],
        "rolling_ic": float(evidence.metrics.get("oos_rolling_ic", 0.0)),
        "negative_streak": int(evidence.metrics.get("fold_negative_ic_streak", 0.0)),
        "max_drawdown": float(evidence.metrics.get("max_drawdown", 0.0)),
        "turnover": float(evidence.metrics.get("turnover_avg", 0.0)),
        "portfolio_config": dict(evidence.portfolio_config),
        "portfolio_diagnostics": dict(evidence.portfolio_diagnostics),
        "drawdown_diagnostics": dict(evidence.drawdown_diagnostics),
        "slippage_adjusted_sharpe": float(evidence.metrics.get("slippage_adjusted_sharpe", 0.0)),
        "prediction_evidence": campaign_context.get("prediction_evidence"),
        "blockers": eligibility_blockers(evidence.eligibility),
        "passed": passed,
    }


def mapping_payload(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def require_non_null_campaign_evidence(payload: Mapping[str, object]) -> None:
    required_by_section = {
        "sample_build": (
            "sample_start",
            "sample_end",
            "universe",
            "horizon_days",
            "max_feature_age_days",
        ),
        "replay_context": (
            "sample_start",
            "sample_end",
            "universe",
            "horizon_days",
            "max_feature_age_days",
        ),
        "campaign_evidence": (
            "sample_start",
            "sample_end",
            "universe",
            "horizon_days",
            "max_feature_age_days",
            "prediction_evidence",
        ),
    }
    missing: list[str] = []
    for section_name, fields in required_by_section.items():
        section = mapping_payload(payload.get(section_name))
        for field in fields:
            if is_missing(section.get(field)):
                missing.append(f"{section_name}.{field}")
    if is_missing(payload.get("prediction_evidence")):
        missing.append("prediction_evidence")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"campaign manifest missing required non-null fields: {joined}")


def is_missing(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def eligibility_blockers(eligibility_payload: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    checks = eligibility_payload.get("checks")
    if not isinstance(checks, list):
        return blockers
    for check in checks:
        if isinstance(check, dict) and not bool(check.get("passed", True)):
            blockers.append(str(check.get("name", "unknown")))
    return blockers


__all__ = [
    "campaign_evidence_payload",
    "eligibility_blockers",
    "mapping_payload",
    "replay_context_payload",
    "require_non_null_campaign_evidence",
]

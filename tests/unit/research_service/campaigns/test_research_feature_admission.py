from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from quant_platform.config import PlatformSettings
from quant_platform.services.research_service.campaigns.evaluation.feature_admission import (
    annotate_feature_audits,
    resolve_campaign_feature_admission,
    supervised_to_boosting_samples,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from pathlib import Path


def _sample(features: dict[str, float]) -> SupervisedAlphaSample:
    return SupervisedAlphaSample(
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        instrument_id=uuid.uuid4(),
        features=features,
        forward_return=0.01,
    )


def test_passing_feature_admission_quarantines_failed_features() -> None:
    samples = [_sample({"alpha": 1.0, "beta": -1.0, "gamma": 0.5})]
    audits = [
        {"feature_name": "alpha", "passed": True},
        {"feature_name": "beta", "passed": False},
        {"feature_name": "gamma", "passed": True},
    ]

    admission = resolve_campaign_feature_admission(
        samples=samples,
        feature_audits=audits,
        audit_mode="paper",
        feature_admission="passing",
        min_admitted_features=2,
    )
    annotated = annotate_feature_audits(audits, admission)
    boosting = supervised_to_boosting_samples(samples, admission.admitted_features)

    assert admission.passed is True
    assert admission.admitted_features == ("alpha", "gamma")
    assert admission.quarantined_features == ("beta",)
    assert annotated[1]["admission"] == "quarantined"
    assert boosting[0].features == {"alpha": 1.0, "gamma": 0.5}


def test_all_feature_admission_preserves_all_or_nothing_failure() -> None:
    samples = [_sample({"alpha": 1.0, "beta": -1.0, "gamma": 0.5})]

    admission = resolve_campaign_feature_admission(
        samples=samples,
        feature_audits=[
            {"feature_name": "alpha", "passed": True},
            {"feature_name": "beta", "passed": False},
            {"feature_name": "gamma", "passed": True},
        ],
        audit_mode="paper",
        feature_admission="all",
        min_admitted_features=2,
    )

    assert admission.passed is False
    assert admission.admitted_features == ()
    assert admission.quarantined_features == ("beta",)
    assert admission.blockers == ("feature audit failed or missing for 'beta'",)


def test_all_feature_admission_enforces_minimum_feature_count() -> None:
    samples = [_sample({"alpha": 1.0, "beta": -1.0})]

    admission = resolve_campaign_feature_admission(
        samples=samples,
        feature_audits=[
            {"feature_name": "alpha", "passed": True},
            {"feature_name": "beta", "passed": True},
        ],
        audit_mode="paper",
        feature_admission="all",
        min_admitted_features=3,
    )

    assert admission.passed is False
    assert admission.admitted_features == ()
    assert admission.quarantined_features == ()
    assert admission.blockers == ("admitted feature count 2 < 3",)


def test_passing_feature_admission_fails_when_too_few_features_pass() -> None:
    samples = [_sample({"alpha": 1.0, "beta": -1.0, "gamma": 0.5})]

    admission = resolve_campaign_feature_admission(
        samples=samples,
        feature_audits=[{"feature_name": "alpha", "passed": True}],
        audit_mode="paper",
        feature_admission="passing",
        min_admitted_features=2,
    )

    assert admission.passed is False
    assert admission.admitted_features == ("alpha",)
    assert admission.quarantined_features == ("beta", "gamma")


def test_paper_admission_can_scope_to_candidate_features_only() -> None:
    samples = [
        _sample(
            {
                "classical_baseline": 0.2,
                "catalyst_sentiment_21d_decay": 0.8,
                "earnings_quality_21d_decay": 0.5,
            }
        )
    ]

    admission = resolve_campaign_feature_admission(
        samples=samples,
        feature_audits=[
            {"feature_name": "catalyst_sentiment_21d_decay", "passed": True},
            {"feature_name": "earnings_quality_21d_decay", "passed": False},
        ],
        audit_mode="paper",
        feature_admission="passing",
        min_admitted_features=1,
        candidate_feature_names=(
            "catalyst_sentiment_21d_decay",
            "earnings_quality_21d_decay",
        ),
    )

    assert admission.admitted_features == ("catalyst_sentiment_21d_decay",)
    assert admission.quarantined_features == ("earnings_quality_21d_decay",)
    assert admission.audited_features == (
        "catalyst_sentiment_21d_decay",
        "earnings_quality_21d_decay",
    )


@pytest.mark.asyncio
async def test_campaign_admission_context_records_v12_attribution_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_platform.research.campaign.admission as admission_ops

    async def _audits(**_kwargs: Any) -> list[dict[str, object]]:
        return [{"feature_name": "alpha", "passed": True}]

    monkeypatch.setattr(admission_ops, "_run_campaign_feature_audits", _audits)
    diagnostics = tmp_path / "feature_direction_diagnostics.json"
    attribution = tmp_path / "feature_failure_attribution.json"

    resolution = await admission_ops.resolve_campaign_admission(
        settings=PlatformSettings(_env_file=None),
        samples=[_sample({"alpha": 1.0})],
        sample_build={"samples": 1},
        output_root=tmp_path,
        sample_slug="v12",
        feature_set_version="paper-alpha-catalyst-v10",
        horizon_days=21,
        slippage_bps_per_turnover=10.0,
        feature_audit_mode="paper",
        feature_card_dir=tmp_path,
        feature_admission="passing",
        min_admitted_features=1,
        date_policy="nyse-sessions",
        feature_diagnostics_path=diagnostics,
        feature_attribution_path=attribution,
    )

    assert resolution.admission.passed is True
    assert resolution.campaign_context["feature_direction_diagnostics"] == str(diagnostics)
    assert resolution.campaign_context["feature_failure_attribution"] == str(attribution)

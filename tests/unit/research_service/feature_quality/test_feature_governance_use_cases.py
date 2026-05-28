from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from quant_platform.application.features.governance import (
    FeatureAuditUseCase,
    assert_feature_audit,
    feature_audit_status,
    retire_feature_audit,
)
from quant_platform.application.features.governance_campaign import audit_campaign_features
from quant_platform.application.features.governance_payloads import (
    csv_names,
    dumps_payload,
    feature_state_meets_minimum,
)
from quant_platform.application.features.governance_requests import (
    CampaignFeatureAuditRequest,
    FeatureAuditAssertRequest,
    FeatureAuditRetireRequest,
    FeatureAuditRunRequest,
    FeatureAuditStatusRequest,
)
from quant_platform.core.domain.research import FeatureAuditResult, FeatureProductionState
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from pathlib import Path

AS_OF = datetime(2026, 1, 2, tzinfo=UTC)


def _audit_result(
    *,
    feature_name: str = "quality_alpha",
    status: FeatureProductionState = FeatureProductionState.PAPER,
    passed: bool = True,
    blockers: tuple[str, ...] = (),
) -> FeatureAuditResult:
    return FeatureAuditResult(
        audit_id=uuid.uuid4(),
        feature_name=feature_name,
        feature_version="v1",
        feature_set_version="features-v1",
        as_of=AS_OF,
        sample_start=datetime(2025, 1, 1, tzinfo=UTC),
        sample_end=datetime(2025, 12, 31, tzinfo=UTC),
        status=status,
        passed=passed,
        metrics={"ic": 0.05},
        gate_results={"noise": True},
        artifact_uri="artifact://feature_audit_manifest.json",
        schema_hash="abc",
        code_commit="deadbeef",
        blockers=blockers,
    )


@pytest.mark.asyncio
async def test_feature_audit_status_reads_latest_repository_row(tmp_path: Path) -> None:
    repo = _FeatureAuditRepo([_audit_result()])

    result = await feature_audit_status(
        request=FeatureAuditStatusRequest(
            feature_name="quality_alpha",
            feature_version="v1",
            limit=10,
            output_root=None,
        ),
        object_store_root=tmp_path,
        repository=repo,
        artifact_store=None,
    )

    audits = cast("list[dict[str, Any]]", result.payload["audits"])
    assert result.payload["count"] == 1
    assert audits[0]["feature_name"] == "quality_alpha"
    assert repo.latest_calls == [("quality_alpha", "v1")]


@pytest.mark.asyncio
async def test_feature_audit_status_filters_artifact_manifests(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "quality_alpha",
        "v1",
        {
            "generated_at": "2026-01-03T00:00:00+00:00",
            "feature": {"name": "quality_alpha", "version": "v1"},
            "passed": True,
        },
    )
    _write_manifest(
        tmp_path,
        "noise_alpha",
        "v1",
        {
            "generated_at": "2026-01-02T00:00:00+00:00",
            "feature": {"name": "noise_alpha", "version": "v1"},
            "passed": False,
        },
    )

    result = await feature_audit_status(
        request=FeatureAuditStatusRequest(
            feature_name="quality_alpha",
            feature_version="v1",
            limit=10,
            output_root=None,
        ),
        object_store_root=tmp_path,
        repository=None,
        artifact_store=None,
    )

    audits = cast("list[dict[str, Any]]", result.payload["audits"])
    assert result.payload["count"] == 1
    assert audits[0]["feature"]["name"] == "quality_alpha"


@pytest.mark.asyncio
async def test_assert_feature_audit_blocks_manifest_below_required_state(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "quality_alpha",
        "v1",
        {
            "generated_at": "2026-01-03T00:00:00+00:00",
            "feature": {"name": "quality_alpha", "version": "v1", "state": "shadow"},
            "passed": True,
            "blockers": [],
        },
    )

    result = await assert_feature_audit(
        request=FeatureAuditAssertRequest(
            manifest=manifest,
            feature_name=None,
            feature_version=None,
            minimum_state="paper",
        ),
        repository=None,
        artifact_store=None,
    )

    assert result.passed is False
    assert "below required" in cast("list[str]", result.payload["blockers"])[0]


@pytest.mark.asyncio
async def test_assert_feature_audit_accepts_repository_latest_row(tmp_path: Path) -> None:
    repo = _FeatureAuditRepo([_audit_result()])

    result = await assert_feature_audit(
        request=FeatureAuditAssertRequest(
            manifest=None,
            feature_name="quality_alpha",
            feature_version="v1",
            minimum_state="paper",
        ),
        repository=repo,
        artifact_store=None,
    )

    assert result.passed is True
    assert result.payload["feature_name"] == "quality_alpha"


@pytest.mark.asyncio
async def test_retire_feature_audit_persists_retired_marker(tmp_path: Path) -> None:
    repo = _FeatureAuditRepo([])

    result = await retire_feature_audit(
        request=FeatureAuditRetireRequest(
            feature_name="old_alpha",
            feature_version="v2",
            feature_set_version="features-v1",
            reason="superseded by lower-cost signal",
        ),
        repository=repo,
    )

    assert result.passed is False
    assert repo.saved[-1].status == FeatureProductionState.RETIRED
    assert repo.saved[-1].blockers == ("superseded by lower-cost signal",)


@pytest.mark.asyncio
async def test_feature_audit_use_case_delegates_admin_methods(tmp_path: Path) -> None:
    repo = _FeatureAuditRepo([_audit_result()])
    use_case = FeatureAuditUseCase(object_store_root=tmp_path, repository=repo)

    status = await use_case.status(
        FeatureAuditStatusRequest(
            feature_name="quality_alpha",
            feature_version="v1",
            limit=10,
            output_root=None,
        )
    )
    asserted = await use_case.assert_latest(
        FeatureAuditAssertRequest(
            manifest=None,
            feature_name="quality_alpha",
            feature_version="v1",
            minimum_state="paper",
        )
    )
    retired = await use_case.retire(
        FeatureAuditRetireRequest(
            feature_name="old_alpha",
            feature_version="v1",
            feature_set_version="features-v1",
            reason="no longer economical",
        )
    )

    assert status.payload["count"] == 1
    assert asserted.passed is True
    assert retired.passed is False


@pytest.mark.asyncio
async def test_feature_audit_use_case_delegates_campaign_off_mode(tmp_path: Path) -> None:
    use_case = FeatureAuditUseCase(object_store_root=tmp_path)

    rows = await use_case.audit_campaign_features(
        CampaignFeatureAuditRequest(
            samples=(),
            feature_set_version="features-v1",
            horizon_days=21,
            slippage_bps_per_turnover=0.0,
            mode="off",
            feature_card_dir=None,
        )
    )

    assert rows == []


@pytest.mark.asyncio
async def test_audit_campaign_features_reports_missing_cards_in_paper_mode(
    tmp_path: Path,
) -> None:
    rows = await audit_campaign_features(
        request=CampaignFeatureAuditRequest(
            samples=(
                SupervisedAlphaSample(
                    as_of=AS_OF,
                    instrument_id=uuid.uuid4(),
                    features={"quality_alpha": 1.0, "_reserved": 2.0},
                    forward_return=0.02,
                ),
            ),
            feature_set_version="features-v1",
            horizon_days=21,
            slippage_bps_per_turnover=0.0,
            mode="paper",
            feature_card_dir=tmp_path / "cards",
        ),
        object_store_root=tmp_path,
        repository=None,
        artifact_store=None,
    )

    assert rows == [
        {
            "feature_name": "quality_alpha",
            "passed": False,
            "blockers": [f"missing feature card: {tmp_path / 'cards' / 'quality_alpha.json'}"],
        }
    ]


@pytest.mark.asyncio
async def test_audit_campaign_features_honors_candidate_scope_in_paper_mode(
    tmp_path: Path,
) -> None:
    rows = await audit_campaign_features(
        request=CampaignFeatureAuditRequest(
            samples=(
                SupervisedAlphaSample(
                    as_of=AS_OF,
                    instrument_id=uuid.uuid4(),
                    features={"momentum_1m": 1.0, "text_sentiment_21d_decay": 0.5},
                    forward_return=0.02,
                ),
            ),
            feature_set_version="paper-alpha-text-v1",
            horizon_days=21,
            slippage_bps_per_turnover=0.0,
            mode="paper",
            feature_card_dir=tmp_path / "cards",
            candidate_feature_names=("text_sentiment_21d_decay",),
        ),
        object_store_root=tmp_path,
        repository=None,
        artifact_store=None,
    )

    assert [row["feature_name"] for row in rows] == ["text_sentiment_21d_decay"]


@pytest.mark.asyncio
async def test_feature_audit_use_case_builds_samples_when_missing_samples_path(
    tmp_path: Path,
) -> None:
    async def _sample_builder(request: FeatureAuditRunRequest, output: Path) -> Path:
        assert request.contracts_file == "contracts.json"
        assert output.name == "samples.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("[]", encoding="utf-8")
        return output

    use_case = FeatureAuditUseCase(
        object_store_root=tmp_path,
        sample_builder=_sample_builder,
    )

    path = await use_case._resolve_samples_path(  # noqa: SLF001 - focused use-case seam.
        FeatureAuditRunRequest(
            feature_card=tmp_path / "feature_card.json",
            samples=None,
            contracts_file="contracts.json",
            start=AS_OF,
            end=AS_OF,
            feature_set_version="features-v1",
            horizon_days=21,
            bar_seconds=86400,
            max_feature_age_days=3,
            output_root=None,
            baseline_features="",
            slippage_bps_per_turnover=0.0,
            min_daily_groups=60,
            min_coverage=0.95,
            min_oos_ic=0.02,
            min_icir=0.0,
            max_negative_ic_streak=5,
            max_turnover=2.0,
            persist=False,
        )
    )

    assert path.is_file()


def test_feature_governance_payload_helpers() -> None:
    assert feature_state_meets_minimum("live", "paper") is True
    assert feature_state_meets_minimum("shadow", "paper") is False
    assert csv_names("alpha, beta,, ") == ("alpha", "beta")
    assert dumps_payload({"as_of": AS_OF, "id": uuid.UUID(int=1)})


def _write_manifest(
    root: Path,
    feature_name: str,
    feature_version: str,
    payload: dict[str, object],
) -> Path:
    path = (
        root
        / "research"
        / "feature_audits"
        / feature_name
        / feature_version
        / str(uuid.uuid4())
        / "feature_audit_manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _FeatureAuditRepo:
    def __init__(self, rows: list[FeatureAuditResult]) -> None:
        self._rows = rows
        self.saved: list[FeatureAuditResult] = []
        self.latest_calls: list[tuple[str, str | None]] = []

    async def save_feature_audit(self, result: FeatureAuditResult) -> None:
        self.saved.append(result)
        self._rows.append(result)

    async def latest_feature_audit(
        self,
        feature_name: str,
        feature_version: str | None = None,
    ) -> FeatureAuditResult | None:
        self.latest_calls.append((feature_name, feature_version))
        for row in self._rows:
            if row.feature_name == feature_name and (
                feature_version is None or row.feature_version == feature_version
            ):
                return row
        return None

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 100,
    ) -> list[FeatureAuditResult]:
        rows = [
            row for row in self._rows if feature_name is None or row.feature_name == feature_name
        ]
        return rows[:limit]

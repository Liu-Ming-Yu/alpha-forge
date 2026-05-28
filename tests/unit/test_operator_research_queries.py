from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from quant_platform.application.operator.queries import OperatorResearchQueryService
from quant_platform.core.domain.research import FeatureAuditResult, FeatureProductionState

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_operator_research_queries_list_and_read_campaign_manifests(
    tmp_path: Path,
) -> None:
    run_id = str(uuid.uuid4())
    manifest = {
        "run_id": run_id,
        "created_at": "2026-01-02T00:00:00+00:00",
        "metrics": {"oos_ic": 0.07},
    }
    _write_json(
        tmp_path / "research" / "walk_forward" / run_id / "campaign_manifest.json",
        manifest,
    )
    service = OperatorResearchQueryService(object_store_root=tmp_path)

    listed = await service.list_research_campaigns(limit=5)
    loaded = await service.read_research_campaign(run_id)

    assert listed["count"] == 1
    campaigns = cast("list[dict[str, Any]]", listed["campaigns"])
    assert campaigns[0]["run_id"] == run_id
    assert loaded == manifest


@pytest.mark.asyncio
async def test_operator_research_queries_returns_none_for_missing_campaign(
    tmp_path: Path,
) -> None:
    service = OperatorResearchQueryService(object_store_root=tmp_path)

    assert await service.read_research_campaign(str(uuid.uuid4())) is None


@pytest.mark.asyncio
async def test_operator_research_queries_filters_feature_audits_from_artifacts(
    tmp_path: Path,
) -> None:
    wanted = {
        "generated_at": "2026-01-03T00:00:00+00:00",
        "feature": {"name": "quality_alpha", "version": "v1"},
        "passed": True,
    }
    ignored = {
        "generated_at": "2026-01-02T00:00:00+00:00",
        "feature": {"name": "noise_alpha", "version": "v1"},
        "passed": False,
    }
    _write_json(
        tmp_path
        / "research"
        / "feature_audits"
        / "quality_alpha"
        / "v1"
        / str(uuid.uuid4())
        / "feature_audit_manifest.json",
        wanted,
    )
    _write_json(
        tmp_path
        / "research"
        / "feature_audits"
        / "noise_alpha"
        / "v1"
        / str(uuid.uuid4())
        / "feature_audit_manifest.json",
        ignored,
    )
    service = OperatorResearchQueryService(object_store_root=tmp_path)

    payload = await service.list_feature_audits(feature_name="quality_alpha")

    assert payload["count"] == 1
    audits = cast("list[dict[str, Any]]", payload["audits"])
    assert audits[0]["feature"]["name"] == "quality_alpha"


@pytest.mark.asyncio
async def test_operator_research_queries_prefers_repository_feature_audits(
    tmp_path: Path,
) -> None:
    repo = _FeatureAuditRepo(
        [
            FeatureAuditResult(
                audit_id=uuid.uuid4(),
                feature_name="quality_alpha",
                feature_version="v1",
                feature_set_version="features-v1",
                as_of=datetime(2026, 1, 2, tzinfo=UTC),
                sample_start=datetime(2025, 1, 1, tzinfo=UTC),
                sample_end=datetime(2025, 12, 31, tzinfo=UTC),
                status=FeatureProductionState.PAPER,
                passed=True,
                metrics={"ic": 0.05},
                gate_results={"noise": True},
                artifact_uri="artifact://audit.json",
                schema_hash="abc",
                code_commit="deadbeef",
            )
        ]
    )
    service = OperatorResearchQueryService(
        object_store_root=tmp_path,
        feature_audit_repository=repo,
    )

    payload = await service.list_feature_audits(feature_name="quality_alpha", limit=10)

    assert repo.last_feature_name == "quality_alpha"
    assert repo.last_limit == 10
    assert payload["count"] == 1
    audits = cast("list[dict[str, Any]]", payload["audits"])
    assert audits[0]["feature_name"] == "quality_alpha"
    assert audits[0]["status"] == "paper"


class _FeatureAuditRepo:
    def __init__(self, rows: list[FeatureAuditResult]) -> None:
        self._rows = rows
        self.last_feature_name: str | None = None
        self.last_limit: int | None = None

    async def save_feature_audit(self, result: FeatureAuditResult) -> None:
        self._rows.append(result)

    async def latest_feature_audit(
        self,
        feature_name: str,
        feature_version: str | None = None,
    ) -> FeatureAuditResult | None:
        del feature_version
        for row in self._rows:
            if row.feature_name == feature_name:
                return row
        return None

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 100,
    ) -> list[FeatureAuditResult]:
        self.last_feature_name = feature_name
        self.last_limit = limit
        rows = [
            row for row in self._rows if feature_name is None or row.feature_name == feature_name
        ]
        return rows[:limit]

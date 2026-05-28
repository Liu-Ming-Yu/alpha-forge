"""Feature-governance application use cases.

The CLI and research-campaign entrypoints delegate here so feature admission is
no longer owned by argparse handlers.  Concrete repository construction remains
in bootstrap.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.features.governance_admin import (
    assert_feature_audit,
    feature_audit_status,
    retire_feature_audit,
)
from quant_platform.application.features.governance_campaign import (
    audit_campaign_features as _audit_campaign_features,
)
from quant_platform.application.features.governance_payloads import (
    csv_names as _csv_names,
)
from quant_platform.application.features.governance_payloads import (
    dumps_payload,
    feature_audit_result_payload,
    feature_state_meets_minimum,
)
from quant_platform.application.features.governance_payloads import (
    manifest_path as _manifest_path,
)
from quant_platform.application.features.governance_requests import (
    CampaignFeatureAuditRequest,
    FeatureAuditAssertRequest,
    FeatureAuditCommandResult,
    FeatureAuditRetireRequest,
    FeatureAuditRunRequest,
    FeatureAuditStatusRequest,
)
from quant_platform.services.research_service.feature_quality.audit.feature_audit import (
    FeatureAuditRunner,
    FeatureAuditThresholds,
    load_feature_definition,
)
from quant_platform.services.research_service.sampling.factory import load_supervised_samples

if TYPE_CHECKING:
    from quant_platform.core.contracts import ArtifactStore, FeatureAuditRepository

SampleBuilder = Callable[["FeatureAuditRunRequest", Path], Awaitable[Path]]

__all__ = [
    "CampaignFeatureAuditRequest",
    "FeatureAuditAssertRequest",
    "FeatureAuditCommandResult",
    "FeatureAuditRetireRequest",
    "FeatureAuditRunRequest",
    "FeatureAuditStatusRequest",
    "FeatureAuditUseCase",
    "dumps_payload",
    "feature_audit_result_payload",
    "feature_state_meets_minimum",
]


class FeatureAuditUseCase:
    """Application service for feature audit admission workflows."""

    def __init__(
        self,
        *,
        object_store_root: Path,
        repository: FeatureAuditRepository | None = None,
        sample_builder: SampleBuilder | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._object_store_root = object_store_root
        self._repository = repository
        self._sample_builder = sample_builder
        self._artifact_store = artifact_store

    async def run(self, request: FeatureAuditRunRequest) -> FeatureAuditCommandResult:
        samples_path = await self._resolve_samples_path(request)
        output_root = request.output_root or self._object_store_root
        feature = load_feature_definition(request.feature_card)
        manifest = FeatureAuditRunner(
            thresholds=FeatureAuditThresholds(
                min_daily_groups=request.min_daily_groups,
                min_coverage=request.min_coverage,
                min_oos_ic=request.min_oos_ic,
                min_icir=request.min_icir,
                max_negative_ic_streak=request.max_negative_ic_streak,
                max_turnover=request.max_turnover,
            ),
            slippage_bps_per_turnover=float(request.slippage_bps_per_turnover),
            baseline_features=_csv_names(request.baseline_features),
            artifact_store=self._artifact_store,
        ).run(
            feature=feature,
            samples=load_supervised_samples(samples_path),
            feature_set_version=request.feature_set_version,
            output_root=output_root,
        )
        manifest_path = _manifest_path(output_root, feature, manifest.audit_id)
        result = manifest.to_result(str(manifest_path))
        if request.persist or self._repository is not None:
            if self._repository is None:
                raise ValueError("--persist requires QP__STORAGE__POSTGRES_DSN")
            await self._repository.save_feature_audit(result)
        payload = feature_audit_result_payload(result)
        payload["manifest"] = str(manifest_path)
        return FeatureAuditCommandResult(payload=payload, passed=result.passed)

    async def status(self, request: FeatureAuditStatusRequest) -> FeatureAuditCommandResult:
        return await feature_audit_status(
            request=request,
            object_store_root=self._object_store_root,
            repository=self._repository,
            artifact_store=self._artifact_store,
        )

    async def assert_latest(
        self,
        request: FeatureAuditAssertRequest,
    ) -> FeatureAuditCommandResult:
        return await assert_feature_audit(
            request=request,
            repository=self._repository,
            artifact_store=self._artifact_store,
        )

    async def retire(self, request: FeatureAuditRetireRequest) -> FeatureAuditCommandResult:
        return await retire_feature_audit(
            request=request,
            repository=self._repository,
        )

    async def audit_campaign_features(
        self,
        request: CampaignFeatureAuditRequest,
    ) -> list[dict[str, object]]:
        return await _audit_campaign_features(
            request=request,
            object_store_root=self._object_store_root,
            repository=self._repository,
            artifact_store=self._artifact_store,
        )

    async def _resolve_samples_path(self, request: FeatureAuditRunRequest) -> Path:
        if request.samples is not None:
            return request.samples
        if not (request.contracts_file and request.start and request.end):
            raise ValueError(
                "features audit run requires either --samples or "
                "--contracts-file plus --start/--end"
            )
        if self._sample_builder is None:
            raise ValueError("feature audit sample builder is not configured")
        output = (
            self._object_store_root
            / "research"
            / "feature_audits"
            / "_inputs"
            / f"{request.feature_set_version}_{request.start.date()}_{request.end.date()}"
            / "samples.json"
        )
        return await self._sample_builder(request, output)

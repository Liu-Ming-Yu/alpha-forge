"""Operator-facing read use cases.

These query services keep FastAPI handlers away from object-store path
conventions and repository construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.features.governance import feature_audit_result_payload
from quant_platform.application.operator.payload_coercion import optional_mapping
from quant_platform.application.research.evidence import (
    list_campaign_evidence,
    list_feature_audit_manifests,
    read_campaign_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.core.contracts import FeatureAuditRepository


class OperatorResearchQueryService:
    """Read-only research evidence queries for operator surfaces."""

    def __init__(
        self,
        *,
        object_store_root: Path,
        feature_audit_repository: FeatureAuditRepository | None = None,
    ) -> None:
        self._object_store_root = object_store_root
        self._feature_audit_repository = feature_audit_repository

    async def list_research_campaigns(self, *, limit: int = 20) -> dict[str, object]:
        rows = list_campaign_evidence(self._object_store_root, limit=max(1, limit))
        return {"campaigns": rows, "count": len(rows)}

    async def read_research_campaign(self, run_id: str) -> dict[str, object] | None:
        return read_campaign_evidence(self._object_store_root, run_id)

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 50,
    ) -> dict[str, object]:
        capped_limit = max(1, min(limit, 200))
        if self._feature_audit_repository is not None:
            audit_rows = await self._feature_audit_repository.list_feature_audits(
                feature_name=feature_name,
                limit=capped_limit,
            )
            audits = [feature_audit_result_payload(row) for row in audit_rows]
            return {"audits": audits, "count": len(audits)}

        manifest_rows = list_feature_audit_manifests(self._object_store_root, limit=capped_limit)
        if feature_name:
            manifest_rows = [
                row
                for row in manifest_rows
                if optional_mapping(row.get("feature"), name="feature").get("name") == feature_name
            ]
        return {"audits": manifest_rows, "count": len(manifest_rows)}

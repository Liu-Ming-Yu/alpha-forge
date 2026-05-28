"""Application-level readers for research evidence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.services.research_service.feature_quality.audit.feature_audit import (
    list_feature_audit_manifests,
)
from quant_platform.services.research_service.intraday.backtesting.backtest import (
    assert_backtest_evidence,
)
from quant_platform.services.research_service.sampling.factory import (
    list_campaign_manifests,
    read_campaign_manifest,
    walk_forward_object_root,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class CampaignManifestEvidence:
    """Latest campaign manifest evidence found under the object store."""

    root: Path
    path: Path | None
    payload: Mapping[str, object] | None


def campaign_manifest_root(object_store_root: Path | str) -> Path:
    """Return the canonical walk-forward campaign artifact root."""
    return walk_forward_object_root(Path(object_store_root))


def list_campaign_evidence(
    object_store_root: Path | str,
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    """List campaign manifests newest-first from the object store."""
    return list_campaign_manifests(campaign_manifest_root(object_store_root), limit=limit)


def read_campaign_evidence(
    object_store_root: Path | str,
    run_id: str,
) -> dict[str, object] | None:
    """Read one campaign manifest by run ID."""
    path = campaign_manifest_root(object_store_root) / run_id / "campaign_manifest.json"
    if not path.is_file():
        return None
    return read_campaign_manifest(path)


def latest_campaign_manifest_evidence(
    object_store_root: Path | str,
) -> CampaignManifestEvidence:
    """Return the latest campaign manifest payload and path when available."""
    root = campaign_manifest_root(object_store_root)
    if not root.exists():
        return CampaignManifestEvidence(root=root, path=None, payload=None)
    manifests = list_campaign_manifests(root, limit=1)
    if not manifests:
        return CampaignManifestEvidence(root=root, path=None, payload=None)
    payload = manifests[0]
    artifact_root = payload.get("artifact_root")
    path = Path(str(artifact_root)) / "campaign_manifest.json" if artifact_root else None
    return CampaignManifestEvidence(root=root, path=path, payload=payload)


def validate_backtest_evidence_manifest(path: Path | str) -> None:
    """Validate an intraday backtest evidence manifest."""
    assert_backtest_evidence(Path(path))


__all__ = [
    "CampaignManifestEvidence",
    "campaign_manifest_root",
    "latest_campaign_manifest_evidence",
    "list_campaign_evidence",
    "list_feature_audit_manifests",
    "read_campaign_evidence",
    "validate_backtest_evidence_manifest",
]

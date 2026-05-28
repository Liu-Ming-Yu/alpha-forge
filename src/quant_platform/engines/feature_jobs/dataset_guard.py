"""Feature dataset admission checks used by proposal generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.services.data_service.reference.contract_master import ContractMaster


class FeatureDatasetSession(Protocol):
    contract_master: ContractMaster
    feature_repo: FeatureRepository


async def load_required_feature_dataset_id(
    session: FeatureDatasetSession,
    *,
    settings: PlatformSettings,
    feature_set_version: str,
    as_of: datetime,
) -> uuid.UUID | None:
    """Load and freshness-check the current feature dataset when V2 requires it."""
    if not getattr(settings.v2, "require_feature_datasets", False):
        return None

    dataset_catalog = getattr(session, "dataset_catalog", None)
    if dataset_catalog is None:
        return None

    from quant_platform.services.research_service.feature_quality.snapshot import (
        load_feature_snapshot,
    )

    instrument_ids = [i.instrument_id for i in session.contract_master.list_active()]
    snapshot = await load_feature_snapshot(
        dataset_catalog=dataset_catalog,
        feature_repo=session.feature_repo,
        instrument_ids=instrument_ids,
        feature_set_version=feature_set_version,
        as_of=as_of,
    )
    age_seconds = (as_of - snapshot.as_of).total_seconds()
    max_age = settings.v2.max_feature_age_seconds
    if age_seconds > max_age:
        raise RuntimeError(
            f"feature snapshot is stale: age={age_seconds:.0f}s "
            f"exceeds max_feature_age_seconds={max_age}"
        )
    return snapshot.dataset.dataset_id

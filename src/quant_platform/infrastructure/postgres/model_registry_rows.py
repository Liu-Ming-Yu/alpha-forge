"""Row mappers for the Postgres model registry adapter."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from quant_platform.services.research_service.modeling.registry.model_registry import (
    RegisteredModel,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def row_to_registered_model(row: Mapping[str, Any]) -> RegisteredModel:
    """Convert a SQLAlchemy mapping row into a ``RegisteredModel``."""
    meta_raw = row["metadata_json"]
    metadata = json.loads(meta_raw) if isinstance(meta_raw, (str, bytes)) else dict(meta_raw or {})
    return RegisteredModel(
        model_id=uuid.UUID(str(row["model_id"])),
        strategy_name=row["strategy_name"],
        model_version=row["model_version"],
        feature_set_version=row["feature_set_version"],
        created_at=row["created_at"],
        metadata=metadata,
        active=bool(row["active"]),
    )

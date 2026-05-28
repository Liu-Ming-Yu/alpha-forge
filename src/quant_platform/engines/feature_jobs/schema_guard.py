"""Pure feature schema validation for engine plugin payloads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quant_platform.core.exceptions import DataStalenessError

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping


def validate_required_feature_schema(
    *,
    engine_name: str,
    feature_data: Mapping[uuid.UUID, Mapping[str, object]],
    required_features: Iterable[str],
    allow_empty: bool = True,
) -> None:
    """Fail closed when feature rows do not match the engine's declared schema."""
    if not feature_data:
        if not allow_empty:
            raise DataStalenessError(
                f"{engine_name} feature schema validation failed: no feature data available"
            )
        return
    required = set(required_features)
    if not required:
        return

    missing: list[str] = []
    invalid: list[str] = []
    for instrument_id, features in feature_data.items():
        absent = sorted(required - set(features))
        if absent:
            missing.append(f"{instrument_id}: {', '.join(absent)}")
        for name in required & set(features):
            value = features[name]
            try:
                numeric_value = _feature_float(value)
            except (TypeError, ValueError, OverflowError):
                invalid.append(f"{instrument_id}: {name}={value!r}")
                continue
            if not math.isfinite(numeric_value):
                invalid.append(f"{instrument_id}: {name}={value!r}")
    if missing or invalid:
        details = []
        if missing:
            details.append("missing required features [" + "; ".join(missing[:5]) + "]")
        if invalid:
            details.append("non-finite feature values [" + "; ".join(invalid[:5]) + "]")
        raise DataStalenessError(
            f"{engine_name} feature schema validation failed: " + "; ".join(details)
        )


def _feature_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"feature value must be numeric, got {type(value).__name__}")

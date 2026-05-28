"""Unit tests for engine feature schema validation."""

from __future__ import annotations

import uuid

import pytest

from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.feature_jobs.schema_guard import validate_required_feature_schema


def test_feature_schema_guard_noops_without_required_features() -> None:
    validate_required_feature_schema(
        engine_name="equity",
        feature_data={uuid.uuid4(): {"momentum": "bad-but-unused"}},
        required_features=[],
    )


def test_feature_schema_guard_accepts_finite_numeric_values() -> None:
    validate_required_feature_schema(
        engine_name="equity",
        feature_data={uuid.uuid4(): {"momentum": 1, "value": "0.25"}},
        required_features=["momentum", "value"],
    )


def test_feature_schema_guard_rejects_empty_payload_when_required() -> None:
    with pytest.raises(DataStalenessError, match="no feature data available"):
        validate_required_feature_schema(
            engine_name="equity",
            feature_data={},
            required_features=["momentum"],
            allow_empty=False,
        )


def test_feature_schema_guard_rejects_missing_features() -> None:
    with pytest.raises(DataStalenessError, match="missing required features"):
        validate_required_feature_schema(
            engine_name="equity",
            feature_data={uuid.uuid4(): {"momentum": 1.0}},
            required_features=["momentum", "value"],
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "not-a-number", object()])
def test_feature_schema_guard_rejects_non_finite_values(bad_value: object) -> None:
    with pytest.raises(DataStalenessError, match="non-finite feature values"):
        validate_required_feature_schema(
            engine_name="equity",
            feature_data={uuid.uuid4(): {"momentum": bad_value}},
            required_features=["momentum"],
        )

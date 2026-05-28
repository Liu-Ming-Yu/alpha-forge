"""LinearWeightSignalModel feature_set_version enforcement tests."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.core.domain.research import FeatureVector, StrategyRun
from quant_platform.core.domain.research.runs import RunStatus, RunType
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

_NOW = datetime(2026, 5, 8, 14, 30, tzinfo=UTC)


def _vector(*, fsv: str, feature_value: float = 0.5) -> FeatureVector:
    return FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        as_of=_NOW,
        features={"momentum": feature_value},
        feature_set_version=fsv,
        artifact_uri="memory://test",
    )


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="test",
        strategy_version="0.1.0",
        run_type=RunType.LIVE,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
    )


def test_pinned_fsv_matches_passes() -> None:
    model = LinearWeightSignalModel(
        {"momentum": 1.0},
        expected_feature_set_version="v2026-05-08",
    )
    scores = model.score([_vector(fsv="v2026-05-08")], _strategy_run())
    assert len(scores) == 1
    assert math.isclose(scores[0].score, 0.5)


def test_pinned_fsv_mismatch_raises() -> None:
    model = LinearWeightSignalModel(
        {"momentum": 1.0},
        expected_feature_set_version="v2026-05-08",
    )
    with pytest.raises(ValueError, match="Feature set version mismatch"):
        model.score([_vector(fsv="v2026-04-01")], _strategy_run())


def test_unpinned_consistent_batch_passes() -> None:
    model = LinearWeightSignalModel({"momentum": 1.0})
    vectors = [_vector(fsv="v2026-05-08") for _ in range(3)]
    scores = model.score(vectors, _strategy_run())
    assert len(scores) == 3


def test_unpinned_mixed_batch_raises() -> None:
    """Even without an explicit expected_fsv, a mixed-version batch must
    fail loudly: it is a symptom of a stale or partial feature rebuild
    and would otherwise produce silently inconsistent scores."""
    model = LinearWeightSignalModel({"momentum": 1.0})
    vectors = [
        _vector(fsv="v2026-05-08"),
        _vector(fsv="v2026-04-01"),  # stale
    ]
    with pytest.raises(ValueError, match="Inconsistent feature_set_version within scoring batch"):
        model.score(vectors, _strategy_run())

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.application.features.registry import FeatureFamilyRegistry
from quant_platform.core.domain.research import (
    FeatureInputContext,
    FeatureRequest,
    FeatureResult,
    FeatureVector,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import FeatureComputer

_AS_OF = datetime(2026, 5, 19, tzinfo=UTC)
_INSTRUMENT = uuid.uuid4()
_RUN = uuid.uuid4()


@dataclass(frozen=True)
class _Plugin:
    name: str
    feature_set_version: str
    required_inputs: tuple[str, ...]
    computers: tuple[FeatureComputer, ...] = ()

    def build_computers(self) -> tuple[FeatureComputer, ...]:
        return self.computers


@dataclass(frozen=True)
class _Computer:
    feature_family: str
    feature_set_version: str
    output_features: tuple[str, ...]
    required_inputs: tuple[str, ...] = ()
    schema_hash_override: str | None = None

    @property
    def schema_hash(self) -> str:
        return self.schema_hash_override or ordered_feature_schema_hash(self.output_features)

    async def compute(self, request: FeatureRequest) -> FeatureResult:
        return FeatureResult(
            feature_set_version=request.feature_set_version,
            vectors=(
                FeatureVector(
                    vector_id=uuid.uuid4(),
                    instrument_id=request.instruments[0],
                    as_of=request.as_of,
                    feature_set_version=request.feature_set_version,
                    features={self.output_features[0]: 1.0},
                    strategy_run_id=_RUN,
                ),
            ),
            diagnostics={"schema_hash": self.schema_hash},
            passed=True,
        )


def _ohlcv_computer(feature_set_version: str = "ohlcv-v1") -> _Computer:
    return _Computer(
        feature_family="ohlcv",
        feature_set_version=feature_set_version,
        output_features=("momentum_21d",),
    )


def _request(
    feature_set_version: str = "ohlcv-v1",
    *,
    available_inputs: tuple[str, ...] = ("bars_eod",),
) -> FeatureRequest:
    return FeatureRequest(
        feature_set_version=feature_set_version,
        instruments=(_INSTRUMENT,),
        start=_AS_OF,
        end=_AS_OF,
        as_of=_AS_OF,
        context=FeatureInputContext(available_inputs=available_inputs),
    )


def test_duplicate_feature_family_plugin_keys_fail() -> None:
    plugin = _Plugin("ohlcv", "ohlcv-v1", ("bars_eod",), (_ohlcv_computer(),))

    with pytest.raises(ValueError, match="duplicate feature family plugin"):
        FeatureFamilyRegistry.from_plugins((plugin, plugin))


def test_unknown_feature_family_fails_clearly() -> None:
    registry = FeatureFamilyRegistry.from_plugins(
        (_Plugin("ohlcv", "ohlcv-v1", (), (_ohlcv_computer(),)),)
    )

    with pytest.raises(ValueError, match="unknown feature family"):
        registry.get(feature_family="text", feature_set_version="text-v1")


@pytest.mark.asyncio
async def test_feature_set_version_mismatch_fails_closed() -> None:
    registry = FeatureFamilyRegistry.from_plugins(
        (_Plugin("ohlcv", "ohlcv-v1", (), (_ohlcv_computer(),)),)
    )

    result = await registry.compute(feature_family="ohlcv", request=_request("ohlcv-v2"))

    assert result.passed is False
    assert result.diagnostics["blockers"] == ("feature_set_version_mismatch",)


@pytest.mark.asyncio
async def test_feature_family_with_no_computers_fails_closed() -> None:
    registry = FeatureFamilyRegistry.from_plugins((_Plugin("ohlcv", "ohlcv-v1", ()),))

    result = await registry.compute(feature_family="ohlcv", request=_request())

    assert result.passed is False
    assert result.diagnostics["blockers"] == ("feature_family_has_no_computers",)


@pytest.mark.asyncio
async def test_missing_required_inputs_fail_closed() -> None:
    registry = FeatureFamilyRegistry.from_plugins(
        (
            _Plugin(
                "ohlcv",
                "ohlcv-v1",
                ("bars_eod",),
                (
                    _Computer(
                        feature_family="ohlcv",
                        feature_set_version="ohlcv-v1",
                        output_features=("momentum_21d",),
                    ),
                ),
            ),
        )
    )

    result = await registry.compute(
        feature_family="ohlcv",
        request=_request(available_inputs=()),
    )

    assert result.passed is False
    assert result.diagnostics["blockers"] == ("feature_required_inputs_missing",)
    assert result.diagnostics["missing_inputs"] == ("bars_eod",)


@pytest.mark.asyncio
async def test_computer_missing_required_inputs_fail_closed() -> None:
    registry = FeatureFamilyRegistry.from_plugins(
        (
            _Plugin(
                "ohlcv",
                "ohlcv-v1",
                (),
                (
                    _Computer(
                        feature_family="ohlcv",
                        feature_set_version="ohlcv-v1",
                        output_features=("momentum_21d",),
                        required_inputs=("bars_eod",),
                    ),
                ),
            ),
        )
    )

    result = await registry.compute(
        feature_family="ohlcv",
        request=_request(available_inputs=()),
    )

    assert result.passed is False
    assert result.diagnostics["blockers"] == ("feature_required_inputs_missing",)
    assert result.diagnostics["missing_inputs"] == ("bars_eod",)


@pytest.mark.asyncio
async def test_schema_hash_mismatch_fails_closed() -> None:
    registry = FeatureFamilyRegistry.from_plugins(
        (
            _Plugin(
                "ohlcv",
                "ohlcv-v1",
                ("bars_eod",),
                (
                    _Computer(
                        feature_family="ohlcv",
                        feature_set_version="ohlcv-v1",
                        output_features=("momentum_21d",),
                        schema_hash_override="wrong",
                    ),
                ),
            ),
        )
    )

    result = await registry.compute(feature_family="ohlcv", request=_request())

    assert result.passed is False
    assert result.diagnostics["blockers"] == ("feature_schema_hash_mismatch",)


@pytest.mark.asyncio
async def test_registered_feature_computer_returns_typed_result() -> None:
    registry = FeatureFamilyRegistry.from_plugins(
        (
            _Plugin(
                "ohlcv",
                "ohlcv-v1",
                ("bars_eod",),
                (
                    _Computer(
                        feature_family="ohlcv",
                        feature_set_version="ohlcv-v1",
                        output_features=("momentum_21d",),
                    ),
                ),
            ),
        )
    )

    result = await registry.compute(feature_family="ohlcv", request=_request())

    assert result.passed is True
    assert result.feature_set_version == "ohlcv-v1"
    assert result.vectors[0].features == {"momentum_21d": 1.0}


def test_feature_request_and_result_contracts_are_type_safe() -> None:
    with pytest.raises(ValueError, match="instruments must be unique"):
        FeatureRequest(
            feature_set_version="ohlcv-v1",
            instruments=(_INSTRUMENT, _INSTRUMENT),
            start=_AS_OF,
            end=_AS_OF,
            as_of=_AS_OF,
        )

    with pytest.raises(ValueError, match="feature_set_version mismatch"):
        FeatureResult(
            feature_set_version="ohlcv-v1",
            vectors=(
                FeatureVector(
                    vector_id=uuid.uuid4(),
                    instrument_id=_INSTRUMENT,
                    as_of=_AS_OF,
                    feature_set_version="other",
                    features={"momentum_21d": 1.0},
                    strategy_run_id=_RUN,
                ),
            ),
            diagnostics={},
            passed=False,
        )


def test_feature_request_requires_typed_input_context() -> None:
    with pytest.raises(TypeError, match="FeatureInputContext"):
        FeatureRequest(
            feature_set_version="ohlcv-v1",
            instruments=(_INSTRUMENT,),
            start=_AS_OF,
            end=_AS_OF,
            as_of=_AS_OF,
            context={  # type: ignore[arg-type]
                "available_inputs": ("bars_eod",),
                "schema_hashes": {"bars_eod": "abc"},
            },
        )

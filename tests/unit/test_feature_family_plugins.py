"""Tests for the executable feature-family plugins and registry assembly."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.bootstrap.data.feature_plugins import (
    build_feature_family_plugins,
    build_feature_registry,
)
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureRequest
from quant_platform.core.domain.signals.feature_inputs import BARS_EOD_INPUT, FeatureInputContext
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.services.research_service.features.plugins import (
    CATALYST_FAMILY,
)


def _bars(instrument_id: uuid.UUID, count: int) -> list[MarketBar]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows: list[MarketBar] = []
    for offset in range(count):
        price = Decimal("100") + Decimal(offset % 17)
        rows.append(
            MarketBar(
                bar_id=uuid.uuid4(),
                instrument_id=instrument_id,
                timestamp=start + timedelta(days=offset),
                bar_seconds=86400,
                open=price - Decimal("0.50"),
                high=price + Decimal("1.00"),
                low=price - Decimal("1.00"),
                close=price,
                volume=1_000_000 + offset * 1_000,
            )
        )
    return rows


def _request(version: str, instruments: tuple[uuid.UUID, ...], bar_data: object) -> FeatureRequest:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    return FeatureRequest(
        feature_set_version=version,
        instruments=instruments,
        start=as_of - timedelta(days=365),
        end=as_of,
        as_of=as_of,
        context=FeatureInputContext(
            available_inputs=(BARS_EOD_INPUT,),
            payloads={BARS_EOD_INPUT: bar_data},
        ),
    )


def test_build_feature_registry_registers_every_family() -> None:
    registry = build_feature_registry(InMemoryFeatureRepository())
    keys = set(registry.keys())
    # close + catalyst v10 + event + composite = 4 family/versions.
    assert len(keys) == 4
    assert "close:1.1.0" in keys
    assert "catalyst:paper-alpha-catalyst-v10" in keys
    assert "composite:paper-alpha-composite-v1" in keys
    assert registry.family_for_version("paper-alpha-catalyst-v10") == "catalyst"
    assert registry.family_for_version("1.1.0") == "close"
    assert registry.family_for_version("not-a-real-version") is None


def test_feature_family_plugins_have_consistent_schema_hashes() -> None:
    for plugin in build_feature_family_plugins(InMemoryFeatureRepository()):
        for computer in plugin.build_computers():
            assert computer.feature_family == plugin.name
            assert computer.feature_set_version == plugin.feature_set_version
            assert computer.schema_hash == ordered_feature_schema_hash(computer.output_features)


@pytest.mark.asyncio
async def test_registry_computes_current_catalyst_version() -> None:
    registry = build_feature_registry(InMemoryFeatureRepository())
    instruments = tuple(uuid.uuid4() for _ in range(3))
    bar_data = {instrument_id: _bars(instrument_id, 320) for instrument_id in instruments}

    request = _request("paper-alpha-catalyst-v10", instruments, bar_data)
    result = await registry.compute(feature_family=CATALYST_FAMILY, request=request)
    assert result.passed, result.diagnostics
    assert result.feature_set_version == "paper-alpha-catalyst-v10"
    assert result.vectors
    for vector in result.vectors:
        assert vector.feature_set_version == "paper-alpha-catalyst-v10"
        assert vector.features


@pytest.mark.asyncio
async def test_registry_fails_closed_when_bars_input_missing() -> None:
    registry = build_feature_registry(InMemoryFeatureRepository())
    instruments = (uuid.uuid4(),)
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    request = FeatureRequest(
        feature_set_version="paper-alpha-catalyst-v10",
        instruments=instruments,
        start=as_of - timedelta(days=30),
        end=as_of,
        as_of=as_of,
        context=FeatureInputContext(),
    )
    result = await registry.compute(feature_family=CATALYST_FAMILY, request=request)
    assert not result.passed
    assert result.diagnostics.get("blockers") == ("feature_required_inputs_missing",)


@pytest.mark.asyncio
async def test_registry_rejects_unknown_version() -> None:
    registry = build_feature_registry(InMemoryFeatureRepository())
    instruments = (uuid.uuid4(),)
    request = _request("paper-alpha-catalyst-next", instruments, {})
    result = await registry.compute(feature_family=CATALYST_FAMILY, request=request)
    assert not result.passed
    assert result.diagnostics.get("blockers") == ("feature_set_version_mismatch",)

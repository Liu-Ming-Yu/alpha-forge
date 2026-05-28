from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureVector
from quant_platform.services.research_service.features.paper_alpha.composite import (
    PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
    build_paper_alpha_composite_feature_bundle,
)


@pytest.mark.asyncio
async def test_composite_bundle_merges_admitted_source_features() -> None:
    as_of = datetime(2026, 4, 17, tzinfo=UTC)
    instrument_id = uuid.uuid4()
    repo = _Repo(
        {
            "paper-alpha-catalyst-v10": (
                _vector(
                    instrument_id,
                    as_of,
                    "paper-alpha-catalyst-v10",
                    {"v10_stability_abs_text_specificity_event_surprise_21d": 0.4},
                ),
            ),
            "paper-alpha-event-reaction-v2": (
                _vector(
                    instrument_id,
                    as_of,
                    "paper-alpha-event-reaction-v2",
                    {"event_reaction_v2_sec_count_7_9_momo1_momo3_21d": 0.3},
                ),
            ),
            "paper-alpha-intraday-microstructure-v2": (
                _vector(
                    instrument_id,
                    as_of,
                    "paper-alpha-intraday-microstructure-v2",
                    {"intraday_v2_signed_range_expansion_band_2_3_close_pressure_21d": 0.2},
                ),
            ),
        }
    )

    bundle = await build_paper_alpha_composite_feature_bundle(
        {instrument_id: _bars(instrument_id, as_of)},
        source_feature_repo=repo,
        as_of=as_of,
    )

    row = bundle.alpha_features[instrument_id]
    assert PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION == "paper-alpha-composite-v1"
    assert "momentum_1m" in row
    assert row["v10_stability_abs_text_specificity_event_surprise_21d"] == 0.4
    assert row["event_reaction_v2_sec_count_7_9_momo1_momo3_21d"] == 0.3
    assert row["intraday_v2_signed_range_expansion_band_2_3_close_pressure_21d"] == 0.2


@pytest.mark.asyncio
async def test_composite_bundle_can_merge_v10_text_features() -> None:
    as_of = datetime(2026, 4, 17, tzinfo=UTC)
    instrument_id = uuid.uuid4()
    repo = _Repo(
        {
            "paper-alpha-catalyst-v10": (
                _vector(
                    instrument_id,
                    as_of,
                    "paper-alpha-catalyst-v10",
                    {"v10_stability_abs_text_specificity_event_surprise_21d": 0.5},
                ),
            ),
        }
    )

    bundle = await build_paper_alpha_composite_feature_bundle(
        {instrument_id: _bars(instrument_id, as_of)},
        source_feature_repo=repo,
        as_of=as_of,
        text_feature_set_version="paper-alpha-catalyst-v10",
    )

    row = bundle.alpha_features[instrument_id]
    assert row["v10_stability_abs_text_specificity_event_surprise_21d"] == 0.5


class _Repo:
    def __init__(self, vectors: dict[str, tuple[FeatureVector, ...]]) -> None:
        self._vectors = vectors

    async def get_vectors(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        _as_of: datetime,
    ) -> list[FeatureVector]:
        wanted = set(instrument_ids)
        return [
            vector
            for vector in self._vectors.get(feature_set_version, ())
            if vector.instrument_id in wanted
        ]


def _vector(
    instrument_id: uuid.UUID,
    as_of: datetime,
    feature_set_version: str,
    features: dict[str, float],
) -> FeatureVector:
    return FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=as_of,
        available_at=as_of,
        feature_set_version=feature_set_version,
        strategy_run_id=uuid.uuid4(),
        features=features,
    )


def _bars(instrument_id: uuid.UUID, as_of: datetime) -> tuple[MarketBar, ...]:
    bars: list[MarketBar] = []
    for offset in range(40):
        price = Decimal("100") + Decimal(offset)
        ts = as_of - timedelta(days=40 - offset)
        bars.append(
            MarketBar(
                bar_id=uuid.uuid4(),
                instrument_id=instrument_id,
                timestamp=ts,
                bar_seconds=86400,
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal("0.5"),
                volume=1000 + offset,
            )
        )
    return tuple(bars)

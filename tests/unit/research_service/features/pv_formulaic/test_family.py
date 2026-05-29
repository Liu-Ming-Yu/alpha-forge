"""Tests for the live pv_formulaic FeatureBundle assembly (ADR-011 increment 2b)."""

from __future__ import annotations

import uuid
from datetime import UTC, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from quant_platform.core.domain.market_data.bars import MarketBar
from quant_platform.services.research_service.features.pv_formulaic.bars_frame import (
    market_bars_to_ohlcv_frame,
)
from quant_platform.services.research_service.features.pv_formulaic.compute import (
    compute_pv_formulaic_frame,
)
from quant_platform.services.research_service.features.pv_formulaic.family import (
    PV_FORMULAIC_FEATURE_SET_VERSION,
    build_pv_formulaic_feature_bundle,
)


def _bars(n_instruments: int = 3, n_days: int = 55) -> dict[uuid.UUID, list[MarketBar]]:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    by_instrument: dict[uuid.UUID, list[MarketBar]] = {}
    for j in range(n_instruments):
        inst = uuid.UUID(int=j + 1)
        base = 100.0 + 10.0 * j
        bars = []
        for i, day in enumerate(dates):
            close = base + 0.5 * i + 0.3 * (i % 5)
            bars.append(
                MarketBar(
                    bar_id=uuid.uuid4(),
                    instrument_id=inst,
                    timestamp=pd.Timestamp(day).tz_localize(UTC).to_pydatetime(),
                    bar_seconds=86400,
                    open=Decimal(str(close - 0.5)),
                    high=Decimal(str(close + 1.0)),
                    low=Decimal(str(close - 1.0)),
                    close=Decimal(str(close)),
                    volume=1000 + i,
                )
            )
        by_instrument[inst] = bars
    return by_instrument


def test_bundle_carries_raw_latest_row_features() -> None:
    bars = _bars()
    bundle = build_pv_formulaic_feature_bundle(bars)

    assert bundle.alpha_features  # non-empty
    assert all(isinstance(k, uuid.UUID) for k in bundle.alpha_features)

    # Bundle values are the RAW latest-row features (not re-normalized).
    frame = compute_pv_formulaic_frame(market_bars_to_ohlcv_frame(bars))
    latest = frame.sort_values("date").groupby("instrument_id").tail(1).set_index("instrument_id")
    for inst, feats in bundle.alpha_features.items():
        for name, value in feats.items():
            assert value == pytest.approx(float(latest.loc[inst, name]))


def test_bundle_empty_for_no_bars() -> None:
    assert build_pv_formulaic_feature_bundle({}).alpha_features == {}


def test_as_of_uses_only_in_window_bars() -> None:
    bars = _bars(n_instruments=1, n_days=40)
    full = build_pv_formulaic_feature_bundle(bars)
    # Cut off before the last bar: the latest row (and its feature values) differ.
    only = next(iter(bars))
    cutoff = bars[only][-3].timestamp - timedelta(hours=1)
    trimmed = build_pv_formulaic_feature_bundle(bars, as_of=cutoff)
    assert trimmed.alpha_features  # still computes from the in-window history
    assert full.alpha_features[only] != trimmed.alpha_features[only]


def test_feature_set_version_pinned() -> None:
    assert PV_FORMULAIC_FEATURE_SET_VERSION == "pv-formulaic-live-v1"

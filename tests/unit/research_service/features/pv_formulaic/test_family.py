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


def test_bundle_carries_cross_sectionally_rank_normalized_features() -> None:
    bars = _bars()
    bundle = build_pv_formulaic_feature_bundle(bars)

    assert bundle.alpha_features  # non-empty
    assert all(isinstance(k, uuid.UUID) for k in bundle.alpha_features)

    # Bundle values are the RANK-NORMALIZED latest-row features (the live half of
    # the dollar-volume fix) — the per-date cross-sectional percentile rank, NOT
    # the raw values. Verify against the same kernel the backtest scorer uses.
    from quant_platform.services.research_service.features.kernel.transforms import (  # noqa: PLC0415
        cross_sectional_rank_normalize,
    )
    from quant_platform.services.research_service.features.pv_formulaic.compute import (  # noqa: PLC0415
        PV_FORMULAIC_FEATURE_NAMES,
    )

    frame = compute_pv_formulaic_frame(market_bars_to_ohlcv_frame(bars))
    latest = frame.sort_values("date").groupby("instrument_id", sort=False).tail(1)
    cols = [n for n in PV_FORMULAIC_FEATURE_NAMES if n in latest.columns]
    expected = cross_sectional_rank_normalize(latest, cols, date_column="date").set_index(
        "instrument_id"
    )
    # Ranks live in [0, 1]; not all identical (a genuine cross-section of 3 names).
    for inst, feats in bundle.alpha_features.items():
        for name, value in feats.items():
            assert 0.0 <= value <= 1.0
            assert value == pytest.approx(float(expected.loc[inst, name]))


def test_bundle_empty_for_no_bars() -> None:
    assert build_pv_formulaic_feature_bundle({}).alpha_features == {}


def test_as_of_uses_only_in_window_bars() -> None:
    # Multiple instruments so the cross-sectional ranks are non-degenerate (a
    # single name would rank 1.0 on every feature regardless of the window).
    bars = _bars(n_instruments=3, n_days=40)
    full = build_pv_formulaic_feature_bundle(bars)
    only = next(iter(bars))
    cutoff = bars[only][-3].timestamp - timedelta(hours=1)
    trimmed = build_pv_formulaic_feature_bundle(bars, as_of=cutoff)
    assert trimmed.alpha_features  # still computes from the in-window history
    # A different as-of ⇒ a different latest-row cross-section ⇒ different ranks.
    assert full.alpha_features[only] != trimmed.alpha_features[only]


def test_feature_set_version_pinned() -> None:
    assert PV_FORMULAIC_FEATURE_SET_VERSION == "pv-formulaic-live-v1"

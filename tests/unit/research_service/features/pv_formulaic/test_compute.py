"""Golden-master parity tests for the live pv+formulaic compute (ADR-011 increment 2).

The point of the kernel extraction is parity *by construction*: the live family
runs the same kernel compute the research factory uses. These tests pin that the
``MarketBar -> adapter -> compute`` path reproduces the research feature matrix on
identical data (so the only thing that could drift — the bars adaptation — is
covered), and that the full pv+formulaic surface (incl. G's wq alphas) is
produced.
"""

from __future__ import annotations

import uuid
from datetime import UTC
from decimal import Decimal

import pandas as pd
from pandas.testing import assert_frame_equal

from quant_platform.core.domain.market_data.bars import MarketBar
from quant_platform.services.research_service.features.pv_formulaic.bars_frame import (
    market_bars_to_ohlcv_frame,
)
from quant_platform.services.research_service.features.pv_formulaic.compute import (
    FORMULAIC_FEATURE_NAMES,
    PV_FORMULAIC_FEATURE_NAMES,
    compute_pv_formulaic_frame,
)

_SORT = ["instrument_id", "date"]


def _synthetic_frame(n_instruments: int = 4, n_days: int = 70) -> pd.DataFrame:
    """Deterministic OHLCV frame in the research bars format."""
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows: list[dict[str, object]] = []
    for j in range(n_instruments):
        inst = uuid.UUID(int=j + 1)
        base = 100.0 + 10.0 * j
        for i, day in enumerate(dates):
            close = base + 0.5 * i + 0.3 * (i % 5) - 0.2 * (i % 7)
            rows.append(
                {
                    "instrument_id": inst,
                    "date": pd.Timestamp(day),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + 5.0 * j + float(i),
                }
            )
    return pd.DataFrame(
        rows, columns=["instrument_id", "date", "open", "high", "low", "close", "volume"]
    )


def _bars_from_frame(df: pd.DataFrame) -> dict[uuid.UUID, list[MarketBar]]:
    by_instrument: dict[uuid.UUID, list[MarketBar]] = {}
    for inst, group in df.groupby("instrument_id", sort=False):
        bars = [
            MarketBar(
                bar_id=uuid.uuid4(),
                instrument_id=inst,
                timestamp=pd.Timestamp(row["date"]).tz_localize(UTC).to_pydatetime(),
                bar_seconds=86400,
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=int(row["volume"]),
            )
            for _, row in group.iterrows()
        ]
        by_instrument[inst] = bars
    return by_instrument


def test_live_path_matches_research_compute_by_construction() -> None:
    df = _synthetic_frame()
    bars = _bars_from_frame(df)

    live = compute_pv_formulaic_frame(market_bars_to_ohlcv_frame(bars))
    reference = compute_pv_formulaic_frame(df)

    live = live.sort_values(_SORT).reset_index(drop=True)
    reference = reference.sort_values(_SORT).reset_index(drop=True)
    assert_frame_equal(live, reference)


def test_produces_full_surface_including_g_alphas() -> None:
    frame = compute_pv_formulaic_frame(_synthetic_frame())
    for name in PV_FORMULAIC_FEATURE_NAMES:
        assert name in frame.columns, f"missing feature column {name!r}"
    # G's three formulaic alphas must be in the produced surface.
    assert {"wq_alpha_002_paraphrase", "wq_alpha_012", "wq_alpha_041"} <= set(
        FORMULAIC_FEATURE_NAMES
    )
    # A short-lookback feature is finite on the latest row per instrument.
    latest = frame.sort_values("date").groupby("instrument_id").tail(1)
    assert latest["close_to_open_return"].notna().all()

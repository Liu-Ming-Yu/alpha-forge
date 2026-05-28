from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.research.fundamentals.features import (
    EXPECTED_SIGNS,
    FEATURE_NAMES,
    compute_starter_features,
)
from quant_platform.research.fundamentals.sharadar import SharadarPanel


def _starter_panel() -> SharadarPanel:
    rows = []
    for idx, assets in enumerate([100.0, 100.0, 100.0, 100.0, 120.0]):
        datekey = pd.Timestamp("2024-01-01") + pd.Timedelta(days=90 * idx)
        rows.append(
            {
                "instrument_id": "instrument-a",
                "ticker": "AAA",
                "datekey": datekey,
                "calendardate": datekey,
                "netinc": 10.0,
                "ncfo": 5.0,
                "fcf": 8.0,
                "equity": 100.0,
                "assets": assets,
                "gp": 30.0,
                "marketcap": 200.0,
                "cashneq": 20.0,
                "debt": 50.0,
                "pb": 2.0,
                "pe": 10.0,
            }
        )
    frame = pd.DataFrame(rows)
    return SharadarPanel(
        frame=frame,
        instrument_coverage=1,
        datekey_min=frame["datekey"].min(),
        datekey_max=frame["datekey"].max(),
        dropped_no_instrument_id=(),
        dropped_missing_datekey=0,
        duplicates_resolved=0,
    )


def test_starter_features_export_only_positive_oriented_names() -> None:
    assert set(EXPECTED_SIGNS) == set(FEATURE_NAMES)
    assert all(sign == 1 for sign in EXPECTED_SIGNS.values())
    assert "accruals_4q" not in FEATURE_NAMES
    assert "asset_growth_yoy" not in FEATURE_NAMES
    assert "debt_to_equity" not in FEATURE_NAMES


def test_negative_premium_accounting_features_are_inverted() -> None:
    features = compute_starter_features(_starter_panel())
    last = features.frame.iloc[-1]

    assert "accruals_4q" not in features.frame.columns
    assert "asset_growth_yoy" not in features.frame.columns
    assert "debt_to_equity" not in features.frame.columns

    assert last["low_accruals_4q"] == pytest.approx(-(40.0 - 20.0) / 105.0)
    assert last["low_asset_growth_yoy"] == pytest.approx(-(120.0 / 100.0 - 1.0))
    assert last["low_debt_to_equity"] == pytest.approx(-(50.0 / 100.0))


def test_sector_neutralization_preserves_positive_oriented_surface() -> None:
    features = compute_starter_features(
        _starter_panel(),
        sector_neutralize=True,
        sector_map={"instrument-a": "Technology"},
    )
    feature_values = features.frame.loc[:, list(FEATURE_NAMES)]

    assert set(features.expected_signs.values()) == {1}
    assert feature_values.fillna(0.0).abs().sum().sum() == pytest.approx(0.0)

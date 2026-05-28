from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from quant_platform.services.research_service.feature_quality.diagnostics.direction import (
    build_feature_direction_diagnostics,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def _sample(day: int, rank: float, forward_return: float) -> SupervisedAlphaSample:
    return SupervisedAlphaSample(
        as_of=datetime(2025, 1, 2, tzinfo=UTC) + timedelta(days=day),
        instrument_id=uuid.uuid5(uuid.NAMESPACE_URL, f"instrument:{rank}"),
        features={"quality_alpha": rank},
        forward_return=forward_return,
    )


def test_feature_direction_diagnostics_compare_positive_and_negative_orientation(
    tmp_path,
) -> None:
    card_dir = tmp_path / "cards"
    card_dir.mkdir()
    (card_dir / "quality_alpha.json").write_text(
        json.dumps(
            {
                "name": "quality_alpha",
                "version": "ohlcv-paper-v1.1",
                "owner": "research",
                "economic_thesis": "Higher quality should rank better after costs.",
                "source_datasets": ["daily_adjusted_ohlcv"],
                "required_lags": ["available_at <= decision time"],
                "valid_universe": "unit-test universe",
                "expected_sign": "positive",
                "horizon_days": 21,
                "expected_turnover": "low",
                "state": "paper",
                "failure_modes": ["crowding", "regime shift"],
                "risk_exposures": ["quality"],
            }
        ),
        encoding="utf-8",
    )
    samples = (
        [_sample(day, rank=0.0, forward_return=0.0) for day in range(253)]
        + [_sample(day, rank=1.0, forward_return=0.01) for day in range(253)]
        + [_sample(day, rank=2.0, forward_return=0.02) for day in range(253)]
    )

    payload = build_feature_direction_diagnostics(
        samples=samples,
        feature_set_version="ohlcv-paper-v1.1",
        feature_card_dir=card_dir,
        slippage_bps_per_turnover=0.0,
    )

    row = payload["features"][0]
    assert row["recommended_orientation"] == "positive"
    assert row["orientations"]["positive"]["metrics"]["ic_mean"] > 0.99
    assert row["orientations"]["negative"]["metrics"]["ic_mean"] < -0.99


def test_feature_direction_diagnostics_reports_missing_cards(tmp_path) -> None:
    payload = build_feature_direction_diagnostics(
        samples=[_sample(0, rank=1.0, forward_return=0.01)],
        feature_set_version="ohlcv-paper-v1.1",
        feature_card_dir=tmp_path,
        slippage_bps_per_turnover=10.0,
    )

    assert payload["missing_cards"] == ["quality_alpha"]
    assert payload["features"][0]["recommended_orientation"] == "none"

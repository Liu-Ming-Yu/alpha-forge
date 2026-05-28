"""Tests for the attribution decomposition."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.services.research_service.reports.attribution import (
    AttributionCycle,
    compute_attribution,
    read_attribution,
    write_attribution,
)

_UTC = UTC


def test_attribution_sums_sector_and_regime_to_total_pnl() -> None:
    iid_a = uuid.uuid4()
    iid_b = uuid.uuid4()
    cycles = [
        AttributionCycle(
            as_of=datetime(2024, 1, 1, tzinfo=_UTC),
            target_weights={iid_a: 0.5, iid_b: 0.5},
            realized_returns={iid_a: 0.02, iid_b: -0.01},
            factor_exposures={iid_a: {"momentum_1m": 1.0}, iid_b: {"momentum_1m": -0.5}},
            regime_label="risk_on",
        ),
        AttributionCycle(
            as_of=datetime(2024, 2, 1, tzinfo=_UTC),
            target_weights={iid_a: 0.5, iid_b: 0.5},
            realized_returns={iid_a: 0.01, iid_b: 0.03},
            factor_exposures={iid_a: {"momentum_1m": 0.5}, iid_b: {"momentum_1m": 0.2}},
            regime_label="risk_off",
        ),
    ]
    sector_map = {iid_a: "Tech", iid_b: "Energy"}
    artifact = compute_attribution(uuid.uuid4(), cycles, sector_map)

    assert artifact.num_cycles == 2
    assert set(artifact.regime_pnl.keys()) == {"risk_on", "risk_off"}
    assert set(artifact.sector_pnl.keys()) == {"Tech", "Energy"}
    # sector sum equals total pnl
    assert sum(artifact.sector_pnl.values()) == pytest.approx(artifact.total_pnl)
    assert sum(artifact.regime_pnl.values()) == pytest.approx(artifact.total_pnl)


def test_attribution_unmapped_instruments_land_in_unmapped_bucket() -> None:
    iid = uuid.uuid4()
    cycles = [
        AttributionCycle(
            as_of=datetime(2024, 1, 1, tzinfo=_UTC),
            target_weights={iid: 0.7},
            realized_returns={iid: 0.05},
            factor_exposures={iid: {"momentum_1m": 1.0}},
            regime_label="risk_on",
        ),
    ]
    artifact = compute_attribution(uuid.uuid4(), cycles, {})
    assert "UNMAPPED" in artifact.sector_pnl
    assert artifact.sector_pnl["UNMAPPED"] == pytest.approx(0.035)


def test_attribution_round_trip(tmp_path) -> None:
    iid = uuid.uuid4()
    cycles = [
        AttributionCycle(
            as_of=datetime(2024, 1, 1, tzinfo=_UTC),
            target_weights={iid: 1.0},
            realized_returns={iid: 0.01},
            factor_exposures={iid: {"momentum_1m": 1.0}},
            regime_label="risk_on",
        ),
    ]
    artifact = compute_attribution(uuid.uuid4(), cycles, {iid: "Tech"})
    path = write_attribution(artifact, tmp_path)
    data = read_attribution(path)
    assert data["num_cycles"] == 1
    assert data["sector_pnl"]["Tech"] == pytest.approx(0.01)


def test_attribution_handles_missing_returns() -> None:
    iid_a = uuid.uuid4()
    iid_b = uuid.uuid4()
    cycles = [
        AttributionCycle(
            as_of=datetime(2024, 1, 1, tzinfo=_UTC),
            target_weights={iid_a: 0.5, iid_b: 0.5},
            realized_returns={iid_a: 0.02},  # iid_b missing
            factor_exposures={iid_a: {"momentum_1m": 1.0}, iid_b: {"momentum_1m": 1.0}},
            regime_label="risk_on",
        )
    ]
    artifact = compute_attribution(uuid.uuid4(), cycles, {iid_a: "Tech", iid_b: "Tech"})
    # iid_b contributes 0 because its return is missing
    assert artifact.total_pnl == pytest.approx(0.5 * 0.02)

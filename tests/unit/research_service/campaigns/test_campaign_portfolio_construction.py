from __future__ import annotations

import uuid
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from quant_platform.research.campaign.model_ops import (
    _governed_campaign_return_scale,
)
from quant_platform.services.research_service.campaigns.portfolio.construction import (
    CampaignPortfolioConfig,
    evaluate_long_only_portfolio,
    fit_fold_volatility_scale,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.factory import (
    AlphaEligibilityThresholds,
    run_sample_walk_forward,
    write_campaign_manifest,
    write_walk_forward_artifacts,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from pathlib import Path


def test_effective_gross_cap_records_top_n_single_name_constraint() -> None:
    config = CampaignPortfolioConfig(
        top_n=10,
        max_single_name_weight=0.05,
        max_gross_exposure=0.60,
    )

    payload = config.to_payload()

    assert payload["effective_max_gross_cap"] == pytest.approx(0.50)
    assert "top_n * max_single_name_weight" in str(payload["effective_gross_cap_reason"])


def test_zero_train_volatility_uses_floor_without_infinite_scale() -> None:
    instrument = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    train_scored = [
        (_sample(instrument, start + timedelta(days=day), {"alpha": 1.0}, 0.0), 1.0)
        for day in range(5)
    ]

    scale = fit_fold_volatility_scale(
        train_scored,
        config=CampaignPortfolioConfig(vol_target=0.15, vol_floor=0.05),
    )

    assert scale.train_realized_vol == 0.0
    assert scale.train_effective_vol == pytest.approx(0.05)
    assert scale.raw_vol_scale == pytest.approx(3.0)
    assert scale.exposure_scale == pytest.approx(1.0)


def test_fewer_than_top_n_positive_names_leaves_unused_cash() -> None:
    instruments = [uuid.uuid4() for _ in range(5)]
    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    scored = [
        (_sample(instruments[0], as_of, {"alpha": 1.0}, 0.01), 3.0),
        (_sample(instruments[1], as_of, {"alpha": 1.0}, 0.01), 2.0),
        (_sample(instruments[2], as_of, {"alpha": 1.0}, 0.01), 1.0),
        (_sample(instruments[3], as_of, {"alpha": 1.0}, -0.01), -1.0),
        (_sample(instruments[4], as_of, {"alpha": 1.0}, -0.01), -2.0),
    ]

    result = evaluate_long_only_portfolio(
        scored,
        slippage_bps_per_turnover=0.0,
        config=CampaignPortfolioConfig(top_n=10),
    )

    day = result.day_diagnostics[0]
    assert day["position_count"] == pytest.approx(3.0)
    assert day["gross_exposure"] == pytest.approx(0.15)
    assert day["cash"] == pytest.approx(0.85)


def test_turnover_cap_carries_weights_across_evaluations() -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 2, tzinfo=UTC)
    config = CampaignPortfolioConfig(
        top_n=1,
        max_single_name_weight=0.20,
        max_position_change=0.20,
        max_daily_turnover=0.05,
    )
    first = evaluate_long_only_portfolio(
        [(_sample(instrument_a, start, {"alpha": 1.0}, 0.01), 1.0)],
        slippage_bps_per_turnover=0.0,
        config=config,
    )

    second = evaluate_long_only_portfolio(
        [(_sample(instrument_b, start + timedelta(days=1), {"alpha": 1.0}, 0.01), 1.0)],
        slippage_bps_per_turnover=0.0,
        config=config,
        previous_weights=first.final_weights,
    )

    assert first.day_diagnostics[0]["turnover"] == pytest.approx(0.05)
    assert second.day_diagnostics[0]["turnover"] == pytest.approx(0.05)
    assert instrument_a in second.final_weights


def test_final_weights_respect_gross_cap_after_path_dependent_caps() -> None:
    previous = {
        uuid.uuid4(): 0.20,
        uuid.uuid4(): 0.20,
        uuid.uuid4(): 0.20,
    }
    new_instruments = [uuid.uuid4() for _ in range(10)]
    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    config = CampaignPortfolioConfig(
        top_n=10,
        max_gross_exposure=0.60,
        min_cash_buffer=0.00,
        max_single_name_weight=0.20,
        max_position_change=0.05,
        max_daily_turnover=1.00,
    )
    scored = [
        (_sample(instrument, as_of, {"alpha": 1.0}, 0.01), 1.0) for instrument in new_instruments
    ]

    result = evaluate_long_only_portfolio(
        scored,
        slippage_bps_per_turnover=0.0,
        config=config,
        previous_weights=previous,
    )

    day = result.day_diagnostics[0]
    assert day["gross_exposure"] <= config.effective_max_gross_cap + 1e-12
    assert sum(result.final_weights.values()) <= config.effective_max_gross_cap + 1e-12


def test_walk_forward_vol_scale_uses_train_returns_not_test_volatility() -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    samples: list[SupervisedAlphaSample] = []
    for day in range(70):
        as_of = start + timedelta(days=day)
        high_vol_test_return = 0.20 if day % 2 == 0 else -0.20
        forward_a = 0.001 if day < 40 else high_vol_test_return
        samples.append(_sample(instrument_a, as_of, {"alpha": 1.0}, forward_a))
        samples.append(_sample(instrument_b, as_of, {"alpha": -1.0}, -0.001))

    evidence = run_sample_walk_forward(
        samples=samples,
        config=WalkForwardConfig(
            train_window_days=40,
            test_window_days=20,
            step_days=20,
            min_folds=1,
        ),
        model_version="no-lookahead",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=-10.0),
        slippage_bps_per_turnover=0.0,
        feature_names=["alpha"],
        portfolio_config=CampaignPortfolioConfig(vol_target=0.15, vol_floor=0.05),
    )

    vol = evidence.portfolio_diagnostics["folds"][0]["volatility_scale"]
    assert vol["train_effective_vol"] == pytest.approx(0.05)
    assert vol["exposure_scale"] == pytest.approx(1.0)


def test_portfolio_artifacts_are_written_and_linked(tmp_path: Path) -> None:
    samples = _walk_forward_samples()
    evidence = run_sample_walk_forward(
        samples=samples,
        config=WalkForwardConfig(
            train_window_days=40,
            test_window_days=20,
            step_days=20,
            min_folds=1,
        ),
        model_version="portfolio-artifacts",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=-10.0),
        slippage_bps_per_turnover=0.0,
        feature_names=["alpha"],
        portfolio_config=CampaignPortfolioConfig(),
    )
    evidence = write_walk_forward_artifacts(evidence, output_root=tmp_path / "wf")
    samples_path = tmp_path / "samples.json"
    samples_path.write_text("[]", encoding="utf-8")

    manifest_path = write_campaign_manifest(
        evidence,
        samples_path=samples_path,
        paper_source_weights={"classical": 1.0},
        git_commit="abc123",
        campaign_context={
            "sample_build": {
                "sample_start": samples[0].as_of.isoformat(),
                "sample_end": samples[-1].as_of.isoformat(),
                "universe": [str(row.instrument_id) for row in samples[:2]],
                "horizon_days": 21,
                "bar_seconds": 86400,
                "max_feature_age_days": 3,
            },
            "prediction_evidence": {"counts": {"classical": 1}},
        },
    )

    assert evidence.artifact_root is not None
    assert (evidence.artifact_root / "portfolio_config.json").is_file()
    assert (evidence.artifact_root / "portfolio_diagnostics.json").is_file()
    assert (evidence.artifact_root / "drawdown_diagnostics.json").is_file()
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "portfolio_config.json" in manifest
    assert "portfolio_diagnostics.json" in manifest
    assert "drawdown_diagnostics.json" in manifest


def test_apply_no_trade_band_holds_sub_band_changes() -> None:
    from quant_platform.services.research_service.campaigns.portfolio.targets import (
        apply_no_trade_band,
    )

    held_name, traded_name = uuid.uuid4(), uuid.uuid4()
    current = {held_name: 0.10, traded_name: 0.10}
    target = {held_name: 0.105, traded_name: 0.20}  # 0.005 vs 0.10 changes
    held = apply_no_trade_band(current=current, target=target, band=0.01)
    assert held[held_name] == pytest.approx(0.10)  # sub-band change suppressed
    assert held[traded_name] == pytest.approx(0.20)  # above-band change applied


def test_apply_no_trade_band_disabled_when_zero() -> None:
    from quant_platform.services.research_service.campaigns.portfolio.targets import (
        apply_no_trade_band,
    )

    name = uuid.uuid4()
    held = apply_no_trade_band(current={name: 0.1}, target={name: 0.1001}, band=0.0)
    assert held[name] == pytest.approx(0.1001)


def test_rebalance_interval_skips_intermediate_days() -> None:
    instrument = uuid.uuid4()
    start = datetime(2025, 1, 2, tzinfo=UTC)
    scored = [
        (_sample(instrument, start + timedelta(days=day), {"alpha": 1.0}, 0.01), 1.0)
        for day in range(6)
    ]
    result = evaluate_long_only_portfolio(
        scored,
        slippage_bps_per_turnover=0.0,
        config=CampaignPortfolioConfig(top_n=1, rebalance_interval_days=3),
    )
    rebalanced = [day["rebalanced"] for day in result.day_diagnostics]
    assert rebalanced == [True, False, False, True, False, False]
    assert result.day_diagnostics[1]["turnover"] == pytest.approx(0.0)
    assert result.day_diagnostics[2]["turnover"] == pytest.approx(0.0)


def test_config_rejects_invalid_cost_controls() -> None:
    with pytest.raises(ValueError, match="no_trade_band"):
        CampaignPortfolioConfig(no_trade_band=-0.01)
    with pytest.raises(ValueError, match="rebalance_interval_days"):
        CampaignPortfolioConfig(rebalance_interval_days=0)


def test_governed_campaign_return_scale_fails_closed() -> None:
    with pytest.raises(ValueError, match="return_scale=1.0"):
        _governed_campaign_return_scale(
            Namespace(return_scale=0.15),
            {"xgboost": 0.15},
        )


def _sample(
    instrument_id: uuid.UUID,
    as_of: datetime,
    features: dict[str, float],
    forward_return: float,
) -> SupervisedAlphaSample:
    return SupervisedAlphaSample(
        as_of=as_of,
        instrument_id=instrument_id,
        features=features,
        forward_return=forward_return,
    )


def _walk_forward_samples() -> list[SupervisedAlphaSample]:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows: list[SupervisedAlphaSample] = []
    for day in range(70):
        as_of = start + timedelta(days=day)
        rows.append(_sample(instrument_a, as_of, {"alpha": 1.0}, 0.01))
        rows.append(_sample(instrument_b, as_of, {"alpha": -1.0}, -0.01))
    return rows

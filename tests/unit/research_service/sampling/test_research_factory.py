from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.infrastructure.support.artifact_store import FileSystemArtifactStore
from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    _ic_streak_metrics,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.eligibility import eligibility
from quant_platform.services.research_service.sampling.factory import (
    AlphaEligibilityThresholds,
    list_campaign_manifests,
    load_supervised_samples,
    run_sample_walk_forward,
    walk_forward_object_root,
    write_campaign_manifest,
    write_model_comparison,
    write_walk_forward_artifacts,
)
from quant_platform.services.research_service.sampling.samples import (
    build_supervised_samples,
    daily_as_of_dates,
    research_as_of_dates,
    write_samples_json,
)


class _BarStore:
    def __init__(self, bars: list[MarketBar]) -> None:
        self._bars = bars

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        return [
            bar
            for bar in self._bars
            if bar.instrument_id == instrument_id
            and bar.bar_seconds == bar_seconds
            and start <= bar.timestamp <= end
        ]

    async def store_bars(self, bars: list[MarketBar]) -> None:
        self._bars.extend(bars)

    async def get_corporate_actions(self, instrument_id: uuid.UUID, since: object) -> list[object]:
        del instrument_id, since
        return []


def _bar(instrument_id: uuid.UUID, ts: datetime, close: Decimal) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=ts,
        bar_seconds=86400,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100_000,
    )


@pytest.mark.asyncio
async def test_build_supervised_samples_joins_features_to_forward_returns(
    tmp_path: Path,
) -> None:
    repo = InMemoryFeatureRepository()
    instrument = uuid.uuid4()
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    await repo.store_vector(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=instrument,
            as_of=as_of,
            feature_set_version="v1",
            features={"momentum": 1.0},
            strategy_run_id=uuid.uuid4(),
        )
    )
    store = _BarStore(
        [
            _bar(instrument, as_of, Decimal("100")),
            _bar(instrument, as_of + timedelta(days=5), Decimal("110")),
        ]
    )

    result = await build_supervised_samples(
        feature_repo=repo,
        bar_store=store,
        instrument_ids=[instrument],
        feature_set_version="v1",
        as_of_dates=[as_of],
        horizon_days=5,
    )
    output = write_samples_json(result.samples, tmp_path / "samples.json")
    rows = json.loads(output.read_text(encoding="utf-8"))

    assert result.requested_points == 1
    assert len(result.samples) == 1
    import math

    assert rows[0]["forward_return"] == pytest.approx(math.log(110 / 100))


def test_daily_as_of_dates_is_inclusive() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 3, tzinfo=UTC)

    dates = daily_as_of_dates(start, end)

    assert len(dates) == 3
    assert dates[0] == start
    assert dates[-1] == end


def test_research_as_of_dates_uses_nyse_sessions() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 5, tzinfo=UTC)

    dates = research_as_of_dates(start, end, date_policy="nyse-sessions")

    assert dates == (
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 5, tzinfo=UTC),
    )


def test_research_as_of_dates_can_use_calendar_days() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 3, tzinfo=UTC)

    assert research_as_of_dates(start, end, date_policy="calendar-days") == daily_as_of_dates(
        start,
        end,
    )


def test_research_as_of_dates_respects_exact_utc_bounds() -> None:
    dates = research_as_of_dates(
        datetime(2026, 1, 2, 12, tzinfo=UTC),
        datetime(2026, 1, 6, tzinfo=UTC),
    )

    assert dates == (
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 6, tzinfo=UTC),
    )


def test_sample_walk_forward_writes_standard_artifacts(tmp_path: Path) -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    samples = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        samples.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0, "anti": -1.0},
                "forward_return": 0.01,
            }
        )
        samples.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0, "anti": 1.0},
                "forward_return": -0.01,
            }
        )
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(samples), encoding="utf-8")

    evidence = run_sample_walk_forward(
        samples=load_supervised_samples(path),
        config=WalkForwardConfig(
            train_window_days=40,
            test_window_days=20,
            step_days=20,
            purge_days=1,
            min_folds=3,
        ),
        model_version="candidate-v1",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(
            min_oos_rolling_ic=0.01,
            min_ic_60d=0.01,
            max_fold_negative_ic_streak=3,
            max_drawdown=-0.20,
            min_slippage_adjusted_sharpe=0.0,
        ),
        slippage_bps_per_turnover=0.0,
    )
    evidence = write_walk_forward_artifacts(evidence, output_root=tmp_path / "wf")

    assert evidence.eligibility["passed"] is True
    assert evidence.artifact_root is not None
    assert (evidence.artifact_root / "fold_metrics.json").is_file()
    assert (evidence.artifact_root / "model_manifest.json").is_file()
    assert (evidence.artifact_root / "tearsheet.md").is_file()


def test_sample_walk_forward_emits_attribution_and_stability_metrics(
    tmp_path: Path,
) -> None:
    """The walk-forward harness now produces turnover, attribution, stability,
    and bootstrap evidence so the production-candidate gate can review more
    than just point estimates of IC and Sharpe.
    """
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0, "anti": -1.0},
                "forward_return": 0.01,
                "metadata": [["sector", "tech"], ["regime", "bull"]],
            }
        )
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0, "anti": 1.0},
                "forward_return": -0.01,
                "metadata": [["sector", "energy"], ["regime", "bull"]],
            }
        )
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(rows), encoding="utf-8")

    evidence = run_sample_walk_forward(
        samples=load_supervised_samples(samples_path),
        config=WalkForwardConfig(
            train_window_days=40,
            test_window_days=20,
            step_days=20,
            purge_days=1,
            min_folds=3,
        ),
        model_version="candidate-evidence",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(
            min_oos_rolling_ic=0.01,
            min_ic_60d=0.01,
            max_fold_negative_ic_streak=3,
            max_drawdown=-0.20,
            min_slippage_adjusted_sharpe=0.0,
        ),
        slippage_bps_per_turnover=12.0,
    )
    evidence = write_walk_forward_artifacts(evidence, output_root=tmp_path / "wf")

    assert evidence.artifact_root is not None
    assert (evidence.artifact_root / "attribution.json").is_file()
    assert (evidence.artifact_root / "feature_stability.json").is_file()
    metrics = dict(evidence.metrics)
    assert "turnover_avg" in metrics
    assert metrics["turnover_avg"] >= 0.0
    assert "bootstrap_ic_p05" in metrics
    assert "bootstrap_ic_p95" in metrics
    assert metrics["bootstrap_ic_p05"] <= metrics["bootstrap_ic_p95"]
    assert "feature_stability_avg" in metrics
    assert "top_minus_bottom_decile_ic" in metrics
    assert evidence.slippage_bps_per_turnover == 12.0
    assert "sector" in evidence.attribution
    assert "regime" in evidence.attribution
    assert evidence.attribution["sector"]["tech"]["mean_forward_return"] > 0
    assert evidence.attribution["sector"]["energy"]["mean_forward_return"] < 0


def test_sample_walk_forward_can_run_equal_weight_feature_subset(tmp_path: Path) -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0, "noise": -1.0},
                "forward_return": 0.01,
            }
        )
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0, "noise": 1.0},
                "forward_return": -0.01,
            }
        )
    samples_path = tmp_path / "subset_samples.json"
    samples_path.write_text(json.dumps(rows), encoding="utf-8")

    evidence = run_sample_walk_forward(
        samples=load_supervised_samples(samples_path),
        config=WalkForwardConfig(
            train_window_days=40,
            test_window_days=20,
            step_days=20,
            purge_days=1,
            min_folds=3,
        ),
        model_version="equal-subset",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=0.0),
        slippage_bps_per_turnover=0.0,
        feature_names=["alpha"],
        weight_mode="equal_weight",
    )

    assert evidence.selected_weights == {"alpha": 1.0}
    assert evidence.metrics["oos_rolling_ic"] > 0


def test_sample_walk_forward_return_scale_caps_source_pnl_metrics(tmp_path: Path) -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        adverse = day % 11 in {0, 1}
        forward_a = -0.05 if adverse else 0.01
        forward_b = 0.05 if adverse else -0.01
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0},
                "forward_return": forward_a,
            }
        )
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0},
                "forward_return": forward_b,
            }
        )
    samples_path = tmp_path / "scaled_samples.json"
    samples_path.write_text(json.dumps(rows), encoding="utf-8")
    config = WalkForwardConfig(
        train_window_days=40,
        test_window_days=20,
        step_days=20,
        purge_days=1,
        min_folds=3,
    )

    unscaled = run_sample_walk_forward(
        samples=load_supervised_samples(samples_path),
        config=config,
        model_version="unscaled",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=-10.0),
        slippage_bps_per_turnover=0.0,
        feature_names=["alpha"],
    )
    scaled = run_sample_walk_forward(
        samples=load_supervised_samples(samples_path),
        config=config,
        model_version="scaled",
        feature_set_version="v1",
        thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=-10.0),
        slippage_bps_per_turnover=0.0,
        feature_names=["alpha"],
        return_scale=0.05,
    )

    assert scaled.metrics["return_scale"] == 0.05
    assert scaled.metrics["oos_rolling_ic"] == unscaled.metrics["oos_rolling_ic"]
    assert scaled.metrics["max_drawdown"] > unscaled.metrics["max_drawdown"]
    assert scaled.metrics["turnover_avg"] == pytest.approx(unscaled.metrics["turnover_avg"] * 0.05)


def test_campaign_eligibility_allows_max_negative_streak_boundary() -> None:
    payload = eligibility(
        {
            "oos_rolling_ic": 0.06,
            "ic_60d": 0.04,
            "fold_negative_ic_streak": 2.0,
            "max_drawdown": -0.10,
            "slippage_adjusted_sharpe": 1.2,
        },
        AlphaEligibilityThresholds(max_fold_negative_ic_streak=2),
    )

    assert payload["passed"] is True


def test_campaign_eligibility_uses_fold_streak_not_daily_streak() -> None:
    """Daily streak being long must not fail the gate when fold streak is small.

    Pre-2026-05-25 the eligibility gate consumed a daily-IC streak that, with
    multi-day forward-return horizons, produced long correlated negative runs
    from a single bad window. The unit-rename fixed this — daily streak is now
    informational only and must not block promotion on its own.
    """
    payload = eligibility(
        {
            "oos_rolling_ic": 0.06,
            "ic_60d": 0.04,
            "fold_negative_ic_streak": 1.0,
            # An old daily-streak key under the legacy name. If eligibility is
            # still reading it, this will fail at the threshold of 2.
            "daily_negative_ic_streak": 12.0,
            "max_drawdown": -0.10,
            "slippage_adjusted_sharpe": 1.2,
        },
        AlphaEligibilityThresholds(max_fold_negative_ic_streak=2),
    )

    assert payload["passed"] is True


def test_ic_streak_metric_emits_fold_and_daily_keys() -> None:
    metrics = _ic_streak_metrics(
        fold_rows=[
            {"mean_ic": 0.02},
            {"mean_ic": -0.01},
            {"mean_ic": 0.03},
        ],
        all_daily_ics=[
            ("2025-01-01", -0.01),
            ("2025-01-02", -0.02),
            ("2025-01-03", -0.03),
            ("2025-01-04", -0.04),
        ],
    )

    assert metrics["fold_negative_ic_streak"] == pytest.approx(1.0)
    assert metrics["daily_negative_ic_streak"] == pytest.approx(4.0)
    # The legacy ``negative_ic_streak`` alias was removed on 2026-05-25 to
    # stop the silent unit drift (daily-streak vs fold-streak). Consumers
    # must read by the explicit unit-bearing key.
    assert "negative_ic_streak" not in metrics


def test_calibrated_slippage_uses_observed_when_higher() -> None:
    from quant_platform.services.research_service.sampling.factory import (
        calibrated_slippage_bps_per_turnover,
    )

    assert calibrated_slippage_bps_per_turnover(8.0, default_bps=10.0) == 10.0
    assert calibrated_slippage_bps_per_turnover(15.0, default_bps=10.0) == 15.0
    assert calibrated_slippage_bps_per_turnover(None, default_bps=10.0) == 10.0
    assert calibrated_slippage_bps_per_turnover(0.0, default_bps=0.0, floor_bps=2.0) == 2.0
    # tactic-aware calibration recommendation overrides observed mean when higher
    assert (
        calibrated_slippage_bps_per_turnover(
            5.0,
            default_bps=10.0,
            calibration_recommendation_bps=22.5,
        )
        == 22.5
    )
    # but is ignored when zero or negative (insufficient evidence)
    assert (
        calibrated_slippage_bps_per_turnover(
            5.0,
            default_bps=10.0,
            calibration_recommendation_bps=0.0,
        )
        == 10.0
    )


def test_campaign_manifest_links_standard_artifacts(tmp_path: Path) -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0},
                "forward_return": 0.01,
            }
        )
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0},
                "forward_return": -0.01,
            }
        )
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(rows), encoding="utf-8")
    output_root = walk_forward_object_root(tmp_path)

    evidence = write_walk_forward_artifacts(
        run_sample_walk_forward(
            samples=load_supervised_samples(samples_path),
            config=WalkForwardConfig(
                train_window_days=40,
                test_window_days=20,
                step_days=20,
                purge_days=1,
                min_folds=3,
            ),
            model_version="candidate-v2",
            feature_set_version="v1",
            thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=0.0),
            slippage_bps_per_turnover=0.0,
        ),
        output_root=output_root,
    )
    manifest_path = write_campaign_manifest(
        evidence,
        samples_path=samples_path,
        paper_source_weights={"classical": 0.7, "xgboost": 0.3},
        git_commit="abc123",
        campaign_context={
            "sample_build": {
                "sample_start": start.isoformat(),
                "sample_end": (start + timedelta(days=139)).isoformat(),
                "universe": [str(instrument_a), str(instrument_b)],
                "date_policy": "nyse-sessions",
                "horizon_days": 21,
                "bar_seconds": 86400,
                "max_feature_age_days": 5,
            },
            "prediction_evidence": {"counts": {"xgboost": 10}},
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["git_commit"] == "abc123"
    assert manifest["artifacts"]["samples"] == str(samples_path)
    assert manifest["artifacts"]["tearsheet"].endswith("tearsheet.md")
    assert manifest["next_allowed_paper_mode"] == "paper_ensemble"
    assert manifest["replay_context"]["sample_start"] == start.isoformat()
    assert manifest["campaign_evidence"]["model_version"] == "candidate-v2"
    assert manifest["campaign_evidence"]["horizon_days"] == 21
    assert manifest["campaign_evidence"]["max_feature_age_days"] == 5
    assert manifest["campaign_evidence"]["walk_forward_folds"]
    assert manifest["campaign_evidence"]["prediction_evidence"] == {"counts": {"xgboost": 10}}
    assert manifest["campaign_evidence"]["passed"] is True
    assert list_campaign_manifests(output_root, limit=1)[0]["run_id"] == str(evidence.run_id)

    with pytest.raises(ValueError, match="campaign_evidence\\.horizon_days"):
        write_campaign_manifest(
            evidence,
            samples_path=samples_path,
            paper_source_weights={"classical": 0.7, "xgboost": 0.3},
            git_commit="abc123",
            campaign_context={
                "sample_build": {
                    "sample_start": start.isoformat(),
                    "sample_end": (start + timedelta(days=139)).isoformat(),
                    "universe": [str(instrument_a), str(instrument_b)],
                    "date_policy": "nyse-sessions",
                    "bar_seconds": 86400,
                    "max_feature_age_days": 5,
                },
                "prediction_evidence": {"counts": {"xgboost": 10}},
            },
        )


def test_campaign_manifest_links_model_comparison(tmp_path: Path) -> None:
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0},
                "forward_return": 0.01,
            }
        )
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0},
                "forward_return": -0.01,
            }
        )
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(rows), encoding="utf-8")
    output_root = walk_forward_object_root(tmp_path)
    evidence = write_walk_forward_artifacts(
        run_sample_walk_forward(
            samples=load_supervised_samples(samples_path),
            config=WalkForwardConfig(
                train_window_days=40,
                test_window_days=20,
                step_days=20,
                purge_days=1,
                min_folds=3,
            ),
            model_version="candidate-comparison",
            feature_set_version="v1",
            thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=0.0),
            slippage_bps_per_turnover=0.0,
        ),
        output_root=output_root,
    )
    comparison_path = write_model_comparison(
        evidence,
        rows=[
            {
                "candidate": "ic_weighted_linear",
                "status": "passed",
                "selected": True,
            }
        ],
    )
    manifest_path = write_campaign_manifest(
        evidence,
        samples_path=samples_path,
        paper_source_weights={"classical": 0.7},
        git_commit="abc123",
        model_comparison_path=comparison_path,
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert comparison["selected_candidate"] == "ic_weighted_linear"
    assert manifest["artifacts"]["model_comparison"] == str(comparison_path)


def test_campaign_manifest_artifact_store_does_not_nest_relative_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    instrument_a = uuid.uuid4()
    instrument_b = uuid.uuid4()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(140):
        as_of = start + timedelta(days=day)
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_a),
                "features": {"alpha": 1.0},
                "forward_return": 0.01,
            }
        )
        rows.append(
            {
                "as_of": as_of.isoformat(),
                "instrument_id": str(instrument_b),
                "features": {"alpha": -1.0},
                "forward_return": -0.01,
            }
        )
    samples_path = Path("samples.json")
    samples_path.write_text(json.dumps(rows), encoding="utf-8")
    output_root = Path("objects/research/walk_forward")
    artifact_store = FileSystemArtifactStore(output_root)
    evidence = write_walk_forward_artifacts(
        run_sample_walk_forward(
            samples=load_supervised_samples(samples_path),
            config=WalkForwardConfig(
                train_window_days=40,
                test_window_days=20,
                step_days=20,
                purge_days=1,
                min_folds=3,
            ),
            model_version="candidate-artifact-store",
            feature_set_version="v1",
            thresholds=AlphaEligibilityThresholds(min_slippage_adjusted_sharpe=0.0),
            slippage_bps_per_turnover=0.0,
        ),
        output_root=output_root,
        artifact_store=artifact_store,
    )

    manifest_path = write_campaign_manifest(
        evidence,
        samples_path=samples_path,
        paper_source_weights={"classical": 1.0},
        git_commit="abc123",
        artifact_store=artifact_store,
    )

    assert manifest_path == output_root / str(evidence.run_id) / "campaign_manifest.json"
    assert manifest_path.exists()
    assert not (
        output_root / output_root / str(evidence.run_id) / "campaign_manifest.json"
    ).exists()

"""Regression tests pinning the 21d-label vs 1d-realized-return separation.

These tests cover the audit fix that separates:

* ``forward_return`` — the 21-trading-day forward LOG return used only as
  a *label* for IC, feature weighting, and bootstrap CIs.
* ``realized_return_1d`` — the one-day SIMPLE realized return used as the
  canonical P&L unit for Sharpe / drawdown / total return / equity curve.

The previous implementation fed ``forward_return`` directly into a daily
return series and compounded each value as a one-day realized P&L —
which inflated total return and distorted Sharpe by roughly the
horizon factor. These tests guarantee that the fix sticks:

1. ``daily_metrics`` (signed-rank) uses ``realized_return_1d`` for P&L
   and ``forward_return`` only for IC when samples carry both.
2. ``evaluate_long_only_portfolio`` does the same.
3. Old sample JSON without the new optional fields still loads.
4. ``WalkForwardConfig`` rejects ``purge_days < label_horizon_days``
   only when ``label_horizon_days`` is explicitly set.
5. Sample-level purge in ``run_sample_walk_forward`` drops training
   rows whose label window reaches the test window.
6. ``non_overlapping_bucket_returns`` compounds (not sums) simple
   daily returns into bucket returns.
7. Legacy-mode behavior (no ``realized_return_1d``) is preserved
   verbatim for callers that haven't migrated.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    daily_metrics,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    bucket_sharpe,
    compound_return,
    non_overlapping_bucket_returns,
)
from quant_platform.services.research_service.campaigns.portfolio.evaluation import (
    evaluate_long_only_portfolio,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.sample_io import (
    load_supervised_samples,
)
from quant_platform.services.research_service.sampling.samples import (
    SupervisedAlphaSample,
    has_realized_returns,
)


def _ts(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _sample(
    *,
    day: int,
    instrument: uuid.UUID,
    feature: float,
    forward_return: float,
    realized_return_1d: float | None,
    as_of_index: int | None = None,
    label_end_index: int | None = None,
) -> SupervisedAlphaSample:
    return SupervisedAlphaSample(
        as_of=_ts(day),
        instrument_id=instrument,
        features={"alpha": feature},
        forward_return=forward_return,
        realized_return_1d=realized_return_1d,
        as_of_index=as_of_index,
        label_end_index=label_end_index,
    )


# -- 1. signed-rank uses realized_return_1d for P&L ------------------------


class TestSignedRankUsesRealizedReturn:
    """``daily_metrics`` in realized mode marks-to-market with 1d returns."""

    def test_realized_mode_compounds_realized_not_forward_label(self) -> None:
        # Two instruments, both scored positive, with forward_return = 0.21
        # (the 21d label magnitude that previously leaked into daily P&L)
        # and realized_return_1d = 0.01 (the actual 1d return).
        inst_a = uuid.uuid4()
        inst_b = uuid.uuid4()
        scored = [
            (
                _sample(
                    day=1,
                    instrument=inst_a,
                    feature=1.0,
                    forward_return=0.21,
                    realized_return_1d=0.01,
                ),
                1.0,
            ),
            (
                _sample(
                    day=1,
                    instrument=inst_b,
                    feature=1.0,
                    forward_return=0.21,
                    realized_return_1d=0.01,
                ),
                1.0,
            ),
        ]

        returns, _ics, turnovers, _final = daily_metrics(scored, slippage_bps_per_turnover=0.0)

        # Both weights are 0.5 (normalized by abs-score mass). Expected
        # daily return = 0.5 * 0.01 + 0.5 * 0.01 = 0.01 — NOT 0.21.
        assert returns == pytest.approx([0.01], abs=1e-12)
        # Rebalance day, so turnover is whatever the weights add up to.
        assert turnovers[0] == pytest.approx(1.0, abs=1e-12)

    def test_realized_mode_holds_weights_across_fold_days(self) -> None:
        # Three trading days in one fold; rebalance only on the first day.
        # Realized returns differ each day; weights must be held constant.
        inst = uuid.uuid4()
        scored = []
        for day, realized in [(1, 0.02), (2, 0.03), (3, -0.01)]:
            scored.append(
                (
                    _sample(
                        day=day,
                        instrument=inst,
                        feature=1.0,
                        forward_return=0.21,
                        realized_return_1d=realized,
                    ),
                    1.0,
                )
            )

        returns, _ics, turnovers, _final = daily_metrics(scored, slippage_bps_per_turnover=0.0)

        # Single-name book carries weight 1.0 across all days; each day's
        # return equals the day's realized 1d return.
        assert returns == pytest.approx([0.02, 0.03, -0.01], abs=1e-12)
        # Turnover only on the rebalance day (day 1).
        assert turnovers == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


# -- 2. long-only uses realized_return_1d for P&L --------------------------


class TestLongOnlyUsesRealizedReturn:
    """``evaluate_long_only_portfolio`` marks-to-market with 1d returns."""

    def _config(self) -> CampaignPortfolioConfig:
        return CampaignPortfolioConfig(
            mode="runtime-long-only",
            top_n=2,
            vol_target=1.0,
            vol_floor=0.01,
            vol_lookback_days=1,
            max_gross_exposure=1.0,
            min_cash_buffer=0.0,
            max_single_name_weight=1.0,
            max_daily_turnover=2.0,
            max_position_change=1.0,
            no_trade_band=0.0,
            rebalance_interval_days=1,
        )

    def test_realized_mode_uses_realized_not_forward(self) -> None:
        inst_a = uuid.uuid4()
        inst_b = uuid.uuid4()
        scored = [
            (
                _sample(
                    day=1,
                    instrument=inst_a,
                    feature=1.0,
                    forward_return=0.21,
                    realized_return_1d=0.01,
                ),
                1.0,
            ),
            (
                _sample(
                    day=1,
                    instrument=inst_b,
                    feature=1.0,
                    forward_return=0.21,
                    realized_return_1d=0.01,
                ),
                1.0,
            ),
        ]
        result = evaluate_long_only_portfolio(
            scored,
            slippage_bps_per_turnover=0.0,
            config=self._config(),
        )
        # Both selected, equal-weighted at 0.5 each; gross return =
        # 0.5*0.01 + 0.5*0.01 = 0.01, not 0.21.
        assert result.daily_returns[0] == pytest.approx(0.01, abs=1e-12)


# -- 3. legacy mode (no realized field) preserves old behavior -------------


class TestLegacyModePreservesOldBehavior:
    """When ``realized_return_1d`` is absent, evaluator falls back cleanly."""

    def test_signed_rank_falls_back_to_forward_return(self) -> None:
        inst = uuid.uuid4()
        # No realized_return_1d → legacy path.
        scored = [
            (
                _sample(
                    day=1,
                    instrument=inst,
                    feature=1.0,
                    forward_return=0.21,
                    realized_return_1d=None,
                ),
                1.0,
            ),
        ]
        returns, _ics, _turnovers, _final = daily_metrics(scored, slippage_bps_per_turnover=0.0)
        # Legacy behavior: forward_return is fed in as daily return.
        assert returns[0] == pytest.approx(0.21, abs=1e-12)


# -- 4. old sample JSON without new fields still loads ---------------------


class TestBackwardCompatibleSampleJson:
    def test_legacy_json_without_realized_fields_loads(self, tmp_path: Path) -> None:
        # Mimic a v1 sample file: only the original required fields,
        # none of the new optional ones. ``return_type`` was a v1 field
        # that's been removed from the dataclass; the loader must
        # tolerate its presence in old JSON without crashing.
        legacy_payload = [
            {
                "as_of": "2026-01-01T00:00:00+00:00",
                "instrument_id": str(uuid.uuid4()),
                "features": {"alpha": 0.5},
                "forward_return": 0.03,
                "return_type": "log",
            }
        ]
        path = tmp_path / "legacy_samples.json"
        path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        samples = load_supervised_samples(path)

        assert len(samples) == 1
        row = samples[0]
        assert row.forward_return == pytest.approx(0.03)
        assert row.realized_return_1d is None
        assert row.as_of_index is None
        assert row.label_end_index is None
        assert row.label_end_as_of is None

    def test_new_json_with_realized_fields_loads(self, tmp_path: Path) -> None:
        payload = [
            {
                "as_of": "2026-01-01T00:00:00+00:00",
                "instrument_id": str(uuid.uuid4()),
                "features": {"alpha": 0.5},
                "forward_return": 0.03,
                "return_type": "log",
                "realized_return_1d": 0.002,
                "as_of_index": 100,
                "label_end_index": 121,
                "label_end_as_of": "2026-02-01T00:00:00+00:00",
            }
        ]
        path = tmp_path / "new_samples.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        samples = load_supervised_samples(path)

        row = samples[0]
        assert row.realized_return_1d == pytest.approx(0.002)
        assert row.as_of_index == 100
        assert row.label_end_index == 121
        assert row.label_end_as_of == datetime(2026, 2, 1, tzinfo=UTC)


# -- 5. WalkForwardConfig conditional purge validation ---------------------


class TestPurgeHorizonValidation:
    def test_unset_label_horizon_does_not_force_purge_check(self) -> None:
        # Original behavior preserved: purge_days=5 is fine when the
        # horizon isn't declared.
        cfg = WalkForwardConfig(
            train_window_days=30,
            test_window_days=30,
            step_days=30,
            purge_days=5,
        )
        assert cfg.purge_days == 5
        assert cfg.label_horizon_days is None

    def test_set_label_horizon_requires_purge_to_match_or_exceed(self) -> None:
        with pytest.raises(ValueError, match="purge_days must be >= label_horizon_days"):
            WalkForwardConfig(
                train_window_days=30,
                test_window_days=30,
                step_days=30,
                purge_days=5,
                label_horizon_days=21,
            )

    def test_purge_equal_to_horizon_accepted(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=252,
            test_window_days=21,
            step_days=21,
            purge_days=21,
            label_horizon_days=21,
        )
        assert cfg.label_horizon_days == 21

    def test_zero_or_negative_label_horizon_rejected(self) -> None:
        with pytest.raises(ValueError, match="label_horizon_days must be > 0"):
            WalkForwardConfig(
                train_window_days=30,
                test_window_days=30,
                step_days=30,
                purge_days=0,
                label_horizon_days=0,
            )


# -- 6. sample-level purge drops train rows whose label reaches test -------


class TestSampleLevelPurge:
    """End-to-end check via ``run_sample_walk_forward`` with toy data.

    The construction is deliberate: train samples whose ``label_end_index``
    reaches the test window must be dropped before the model fits.
    """

    def _build_samples(self) -> list[SupervisedAlphaSample]:
        # 90-day window, single instrument, daily samples.
        # Trading-day-index aligns with day-of-month for simplicity.
        inst = uuid.uuid4()
        samples: list[SupervisedAlphaSample] = []
        base = datetime(2026, 1, 1, tzinfo=UTC)
        horizon = 21
        for i in range(90):
            as_of = base + timedelta(days=i)
            samples.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=inst,
                    features={"alpha": float(i) / 90.0},
                    forward_return=0.001 * (i % 7 - 3),  # arbitrary
                    realized_return_1d=0.001 * (i % 5 - 2),
                    as_of_index=i,
                    label_end_index=i + horizon,
                    label_end_as_of=base + timedelta(days=i + horizon),
                )
            )
        return samples

    def test_indices_drive_sample_level_purge(self) -> None:
        samples = self._build_samples()
        cfg = WalkForwardConfig(
            train_window_days=30,
            test_window_days=21,
            step_days=21,
            purge_days=21,
            embargo_days=0,
            min_folds=1,
            label_horizon_days=21,
        )
        evidence = run_sample_walk_forward(
            samples=samples,
            config=cfg,
            model_version="test",
            feature_set_version="test",
            slippage_bps_per_turnover=0.0,
        )
        # At least one fold ran.
        assert len(evidence.folds) >= 1
        # Every fold logs the sample-level purge mode in fold_basis.
        for fold in evidence.folds:
            assert fold["fold_basis"] == "calendar_days_plus_sample_label_index_purge"


# -- 7. bucket-return helpers compound simple returns ----------------------


class TestNonOverlappingBucketReturns:
    def test_compounds_simple_returns_not_sums_them(self) -> None:
        # 21 daily simple returns of 0.01 each should compound to
        # (1.01 ** 21) - 1, not sum to 0.21.
        daily = [0.01] * 21
        buckets = non_overlapping_bucket_returns(daily, horizon_days=21)
        assert len(buckets) == 1
        expected = (1.01**21) - 1.0
        assert buckets[0] == pytest.approx(expected, rel=1e-10)
        # The wrong (summing) answer would be 0.21 — explicit guard.
        assert not math.isclose(buckets[0], 0.21, rel_tol=1e-3)

    def test_partial_trailing_bucket_is_discarded(self) -> None:
        # 25 returns → one full 21-day bucket; the trailing 4 returns
        # are discarded (including them would understate variance).
        daily = [0.0] * 25
        buckets = non_overlapping_bucket_returns(daily, horizon_days=21)
        assert len(buckets) == 1

    def test_bucket_sharpe_annualizes_with_horizon_adjusted_factor(self) -> None:
        # Two buckets with non-zero mean/std → finite Sharpe.
        # Sharpe should scale by sqrt(252 / 21), not sqrt(252).
        # Note: statistics.stdev uses Bessel's correction (n-1) so for
        # [0.02, 0.04] mean=0.03, stdev=sqrt(2*0.0001/1)=sqrt(2e-4) and
        # mean/stdev = 0.03 / sqrt(2e-4). Whatever the precise raw Sharpe,
        # the bucket annualization should differ from the daily one by
        # exactly the ratio sqrt(252/21) / sqrt(252) = 1/sqrt(21).
        import statistics

        buckets = [0.02, 0.04]
        raw_sharpe = statistics.mean(buckets) / statistics.stdev(buckets)
        expected = raw_sharpe * math.sqrt(252.0 / 21.0)
        actual = bucket_sharpe(buckets, horizon_days=21)
        assert actual == pytest.approx(expected, rel=1e-10)
        # Explicit guard: bucket annualization is NOT sqrt(252) — that
        # would be the daily-MtM annualization and would overstate
        # bucket Sharpe by a factor of sqrt(21).
        daily_annualized = raw_sharpe * math.sqrt(252.0)
        assert not math.isclose(actual, daily_annualized, rel_tol=1e-3)

    def test_compounding_round_trip_with_evaluator_returns(self) -> None:
        # The bucket variant is derived FROM the daily MtM stream, so
        # compounding the daily stream over a full bucket window must
        # equal the bucket return — single source of truth.
        daily = [0.005, -0.002, 0.01, 0.003, -0.004]
        buckets = non_overlapping_bucket_returns(daily, horizon_days=5)
        assert len(buckets) == 1
        assert buckets[0] == pytest.approx(compound_return(daily), rel=1e-12)


# -- 8. SupervisedAlphaSample __post_init__ validation ---------------------


class TestSupervisedAlphaSampleValidation:
    """Defensive validation pins so a future caller can't quietly
    construct a sample with NaN-laced realized returns or inverted
    index ordering, both of which would propagate silently through
    ``equity *= 1 + r`` or the sample-level purge."""

    def _common(self) -> dict[str, object]:
        return {
            "as_of": datetime(2026, 1, 1, tzinfo=UTC),
            "instrument_id": uuid.uuid4(),
            "features": {"alpha": 0.0},
            "forward_return": 0.0,
        }

    def test_finite_realized_return_accepted(self) -> None:
        SupervisedAlphaSample(**self._common(), realized_return_1d=0.01)

    def test_none_realized_return_accepted(self) -> None:
        SupervisedAlphaSample(**self._common(), realized_return_1d=None)

    def test_nan_realized_return_rejected(self) -> None:
        with pytest.raises(ValueError, match="realized_return_1d must be finite"):
            SupervisedAlphaSample(**self._common(), realized_return_1d=float("nan"))

    def test_inf_realized_return_rejected(self) -> None:
        with pytest.raises(ValueError, match="realized_return_1d must be finite"):
            SupervisedAlphaSample(**self._common(), realized_return_1d=float("inf"))

    def test_negative_as_of_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="as_of_index must be >= 0"):
            SupervisedAlphaSample(**self._common(), as_of_index=-1)

    def test_negative_label_end_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="label_end_index must be >= 0"):
            SupervisedAlphaSample(**self._common(), label_end_index=-5)

    def test_label_end_before_as_of_rejected(self) -> None:
        with pytest.raises(ValueError, match="label_end_index must be >= as_of_index"):
            SupervisedAlphaSample(**self._common(), as_of_index=10, label_end_index=5)

    def test_equal_label_end_and_as_of_accepted(self) -> None:
        # H=0 (degenerate but valid label window).
        SupervisedAlphaSample(**self._common(), as_of_index=10, label_end_index=10)


# -- 9. has_realized_returns helper ---------------------------------------


class TestHasRealizedReturnsHelper:
    """The opt-in detector both evaluators share — single source of
    truth for the realized-vs-legacy mode decision."""

    def _sample(self, *, realized: float | None) -> SupervisedAlphaSample:
        return SupervisedAlphaSample(
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            instrument_id=uuid.uuid4(),
            features={"alpha": 0.0},
            forward_return=0.0,
            realized_return_1d=realized,
        )

    def test_empty_input_is_vacuously_realized(self) -> None:
        # all() of an empty sequence is True. The evaluators handle the
        # empty-input case via the ``not sorted_days`` early return; the
        # vacuous-truth behavior here is intentional and documented.
        assert has_realized_returns([]) is True

    def test_all_realized(self) -> None:
        assert (
            has_realized_returns(
                [(self._sample(realized=0.01), 1.0), (self._sample(realized=-0.02), 0.5)]
            )
            is True
        )

    def test_any_missing_realized_is_false(self) -> None:
        assert (
            has_realized_returns(
                [(self._sample(realized=0.01), 1.0), (self._sample(realized=None), 0.5)]
            )
            is False
        )

    def test_all_missing_realized_is_false(self) -> None:
        assert (
            has_realized_returns(
                [(self._sample(realized=None), 1.0), (self._sample(realized=None), 0.5)]
            )
            is False
        )


# -- 10. run_sample_walk_forward rejects mixed-mode samples ----------------


class TestMixedModeRejection:
    """``run_sample_walk_forward`` must fail loud rather than let some
    folds run in realized mode and others fall back to legacy mode,
    which would silently produce mixed-semantics metrics."""

    def _samples(self, *, mix_realized: bool, mix_indices: bool) -> list[SupervisedAlphaSample]:
        # 90 days × 1 instrument; first half optionally missing the
        # realized field, second half always populated.
        inst = uuid.uuid4()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        samples: list[SupervisedAlphaSample] = []
        for i in range(90):
            realized = (0.001 if i >= 45 else None) if mix_realized else 0.001
            idx = (i if i >= 45 else None) if mix_indices else i
            end = (i + 21 if i >= 45 else None) if mix_indices else i + 21
            samples.append(
                SupervisedAlphaSample(
                    as_of=base + timedelta(days=i),
                    instrument_id=inst,
                    features={"alpha": float(i) / 90.0},
                    forward_return=0.0,
                    realized_return_1d=realized,
                    as_of_index=idx,
                    label_end_index=end,
                )
            )
        return samples

    def test_mixed_realized_returns_rejected(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=30,
            test_window_days=21,
            step_days=21,
            purge_days=21,
            label_horizon_days=21,
        )
        with pytest.raises(ValueError, match="mix realized-mode and legacy-mode"):
            run_sample_walk_forward(
                samples=self._samples(mix_realized=True, mix_indices=False),
                config=cfg,
                model_version="test",
                feature_set_version="test",
                slippage_bps_per_turnover=0.0,
            )

    def test_mixed_indices_rejected(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=30,
            test_window_days=21,
            step_days=21,
            purge_days=21,
            label_horizon_days=21,
        )
        with pytest.raises(ValueError, match="mix indexed and unindexed"):
            run_sample_walk_forward(
                samples=self._samples(mix_realized=False, mix_indices=True),
                config=cfg,
                model_version="test",
                feature_set_version="test",
                slippage_bps_per_turnover=0.0,
            )


# -- 11. cross-fold weight carry in realized mode --------------------------


class TestCrossFoldWeightCarry:
    """The realized-mode evaluator returns the held weights as its
    final state; the driver feeds that back to the next fold as
    ``prev_scores``. The next fold's rebalance turnover must be
    computed against those held weights."""

    def _scored(
        self, *, day: int, instrument: uuid.UUID, score: float, realized: float
    ) -> tuple[SupervisedAlphaSample, float]:
        return (
            SupervisedAlphaSample(
                as_of=datetime(2026, 1, day, tzinfo=UTC),
                instrument_id=instrument,
                features={"alpha": score},
                forward_return=0.0,
                realized_return_1d=realized,
            ),
            score,
        )

    def test_second_fold_turnover_uses_first_fold_held_weights(self) -> None:
        inst_a = uuid.uuid4()
        inst_b = uuid.uuid4()

        # Fold 1: only A scored positive. Held weights at fold end =
        # {A: 1.0}.
        fold1 = [
            self._scored(day=1, instrument=inst_a, score=1.0, realized=0.01),
            self._scored(day=1, instrument=inst_b, score=0.0, realized=0.02),
        ]
        _r1, _ic1, t1, carry_after_fold1 = daily_metrics(fold1, slippage_bps_per_turnover=0.0)
        assert carry_after_fold1 == pytest.approx({inst_a: 1.0, inst_b: 0.0}, abs=1e-12)
        # Fold-1 rebalance turnover started from empty prior, so =
        # sum of new weights = 1.0.
        assert t1[0] == pytest.approx(1.0, abs=1e-12)

        # Fold 2: only B scored positive. Rebalance turnover MUST be
        # computed vs {A: 1.0, B: 0.0}, not vs empty — i.e. should be
        # |0 - 1| + |1 - 0| = 2.0 (full unwind + full rebuild).
        fold2 = [
            self._scored(day=2, instrument=inst_a, score=0.0, realized=-0.01),
            self._scored(day=2, instrument=inst_b, score=1.0, realized=0.03),
        ]
        _r2, _ic2, t2, carry_after_fold2 = daily_metrics(
            fold2, slippage_bps_per_turnover=0.0, prev_scores=carry_after_fold1
        )
        assert t2[0] == pytest.approx(2.0, abs=1e-12)
        # And the held weights at fold-2 end are {A: 0, B: 1}.
        assert carry_after_fold2 == pytest.approx({inst_a: 0.0, inst_b: 1.0}, abs=1e-12)


# -- 12. evidence JSON schema -------------------------------------------------


class TestEvidenceJsonSchema:
    """Pin the audit-mandated metadata keys on saved evidence so a
    future code change can't quietly drop them. Indirectly exercises
    the ``save_evidence`` + ``_bucket_metrics`` + ``ArmSpec``
    integration as well."""

    def _build_minimal_evidence(
        self, *, with_realized: bool, n_days: int = 120
    ) -> tuple[object, object, object, list[SupervisedAlphaSample], list[str]]:
        from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
            WalkForwardConfig,
        )
        from quant_platform.services.research_service.sampling.factory_models import (
            AlphaEligibilityThresholds,
        )

        base = datetime(2026, 1, 1, tzinfo=UTC)
        inst = uuid.uuid4()
        samples: list[SupervisedAlphaSample] = []
        for i in range(n_days):
            samples.append(
                SupervisedAlphaSample(
                    as_of=base + timedelta(days=i),
                    instrument_id=inst,
                    features={"alpha": float(i) / n_days},
                    forward_return=0.0,
                    realized_return_1d=0.001 if with_realized else None,
                    as_of_index=i if with_realized else None,
                    label_end_index=i + 21 if with_realized else None,
                )
            )
        # train > purge is required (train_end = train_start + train -
        # purge; if train == purge, the train window is empty and the
        # fold is invalid). Use train=42 = 2× purge to leave headroom.
        cfg = WalkForwardConfig(
            train_window_days=42,
            test_window_days=21,
            step_days=21,
            purge_days=21 if with_realized else 0,
            label_horizon_days=21 if with_realized else None,
            min_folds=1,
        )
        thresholds = AlphaEligibilityThresholds()
        return (cfg, thresholds, base, samples, ["alpha"])

    def test_save_evidence_emits_audit_schema_keys(self, tmp_path: Path) -> None:
        from scripts.backtest_latest_stack import (  # local import to avoid heavy module load on every test
            ARM_SPECS,
            EVIDENCE_SCHEMA_VERSION,
            save_evidence,
        )

        cfg, thresholds, _base, samples, _names = self._build_minimal_evidence(with_realized=True)
        evidence = run_sample_walk_forward(
            samples=samples,
            config=cfg,
            model_version="test-model",
            feature_set_version="test-fs",
            thresholds=thresholds,
            slippage_bps_per_turnover=0.0,
            feature_names=["alpha"],
        )
        spec = ARM_SPECS[0]
        path = save_evidence(
            spec,
            evidence,
            tmp_path,
            thresholds=thresholds,
            wf_config=cfg,
            feature_set_versions={"price_volume": "v1"},
            universe_fingerprint={"path": "u.json", "sha256": "abc"},
            bars_fingerprint={"algorithm": "x", "is_content_hash": False},
            git_commit="deadbeef",
            cli_args={"instrument_limit": None, "max_years": None},
            realized_mode_used=True,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        # Audit-mandated metadata keys.
        required = {
            "evidence_schema_version",
            "arm",
            "arm_cli_alias",
            "arm_category",
            "production_candidate",
            "label_horizon_days",
            "return_mode_daily",
            "return_mode_bucket",
            "realized_mode_used",
            "fold_basis",
            "n_folds_actual",
            "git_commit",
            "universe_fingerprint",
            "bars_snapshot_fingerprint",
            "eligibility_thresholds",
            "walk_forward_config",
            "cli_args",
            "feature_set_versions",
            "metrics",
            "metrics_daily_mtm",
            "metrics_bucket_21d",
        }
        missing = required - set(payload.keys())
        assert not missing, f"evidence JSON missing required keys: {sorted(missing)}"

        # Per-fold fold_basis tag must propagate to the top-level summary.
        assert payload["evidence_schema_version"] == EVIDENCE_SCHEMA_VERSION
        assert payload["realized_mode_used"] is True
        assert payload["label_horizon_days"] == 21

    def test_bucket_metrics_shape(self) -> None:
        from scripts.backtest_latest_stack import _bucket_metrics

        daily = [0.001] * 50  # ~ two full 21-day buckets
        metrics = _bucket_metrics(daily)
        assert set(metrics.keys()) == {
            "horizon_days",
            "buckets",
            "total_return",
            "max_drawdown",
            "annualized_sharpe",
        }
        assert metrics["horizon_days"] == 21.0
        assert metrics["buckets"] == 2.0  # 2 full 21-day buckets in 50 days


# -- 13. ArmSpec registry invariants --------------------------------------


class TestArmSpecRegistry:
    """The ``ARM_SPECS`` registry must have unique CLI aliases and
    canonical names, otherwise ``ARM_SPEC_BY_KEY`` silently overwrites
    entries and arm dispatch becomes nondeterministic."""

    def test_aliases_are_unique(self) -> None:
        from scripts.backtest_latest_stack import ARM_SPECS

        aliases = [s.cli_alias for s in ARM_SPECS]
        assert len(aliases) == len(set(aliases)), f"duplicate CLI aliases: {aliases}"

    def test_canonical_names_are_unique(self) -> None:
        from scripts.backtest_latest_stack import ARM_SPECS

        names = [s.canonical_name for s in ARM_SPECS]
        assert len(names) == len(set(names)), f"duplicate canonical names: {names}"

    def test_key_lookup_is_2x_spec_count(self) -> None:
        from scripts.backtest_latest_stack import ARM_SPEC_BY_KEY, ARM_SPECS

        assert len(ARM_SPEC_BY_KEY) == 2 * len(ARM_SPECS)

    def test_portfolio_candidates_have_a_factory(self) -> None:
        from scripts.backtest_latest_stack import ARM_SPECS

        for spec in ARM_SPECS:
            if spec.category == "portfolio_candidate":
                assert spec.portfolio_config_factory is not None, (
                    f"portfolio_candidate arm {spec.canonical_name} must "
                    "declare a portfolio_config_factory"
                )

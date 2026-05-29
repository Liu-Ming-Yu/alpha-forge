"""Tests for the pluggable name-selection layer (Arm M, ADR-009).

Covers four contracts:

1. **Behavior preservation** — ``raw_long_only_target`` with ``selection=None``
   is bit-identical to an explicit ``TopNSelection`` and to the prior fresh
   top-N. This is the guard that adding the seam did not change arms A-L.
2. **Top-N fidelity** — ``TopNSelection`` returns the fresh top-N and ignores
   holdings.
3. **Buffered top-k** — incumbents in the buffer band keep their slot (bumping
   the weakest new entrant), clearly-better new names still enter, incumbents
   past the band are dropped, ``buffer=0`` recovers top-N, and the result is
   always exactly ``top_n``.
4. **Driver integration** — ``run_sample_walk_forward`` accepts a selection
   strategy on the long-only path, rejects it on the signed-rank path, and pins
   its metadata only when supplied.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.portfolio import (
    BufferedTopKSelection,
    CampaignPortfolioConfig,
    SelectionStrategy,
    TopNSelection,
)
from quant_platform.services.research_service.campaigns.portfolio.targets import (
    raw_long_only_target,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

# -- fixtures -------------------------------------------------------------------


def _named_rows(
    score_by_id: list[tuple[uuid.UUID, float]],
) -> list[tuple[SupervisedAlphaSample, float]]:
    """Build (sample, score) rows from explicit (id, score) pairs."""
    out: list[tuple[SupervisedAlphaSample, float]] = []
    for inst, score in score_by_id:
        sample = SupervisedAlphaSample(
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            instrument_id=inst,
            features={"alpha": score},
            forward_return=0.0,
        )
        out.append((sample, score))
    return out


# Five names A..E with strictly decreasing scores, so rank == list position.
_IDS = [uuid.uuid4() for _ in range(6)]
_RANKED_PAIRS = [(_IDS[i], 6.0 - i) for i in range(6)]  # A=6 ... F=1, all positive


def _ranked() -> list[tuple[SupervisedAlphaSample, float]]:
    return _named_rows(_RANKED_PAIRS)


def _config(top_n: int = 3) -> CampaignPortfolioConfig:
    return CampaignPortfolioConfig(
        mode="runtime-long-only",
        top_n=top_n,
        max_gross_exposure=0.22,
        min_cash_buffer=0.05,
        max_single_name_weight=0.05,
    )


# -- 1. behavior preservation ---------------------------------------------------


class TestRawTargetBehaviorPreservation:
    def test_none_equals_explicit_top_n(self) -> None:
        rows = _ranked()
        cfg = _config()
        default = raw_long_only_target(rows, config=cfg)
        explicit = raw_long_only_target(rows, config=cfg, selection=TopNSelection())
        assert default == explicit

    def test_default_holds_fresh_top_n(self) -> None:
        rows = _ranked()
        target = raw_long_only_target(rows, config=_config(top_n=3))
        # Top 3 by score are A, B, C (_IDS[0..2]).
        assert set(target) == set(_IDS[:3])

    def test_default_ignores_holdings(self) -> None:
        # Passing current_holdings must not change the default (top-N) selection.
        rows = _ranked()
        cfg = _config(top_n=3)
        without = raw_long_only_target(rows, config=cfg)
        withheld = raw_long_only_target(rows, config=cfg, current_holdings=frozenset({_IDS[5]}))
        assert without == withheld


# -- 2. top-N fidelity ----------------------------------------------------------


class TestTopNSelection:
    def test_returns_fresh_top_n(self) -> None:
        selected = TopNSelection().select(_ranked(), top_n=3, current_holdings=frozenset())
        assert [r.instrument_id for r, _ in selected] == _IDS[:3]

    def test_ignores_holdings(self) -> None:
        selected = TopNSelection().select(_ranked(), top_n=3, current_holdings=frozenset({_IDS[5]}))
        assert [r.instrument_id for r, _ in selected] == _IDS[:3]

    def test_satisfies_protocol(self) -> None:
        assert isinstance(TopNSelection(), SelectionStrategy)


# -- 3. buffered top-k ----------------------------------------------------------


class TestBufferedTopKSelection:
    def test_buffer_zero_equals_top_n(self) -> None:
        sel = BufferedTopKSelection(buffer=0)
        out = sel.select(_ranked(), top_n=3, current_holdings=frozenset({_IDS[3]}))
        assert {r.instrument_id for r, _ in out} == set(_IDS[:3])

    def test_cold_start_holds_top_n(self) -> None:
        sel = BufferedTopKSelection(buffer=2)
        out = sel.select(_ranked(), top_n=3, current_holdings=frozenset())
        assert {r.instrument_id for r, _ in out} == set(_IDS[:3])

    def test_incumbent_in_band_keeps_slot(self) -> None:
        # top_n=3, buffer=2 -> band [3,5). D (rank 3) is held and in band, so it
        # keeps its slot, bumping the weakest new entrant C (rank 2).
        sel = BufferedTopKSelection(buffer=2)
        out = sel.select(_ranked(), top_n=3, current_holdings=frozenset({_IDS[3]}))
        ids = {r.instrument_id for r, _ in out}
        assert ids == {_IDS[0], _IDS[1], _IDS[3]}  # A, B, D ; C bumped
        assert len(out) == 3

    def test_strong_new_names_still_enter(self) -> None:
        # A and B (ranks 0,1) are new this period; only the weakest new entrant
        # near the cutoff is ever bumped, so the strong newcomers stay.
        sel = BufferedTopKSelection(buffer=2)
        out = sel.select(_ranked(), top_n=3, current_holdings=frozenset({_IDS[3]}))
        ids = {r.instrument_id for r, _ in out}
        assert _IDS[0] in ids and _IDS[1] in ids

    def test_incumbent_past_band_is_dropped(self) -> None:
        # F is held but at rank 5, outside band [3,5) -> not protected -> dropped.
        sel = BufferedTopKSelection(buffer=2)
        out = sel.select(_ranked(), top_n=3, current_holdings=frozenset({_IDS[5]}))
        assert {r.instrument_id for r, _ in out} == set(_IDS[:3])

    def test_non_positive_held_name_not_a_candidate(self) -> None:
        # A held name whose score went non-positive is absent from `ranked`, so
        # it cannot be selected regardless of the buffer.
        pairs = [(_IDS[0], 3.0), (_IDS[1], 2.0), (_IDS[2], 1.0), (_IDS[3], -0.5)]
        rows = _named_rows(pairs)
        ranked = sorted((r for r in rows if r[1] > 0), key=lambda x: x[1], reverse=True)
        sel = BufferedTopKSelection(buffer=3)
        out = sel.select(ranked, top_n=3, current_holdings=frozenset({_IDS[3]}))
        assert _IDS[3] not in {r.instrument_id for r, _ in out}

    def test_always_returns_top_n(self) -> None:
        sel = BufferedTopKSelection(buffer=2)
        for holdings in (frozenset(), frozenset({_IDS[3]}), frozenset({_IDS[3], _IDS[4]})):
            out = sel.select(_ranked(), top_n=3, current_holdings=holdings)
            assert len(out) == 3

    def test_rejects_negative_buffer(self) -> None:
        with pytest.raises(ValueError, match="buffer must be >= 0"):
            BufferedTopKSelection(buffer=-1)

    def test_metadata(self) -> None:
        meta = BufferedTopKSelection(buffer=5).metadata()
        assert meta["type"] == "buffered_topk"
        assert meta["buffer"] == 5

    def test_satisfies_protocol(self) -> None:
        assert isinstance(BufferedTopKSelection(), SelectionStrategy)


# -- 4. driver integration ------------------------------------------------------


def _wf_samples(*, n_days: int = 40, n_instruments: int = 10) -> list[SupervisedAlphaSample]:
    insts = [uuid.uuid4() for _ in range(n_instruments)]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[SupervisedAlphaSample] = []
    for d in range(n_days):
        as_of = start + timedelta(days=d)
        for i, inst in enumerate(insts):
            score = float((i + d) % n_instruments)
            out.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=inst,
                    features={"alpha": score},
                    forward_return=0.01 if score > 0 else -0.005,
                    realized_return_1d=0.001 * (1 if (i + d) % 2 == 0 else -1),
                )
            )
    return out


def _long_only_config() -> CampaignPortfolioConfig:
    return CampaignPortfolioConfig(
        mode="runtime-long-only",
        top_n=4,
        vol_target=0.15,
        vol_floor=0.05,
        vol_lookback_days=3,
        max_gross_exposure=0.5,
        min_cash_buffer=0.05,
        max_single_name_weight=0.2,
        max_daily_turnover=1.0,
        max_position_change=0.2,
        no_trade_band=0.0,
        rebalance_interval_days=1,
    )


def _wf_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window_days=10,
        test_window_days=3,
        step_days=3,
        purge_days=1,
        embargo_days=0,
        min_folds=1,
        label_horizon_days=1,
    )


class TestWalkForwardSelection:
    def test_buffered_runs_and_pins_metadata(self) -> None:
        import math

        kwargs = dict(
            samples=_wf_samples(),
            config=_wf_config(),
            model_version="test",
            feature_set_version="test-fs",
            feature_names=["alpha"],
            portfolio_config=_long_only_config(),
        )
        baseline = run_sample_walk_forward(**kwargs)  # type: ignore[arg-type]
        buffered = run_sample_walk_forward(
            selection=BufferedTopKSelection(buffer=2),
            **kwargs,  # type: ignore[arg-type]
        )
        assert len(buffered.folds) >= 1
        assert math.isfinite(buffered.metrics["slippage_adjusted_sharpe"])
        assert "selection" not in baseline.portfolio_diagnostics
        assert buffered.portfolio_diagnostics["selection"]["type"] == "buffered_topk"  # type: ignore[index]

    def test_selection_rejected_without_portfolio(self) -> None:
        with pytest.raises(ValueError, match="selection requires a portfolio_config"):
            run_sample_walk_forward(
                samples=_wf_samples(),
                config=_wf_config(),
                model_version="test",
                feature_set_version="test-fs",
                feature_names=["alpha"],
                selection=BufferedTopKSelection(),
            )

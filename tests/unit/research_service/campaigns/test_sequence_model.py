"""Tests for the GRU sequence-model ranker (Arm N, ADR-010).

Two tiers:

1. **Torch-free** — the sequence-construction core (`_Standardizer`,
   `build_train_sequences`, `build_score_sequences`, padding/masking, train-tail
   stitch, gap handling, time-gap channel, order round-trip) and the config
   validation. These always run, including in CI without the `dl` extra — they
   are the critical PIT-safety coverage.
2. **Torch-gated** (`@_needs_torch`, forced `device="cpu"`, tiny epochs + fixed
   seed) — fit/score shapes, protocol satisfaction, IC-loss-learns-a-signal,
   CPU determinism, and `run_sample_walk_forward` integration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.models import (
    AlphaModel,
    FittedAlphaModel,
    GRUSequenceRanker,
)
from quant_platform.services.research_service.campaigns.models.sequence import (
    _Standardizer,
    build_score_sequences,
    build_train_sequences,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.reports.statistics import spearman_ic
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch (dl extra) not installed")

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _row(
    inst: uuid.UUID, idx: int, f1: float, *, f2: float = 0.0, fwd: float = 0.0
) -> SupervisedAlphaSample:
    """A sample with as_of_index=idx (and as_of derived from it)."""
    return SupervisedAlphaSample(
        as_of=_START + timedelta(days=idx),
        instrument_id=inst,
        features={"f1": f1, "f2": f2},
        forward_return=fwd,
        as_of_index=idx,
    )


# -- 1a. standardizer -----------------------------------------------------------


class TestStandardizer:
    def test_pit_safe_uses_train_stats(self) -> None:
        train = np.array([[0.0], [2.0], [4.0]], dtype="float32")  # mean 2, std ~1.633
        std = _Standardizer.fit(train)
        out = std.transform(np.array([[2.0]], dtype="float32"))
        # (2 - 2) / train_std == 0, regardless of the test row's own stats.
        assert out[0, 0] == pytest.approx(0.0)
        # A known train row standardizes to (x - mean)/popstd.
        z0 = std.transform(np.array([[0.0]], dtype="float32"))[0, 0]
        assert z0 == pytest.approx((0.0 - 2.0) / np.std(train), abs=1e-5)

    def test_constant_feature_no_div_by_zero(self) -> None:
        std = _Standardizer.fit(np.array([[5.0], [5.0], [5.0]], dtype="float32"))
        out = std.transform(np.array([[5.0], [7.0]], dtype="float32"))
        assert out[0, 0] == pytest.approx(0.0)  # centered; std forced to 1
        assert np.isfinite(out).all()


# -- 1b. windowing / padding / mask --------------------------------------------


def _std_for(rows: list[SupervisedAlphaSample], names: list[str]) -> _Standardizer:
    from quant_platform.services.research_service.campaigns.models.sequence import _matrix

    return _Standardizer.fit(_matrix(rows, names))


class TestWindowing:
    def test_full_history_mask_all_ones(self) -> None:
        inst = uuid.uuid4()
        rows = [_row(inst, i, float(i)) for i in range(5)]
        std = _std_for(rows, ["f1"])
        batch, _ = build_train_sequences(rows, ["f1"], std, window=3, include_time_gap=False)
        # The target at pos 4 (last) has a full 3-step window, mask all ones.
        last = batch.order.index(4)
        assert batch.mask[last].tolist() == [1.0, 1.0, 1.0]

    def test_short_history_is_left_padded(self) -> None:
        inst = uuid.uuid4()
        rows = [_row(inst, i, float(i)) for i in range(5)]
        std = _std_for(rows, ["f1"])
        batch, _ = build_train_sequences(rows, ["f1"], std, window=3, include_time_gap=False)
        first = batch.order.index(0)  # only 1 observation available
        assert batch.mask[first].tolist() == [0.0, 0.0, 1.0]  # left pad, trailing real
        assert batch.sequences[first, 0].tolist() == [0.0]  # padded step is zeros

    def test_train_tail_caches_last_window_minus_one(self) -> None:
        inst = uuid.uuid4()
        rows = [_row(inst, i, float(i)) for i in range(5)]
        std = _std_for(rows, ["f1"])
        _, tail = build_train_sequences(rows, ["f1"], std, window=3, include_time_gap=False)
        tail_feat, tail_idx = tail[inst]
        assert tail_feat.shape[0] == 2  # window - 1
        assert tail_idx == [3, 4]  # last two as_of_index values

    def test_score_stitches_train_tail(self) -> None:
        inst = uuid.uuid4()
        train = [_row(inst, i, float(i)) for i in range(5)]  # idx 0..4
        std = _std_for(train, ["f1"])
        _, tail = build_train_sequences(train, ["f1"], std, window=3, include_time_gap=False)
        test = [_row(inst, 10, 99.0)]  # a later test obs
        batch = build_score_sequences(
            test, ["f1"], std, window=3, include_time_gap=False, train_tail=tail
        )
        # The single test row's 3-window = [tail_feat[0], tail_feat[1], test0].
        seq = batch.sequences[0]
        expected_tail = std.transform(np.array([[3.0], [4.0]], dtype="float32"))
        assert seq[0].tolist() == pytest.approx(expected_tail[0].tolist())
        assert seq[1].tolist() == pytest.approx(expected_tail[1].tolist())
        assert batch.mask[0].tolist() == [1.0, 1.0, 1.0]

    def test_scoring_train_rows_does_not_prepend_future_tail(self) -> None:
        # Regression: the driver calls score(train) for the volatility scale, so
        # build_score receives rows that OVERLAP the tail's time range. The tail
        # (the LATEST window-1 train vectors) must NOT be prepended in front of
        # the EARLIER train rows — that would scramble time order and leak future
        # train features. score(train) windows must match build_train exactly.
        inst = uuid.uuid4()
        rows = [_row(inst, i, float(i)) for i in range(5)]  # idx 0..4
        std = _std_for(rows, ["f1"])
        train_batch, tail = build_train_sequences(
            rows, ["f1"], std, window=3, include_time_gap=False
        )
        score_batch = build_score_sequences(
            rows, ["f1"], std, window=3, include_time_gap=False, train_tail=tail
        )
        # The earliest train row (idx 0) must be left-padded to a single real
        # step — NOT filled with the future tail vectors (idx 3, 4).
        first = score_batch.order.index(0)
        assert score_batch.mask[first].tolist() == [0.0, 0.0, 1.0]
        assert np.allclose(
            score_batch.sequences[first, -1], std.transform(np.array([[0.0]], dtype="float32"))[0]
        )
        # And scoring the train set reproduces the training windows exactly.
        for tgt in range(5):
            ti = train_batch.order.index(tgt)
            si = score_batch.order.index(tgt)
            assert np.allclose(score_batch.sequences[si], train_batch.sequences[ti])
            assert score_batch.mask[si].tolist() == train_batch.mask[ti].tolist()

    def test_new_instrument_without_tail(self) -> None:
        train_inst, new_inst = uuid.uuid4(), uuid.uuid4()
        train = [_row(train_inst, i, float(i)) for i in range(4)]
        std = _std_for(train, ["f1"])
        _, tail = build_train_sequences(train, ["f1"], std, window=3, include_time_gap=False)
        test = [_row(new_inst, 10, 1.0), _row(new_inst, 11, 2.0)]
        batch = build_score_sequences(
            test, ["f1"], std, window=3, include_time_gap=False, train_tail=tail
        )
        # No tail for the new instrument -> its first row is heavily padded.
        first = batch.order.index(0)
        assert batch.mask[first].tolist() == [0.0, 0.0, 1.0]
        assert len(batch.order) == 2

    def test_order_is_a_permutation(self) -> None:
        insts = [uuid.uuid4() for _ in range(3)]
        rows = [_row(insts[i % 3], i, float(i)) for i in range(12)]
        std = _std_for(rows, ["f1"])
        batch, _ = build_train_sequences(rows, ["f1"], std, window=4, include_time_gap=False)
        assert sorted(batch.order) == list(range(12))


class TestTimeGapChannel:
    def test_gap_channel_appended_and_valued(self) -> None:
        inst = uuid.uuid4()
        # as_of_index 0, 1, 5 -> gaps 0, 1, 4 -> log1p
        rows = [_row(inst, 0, 1.0), _row(inst, 1, 1.0), _row(inst, 5, 1.0)]
        std = _std_for(rows, ["f1", "f2"])
        batch, _ = build_train_sequences(rows, ["f1", "f2"], std, window=3, include_time_gap=True)
        # n_channels = 2 features + 1 gap = 3
        assert batch.sequences.shape[2] == 3
        last = batch.order.index(2)  # full window [obs0, obs1, obs2]
        gap_col = batch.sequences[last, :, -1]
        assert gap_col[0] == pytest.approx(np.log1p(0))  # first step gap 0
        assert gap_col[1] == pytest.approx(np.log1p(1))  # 1 - 0
        assert gap_col[2] == pytest.approx(np.log1p(4))  # 5 - 1

    def test_gap_handles_noncontiguous_index_as_adjacent(self) -> None:
        # A halt (gap in as_of_index) does not drop the observation — it is kept
        # adjacent in the sequence, with the gap recorded only in the gap channel.
        inst = uuid.uuid4()
        rows = [_row(inst, 0, 1.0), _row(inst, 50, 2.0)]  # 50-day halt
        std = _std_for(rows, ["f1"])
        batch, _ = build_train_sequences(rows, ["f1"], std, window=2, include_time_gap=True)
        last = batch.order.index(1)
        assert batch.mask[last].tolist() == [1.0, 1.0]  # both observations present, adjacent


# -- 1c. config validation (no torch needed) -----------------------------------


class TestGRUConfig:
    def test_rejects_bad_objective(self) -> None:
        with pytest.raises(ValueError, match="unsupported objective"):
            GRUSequenceRanker(objective="ndcg")  # type: ignore[arg-type]

    def test_rejects_bad_device(self) -> None:
        with pytest.raises(ValueError, match="unsupported device"):
            GRUSequenceRanker(device="gpu")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window": 0}, "window must be >= 1"),
            ({"hidden_size": 0}, "hidden_size must be >= 1"),
            ({"num_layers": 0}, "num_layers must be >= 1"),
            ({"epochs": 0}, "epochs must be >= 1"),
            ({"lr": 0.0}, "lr must be > 0"),
        ],
    )
    def test_rejects_bad_hyperparams(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            GRUSequenceRanker(**kwargs)  # type: ignore[arg-type]

    def test_objective_encoded_in_name(self) -> None:
        assert GRUSequenceRanker(objective="ic").name == "gru-ranker-ic-v1"
        assert GRUSequenceRanker(objective="mse").name == "gru-ranker-mse-v1"

    def test_name_hardware_independent(self) -> None:
        assert GRUSequenceRanker(device="cpu").name == GRUSequenceRanker(device="auto").name


# -- 2. torch-gated model tests ------------------------------------------------


def _signal_samples(
    *, n_days: int = 30, n_instruments: int = 12, seed: int = 0
) -> list[SupervisedAlphaSample]:
    """Each instrument has a persistent quality q; f1 ~= q and forward ~= q.

    A last-step GRU read-out over the f1 sequence should recover positive IC.
    """
    rng = np.random.default_rng(seed)
    insts = [uuid.uuid4() for _ in range(n_instruments)]
    qualities = rng.normal(0.0, 1.0, size=n_instruments)
    out: list[SupervisedAlphaSample] = []
    for d in range(n_days):
        for i, inst in enumerate(insts):
            q = float(qualities[i])
            f1 = q + float(rng.normal(0.0, 0.1))
            fwd = 0.5 * q + float(rng.normal(0.0, 0.1))
            out.append(
                SupervisedAlphaSample(
                    as_of=_START + timedelta(days=d),
                    instrument_id=inst,
                    features={"f1": f1, "f2": float(rng.normal())},
                    forward_return=fwd,
                    realized_return_1d=0.001 * (1 if (i + d) % 2 == 0 else -1),
                    as_of_index=d,
                )
            )
    return out


@_needs_torch
class TestGRUSequenceRanker:
    def test_fit_score_shapes(self) -> None:
        samples = _signal_samples()
        fitted = GRUSequenceRanker(device="cpu", epochs=3, window=4).fit(samples, ["f1", "f2"])
        scores = fitted.score(samples[:10])
        assert len(scores) == 10
        assert all(isinstance(v, float) and np.isfinite(v) for v in scores)

    def test_empty_score_returns_empty(self) -> None:
        fitted = GRUSequenceRanker(device="cpu", epochs=2, window=3).fit(
            _signal_samples(), ["f1", "f2"]
        )
        assert fitted.score([]) == []

    def test_satisfies_protocols(self) -> None:
        model = GRUSequenceRanker(device="cpu", epochs=2, window=3)
        assert isinstance(model, AlphaModel)
        assert isinstance(model.fit(_signal_samples(), ["f1", "f2"]), FittedAlphaModel)

    def test_feature_weights_normalized_over_named_features(self) -> None:
        fitted = GRUSequenceRanker(device="cpu", epochs=2, window=3).fit(
            _signal_samples(), ["f1", "f2"]
        )
        w = dict(fitted.feature_weights())
        assert set(w) == {"f1", "f2"}  # excludes the time-gap channel
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0.0 for v in w.values())

    def test_ic_loss_learns_known_signal(self) -> None:
        samples = _signal_samples(n_days=40, n_instruments=20, seed=1)
        fitted = GRUSequenceRanker(objective="ic", device="cpu", epochs=60, window=5, seed=0).fit(
            samples, ["f1", "f2"]
        )
        scores = fitted.score(samples)
        labels = [s.forward_return for s in samples]
        ic = spearman_ic(scores, labels)
        assert ic > 0.1  # the GRU recovered the persistent signal

    def test_mse_objective_runs(self) -> None:
        fitted = GRUSequenceRanker(objective="mse", device="cpu", epochs=3, window=4).fit(
            _signal_samples(), ["f1", "f2"]
        )
        assert len(fitted.score(_signal_samples()[:5])) == 5

    def test_cpu_determinism(self) -> None:
        samples = _signal_samples(seed=2)
        a = GRUSequenceRanker(device="cpu", epochs=5, window=4, seed=7).fit(samples, ["f1", "f2"])
        b = GRUSequenceRanker(device="cpu", epochs=5, window=4, seed=7).fit(samples, ["f1", "f2"])
        assert a.score(samples[:10]) == b.score(samples[:10])

    def test_plugs_into_walk_forward(self) -> None:
        import math

        samples = _signal_samples(n_days=30, n_instruments=10)
        evidence = run_sample_walk_forward(
            samples=samples,
            config=WalkForwardConfig(
                train_window_days=10,
                test_window_days=3,
                step_days=3,
                purge_days=1,
                embargo_days=0,
                min_folds=1,
                label_horizon_days=1,
            ),
            model_version="gru-ranker-ic-v1",
            feature_set_version="test-fs",
            feature_names=["f1", "f2"],
            model=GRUSequenceRanker(device="cpu", epochs=3, window=4),
        )
        assert len(evidence.folds) >= 1
        assert math.isfinite(evidence.metrics["slippage_adjusted_sharpe"])
        assert set(evidence.selected_weights) <= {"f1", "f2"}

"""Tests for ``PurgedKFold``.

Focus is the leakage-safety invariants: no train index appears in any
test slice, and the purge/embargo gaps actually drop the samples that
would otherwise leak labels.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.services.research_service.modeling.validation.cross_val import PurgedKFold

_UTC = UTC


def _daily(n: int, start: datetime | None = None) -> list[datetime]:
    start = start or datetime(2024, 1, 1, tzinfo=_UTC)
    return [start + timedelta(days=i) for i in range(n)]


def test_rejects_invalid_n_splits() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)


def test_rejects_negative_gaps() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, purge_days=-1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, embargo_days=-1)


def test_naive_timestamps_rejected() -> None:
    pkf = PurgedKFold(n_splits=2)
    with pytest.raises(ValueError, match="timezone-aware"):
        list(pkf.split([datetime(2024, 1, 1), datetime(2024, 1, 2)]))


def test_descending_timestamps_rejected() -> None:
    pkf = PurgedKFold(n_splits=2)
    ts = _daily(4)
    descending = list(reversed(ts))
    with pytest.raises(ValueError, match="non-decreasing"):
        list(pkf.split(descending))


def test_splits_are_disjoint_and_cover_all() -> None:
    pkf = PurgedKFold(n_splits=5)
    ts = _daily(50)
    splits = list(pkf.split(ts))

    test_union: set[int] = set()
    for s in splits:
        assert not set(s.train_indices).intersection(s.test_indices)
        test_union.update(s.test_indices)

    assert test_union == set(range(50))


def test_remainder_distributed_across_first_folds() -> None:
    pkf = PurgedKFold(n_splits=5)
    ts = _daily(11)
    splits = list(pkf.split(ts))
    sizes = [len(s.test_indices) for s in splits]
    assert sum(sizes) == 11
    # The 11/5 remainder (1) goes to fold 0; other four folds get 2 each.
    assert sizes == [3, 2, 2, 2, 2]


def test_purge_drops_train_rows_next_to_test_window() -> None:
    pkf = PurgedKFold(n_splits=5, purge_days=2)
    ts = _daily(50)
    splits = list(pkf.split(ts))

    for s in splits:
        test_start = s.test_start
        test_end = s.test_end
        for idx in s.train_indices:
            t = ts[idx]
            assert not (test_start - timedelta(days=2) <= t < test_start)
            assert not (test_end < t <= test_end + timedelta(days=2))


def test_embargo_drops_train_rows_after_test_window() -> None:
    pkf = PurgedKFold(n_splits=5, purge_days=0, embargo_days=3)
    ts = _daily(50)
    splits = list(pkf.split(ts))
    for s in splits:
        for idx in s.train_indices:
            t = ts[idx]
            assert not (s.test_end < t <= s.test_end + timedelta(days=3))


def test_embargo_does_not_drop_rows_before_test_window() -> None:
    pkf = PurgedKFold(n_splits=5, purge_days=0, embargo_days=3)
    ts = _daily(50)
    splits = list(pkf.split(ts))
    # In the second fold onward, some train rows sit before the test
    # window.  Without purge, those should be retained.
    second = splits[1]
    assert any(ts[idx] < second.test_start - timedelta(days=3) for idx in second.train_indices)


def test_fold_count_matches_n_splits_for_sufficient_data() -> None:
    pkf = PurgedKFold(n_splits=5)
    assert len(list(pkf.split(_daily(50)))) == 5


def test_small_series_rejected() -> None:
    pkf = PurgedKFold(n_splits=5)
    with pytest.raises(ValueError, match="< n_splits"):
        list(pkf.split(_daily(3)))

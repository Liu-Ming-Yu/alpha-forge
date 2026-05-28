"""Pure fold-generation logic for walk-forward evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta

from quant_platform.services.research_service.modeling.walk_forward.models import (
    WalkForwardConfig,
    WalkForwardFold,
)


def generate_folds(
    start: datetime,
    end: datetime,
    config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    """Enumerate rolling-origin fold boundaries for ``[start, end)``."""
    if end <= start:
        raise ValueError("end must be after start")

    folds: list[WalkForwardFold] = []
    origin = start
    idx = 0
    train_len = timedelta(days=config.train_window_days)
    test_len = timedelta(days=config.test_window_days)
    step = timedelta(days=config.step_days)
    purge = timedelta(days=config.purge_days)
    embargo = timedelta(days=config.embargo_days)
    last_test_end: datetime | None = None

    while True:
        effective_origin = origin
        if last_test_end is not None and embargo.days > 0:
            effective_origin = max(origin, last_test_end + embargo)
        train_start = effective_origin
        train_end_raw = train_start + train_len
        train_end = train_end_raw - purge
        if train_end <= train_start:
            break
        test_start = train_end_raw
        test_end = test_start + test_len
        if test_end > end:
            break
        folds.append(
            WalkForwardFold(
                fold_index=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        idx += 1
        last_test_end = test_end
        origin = origin + step

    if len(folds) < config.min_folds:
        raise ValueError(
            f"generate_folds produced {len(folds)} folds, below min_folds="
            f"{config.min_folds}.  Shorten train_window_days/test_window_days "
            f"or widen [start, end)."
        )
    return folds


__all__ = ["generate_folds"]

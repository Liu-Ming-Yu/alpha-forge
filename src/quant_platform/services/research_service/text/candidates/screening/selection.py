"""Candidate ordering helpers shared by text screening and promotion artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _selected_passing_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
) -> tuple[str, ...]:
    passing = [row for row in rows if bool(row.get("passed"))]
    ordered = sorted(
        passing,
        key=lambda row: (
            _metric(row, "negative_ic_streak"),
            -_metric(row, "null_margin"),
            -_metric(row, "ic_mean"),
            -_metric(row, "icir"),
            str(row.get("feature_name", "")),
        ),
    )
    return tuple(str(row["feature_name"]) for row in ordered[: max(0, count)])


def _metric(row: Mapping[str, object], name: str) -> float:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    try:
        return float(metrics.get(name, 0.0))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _selected_or_passing(screen: Mapping[str, object]) -> tuple[str, ...]:
    raw_selected = screen.get("selected_candidates", ())
    selected = _string_tuple(raw_selected)
    if selected:
        return selected
    return _string_tuple(screen.get("passing_candidates", ()))


def _string_tuple(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        return ()
    return tuple(str(item) for item in raw)

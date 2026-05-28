"""Promotion guard for text candidate screens that require stability confirmation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.text.candidates.screening import (
    write_text_candidate_family_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def promote_text_candidate_screens(
    *,
    main_screen: Mapping[str, object],
    confirmation_screen: Mapping[str, object],
    full_screen: Mapping[str, object],
    feature_card_dir: Path,
    feature_family_file: Path,
    min_passing_candidates: int = 3,
) -> dict[str, object]:
    """Write text metadata only for candidates shared by all required screens."""
    screens = {
        "main": main_screen,
        "confirmation": confirmation_screen,
        "full_window": full_screen,
    }
    blockers = _screen_blockers(screens)
    passing_by_screen = {
        label: sorted(_passing_candidates(screen)) for label, screen in screens.items()
    }
    shared_set = set.intersection(*(set(names) for names in passing_by_screen.values()))
    full_order = _ordered_candidates(full_screen)
    shared = [name for name in full_order if name in shared_set]
    feature_set = _common_feature_set(screens)
    if feature_set is None:
        blockers.append("screen promoted_feature_set_version values do not match")
        feature_set = str(full_screen.get("promoted_feature_set_version", ""))
    if len(shared) < min_passing_candidates:
        blockers.append(
            f"text promotion shared {len(shared)} candidates, required {min_passing_candidates}"
        )
    base = {
        "shared_passing_candidates": shared[:min_passing_candidates],
        "passing_candidates_by_screen": passing_by_screen,
        "min_passing_candidates": min_passing_candidates,
        "promoted_feature_set_version": feature_set,
    }
    if blockers:
        return {
            **base,
            "passed": False,
            "reason": "text promotion blocked tracked feature metadata",
            "blockers": blockers,
            "promotion_artifacts_written": False,
            "feature_family_artifacts": {"written": False},
        }
    promotion_screen = {
        **dict(full_screen),
        "passed": True,
        "candidate_set": str(full_screen.get("candidate_set", "v10-alpha-quality")),
        "passing_candidates": shared[:min_passing_candidates],
        "selected_candidates": shared[:min_passing_candidates],
        "promoted_feature_set_version": feature_set,
    }
    artifacts = write_text_candidate_family_artifacts(
        screen=promotion_screen,
        feature_card_dir=feature_card_dir,
        feature_family_file=feature_family_file,
    )
    return {
        **base,
        "passed": True,
        "reason": "text promotion passed shared-candidate rule",
        "blockers": [],
        "promotion_artifacts_written": bool(artifacts.get("written", False)),
        "feature_family_artifacts": artifacts,
    }


def _screen_blockers(screens: Mapping[str, Mapping[str, object]]) -> list[str]:
    blockers: list[str] = []
    for label, screen in screens.items():
        if not bool(screen.get("passed")):
            blockers.append(f"{label} screen did not pass")
    return blockers


def _passing_candidates(screen: Mapping[str, object]) -> set[str]:
    raw = cast("Sequence[object]", screen.get("passing_candidates", ()))
    return {str(name) for name in raw}


def _ordered_candidates(screen: Mapping[str, object]) -> tuple[str, ...]:
    raw_selected = screen.get("selected_candidates", ())
    if isinstance(raw_selected, Sequence) and not isinstance(raw_selected, str):
        selected = tuple(str(name) for name in raw_selected)
        if selected:
            return selected
    raw_passing = screen.get("passing_candidates", ())
    if isinstance(raw_passing, Sequence) and not isinstance(raw_passing, str):
        return tuple(str(name) for name in raw_passing)
    return ()


def _common_feature_set(screens: Mapping[str, Mapping[str, object]]) -> str | None:
    values = {
        str(screen.get("promoted_feature_set_version", "")).strip() for screen in screens.values()
    }
    values.discard("")
    if len(values) != 1:
        return None
    return next(iter(values))


__all__ = ["promote_text_candidate_screens"]

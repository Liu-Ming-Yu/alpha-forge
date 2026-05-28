"""Promotion guard for intraday candidate screens."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.intraday.candidates.screening import (
    INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES,
    INTRADAY_MICROSTRUCTURE_V2_CANDIDATES,
    write_intraday_candidate_family_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def promote_intraday_candidate_screens(
    *,
    main_screen: Mapping[str, object],
    confirmation_screen: Mapping[str, object],
    full_screen: Mapping[str, object],
    feature_card_dir: Path,
    feature_family_file: Path,
    min_passing_candidates: int = 3,
) -> dict[str, object]:
    """Write tracked metadata only for candidates shared by all required screens."""
    screens = {
        "main": main_screen,
        "confirmation": confirmation_screen,
        "full_window": full_screen,
    }
    blockers = _screen_blockers(screens)
    passing_by_screen = {
        label: sorted(_passing_candidates(screen)) for label, screen in screens.items()
    }
    shared = sorted(set.intersection(*(set(names) for names in passing_by_screen.values())))
    known = {
        candidate.name
        for candidate in (
            *INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES,
            *INTRADAY_MICROSTRUCTURE_V2_CANDIDATES,
        )
    }
    shared_known = [name for name in shared if name in known]
    feature_set = _common_feature_set(screens)
    if feature_set is None:
        blockers.append("screen intraday_feature_set_version values do not match")
        feature_set = str(full_screen.get("intraday_feature_set_version", ""))
    if len(shared_known) < min_passing_candidates:
        blockers.append(
            "intraday promotion shared "
            f"{len(shared_known)} candidates, required {min_passing_candidates}"
        )
    base = {
        "shared_passing_candidates": shared_known,
        "passing_candidates_by_screen": passing_by_screen,
        "min_passing_candidates": min_passing_candidates,
        "intraday_feature_set_version": feature_set,
    }
    if blockers:
        return {
            **base,
            "passed": False,
            "reason": "intraday promotion blocked tracked feature metadata",
            "blockers": blockers,
            "promotion_artifacts_written": False,
            "feature_family_artifacts": {"written": False},
        }
    promotion_screen = {
        **dict(full_screen),
        "passed": True,
        "passing_candidates": shared_known,
        "intraday_feature_set_version": feature_set,
    }
    artifacts = write_intraday_candidate_family_artifacts(
        screen=promotion_screen,
        feature_card_dir=feature_card_dir,
        feature_family_file=feature_family_file,
    )
    return {
        **base,
        "passed": True,
        "reason": "intraday promotion passed shared-candidate rule",
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


def _common_feature_set(screens: Mapping[str, Mapping[str, object]]) -> str | None:
    values = {
        str(screen.get("intraday_feature_set_version", "")).strip() for screen in screens.values()
    }
    values.discard("")
    if len(values) != 1:
        return None
    return next(iter(values))


__all__ = ["promote_intraday_candidate_screens"]

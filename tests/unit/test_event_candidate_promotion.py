from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.services.research_service.events.candidates.promotion import (
    promote_event_candidate_screens,
)

if TYPE_CHECKING:
    from pathlib import Path

_PASSING = [
    "event_reaction_v2_sec_density_price_reversal_21d",
    "event_reaction_v2_attention_gap_reversal_21d",
    "event_reaction_v2_post_event_drift_quality_21d",
]


def test_event_promotion_writes_tracked_family_after_shared_rule(tmp_path: Path) -> None:
    family_file = tmp_path / "infra" / "config" / "feature_families" / "event.json"
    card_dir = tmp_path / "infra" / "config" / "feature_cards" / "event"

    result = promote_event_candidate_screens(
        main_screen=_screen(_PASSING + ["event_reaction_v2_extreme_attention_reversal_21d"]),
        confirmation_screen=_screen(
            _PASSING + ["event_reaction_v2_crowded_medium_momentum_decay_21d"]
        ),
        full_screen=_screen(_PASSING),
        feature_card_dir=card_dir,
        feature_family_file=family_file,
        min_passing_candidates=3,
    )

    assert result["passed"] is True
    assert result["promotion_artifacts_written"] is True
    assert result["shared_passing_candidates"] == sorted(_PASSING)
    assert family_file.exists()
    payload = json.loads(family_file.read_text(encoding="utf-8"))
    assert sorted(name for members in payload["families"].values() for name in members) == sorted(
        _PASSING
    )
    assert len(list(card_dir.glob("*.json"))) == 3


def test_event_promotion_blocks_when_fewer_than_three_shared(tmp_path: Path) -> None:
    family_file = tmp_path / "family.json"
    card_dir = tmp_path / "cards"

    result = promote_event_candidate_screens(
        main_screen=_screen(_PASSING[:2]),
        confirmation_screen=_screen(_PASSING[:2]),
        full_screen=_screen(_PASSING[:2]),
        feature_card_dir=card_dir,
        feature_family_file=family_file,
        min_passing_candidates=3,
    )

    assert result["passed"] is False
    assert result["promotion_artifacts_written"] is False
    assert "shared 2 candidates" in " ".join(result["blockers"])
    assert not family_file.exists()
    assert not card_dir.exists()


def test_event_promotion_blocks_failed_screen_without_artifacts(tmp_path: Path) -> None:
    result = promote_event_candidate_screens(
        main_screen=_screen(_PASSING, passed=False),
        confirmation_screen=_screen(_PASSING),
        full_screen=_screen(_PASSING),
        feature_card_dir=tmp_path / "cards",
        feature_family_file=tmp_path / "family.json",
        min_passing_candidates=3,
    )

    assert result["passed"] is False
    assert "main screen did not pass" in result["blockers"]
    assert not (tmp_path / "family.json").exists()


def _screen(names: list[str], *, passed: bool = True) -> dict[str, object]:
    return {
        "passed": passed,
        "event_feature_set_version": "paper-alpha-event-reaction-v2",
        "passing_candidates": names,
        "candidate_set": "event-reaction-v2",
    }

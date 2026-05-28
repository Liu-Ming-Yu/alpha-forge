from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.services.research_service.intraday.candidates.promotion import (
    promote_intraday_candidate_screens,
)

if TYPE_CHECKING:
    from pathlib import Path

_PASSING = [
    "opening_drive_confirmation_1d_decay",
    "close_pressure_continuation_1d_decay",
    "vwap_accumulation_pressure_1d_decay",
]


def test_intraday_promotion_writes_tracked_family_after_shared_rule(tmp_path: Path) -> None:
    family_file = tmp_path / "infra" / "config" / "feature_families" / "intraday.json"
    card_dir = tmp_path / "infra" / "config" / "feature_cards" / "intraday"

    result = promote_intraday_candidate_screens(
        main_screen=_screen(_PASSING + ["range_expansion_drift_1d_decay"]),
        confirmation_screen=_screen(_PASSING + ["intraday_volatility_compression_3d_decay"]),
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


def test_intraday_promotion_blocks_when_fewer_than_three_shared(tmp_path: Path) -> None:
    family_file = tmp_path / "family.json"
    card_dir = tmp_path / "cards"

    result = promote_intraday_candidate_screens(
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


def test_intraday_promotion_blocks_failed_screen_without_artifacts(tmp_path: Path) -> None:
    result = promote_intraday_candidate_screens(
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
        "intraday_feature_set_version": "paper-alpha-intraday-microstructure-v1",
        "passing_candidates": names,
        "candidate_set": "seed",
    }

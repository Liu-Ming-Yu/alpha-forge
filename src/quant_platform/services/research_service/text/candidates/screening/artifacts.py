"""Artifact writers for promoted text candidate families."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.services.research_service.text.candidates.catalog import (
    TextCandidateSpec,
    text_candidate_specs_by_name,
)
from quant_platform.services.research_service.text.candidates.screening.selection import (
    _selected_or_passing,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def write_text_candidate_family_artifacts(
    *,
    screen: Mapping[str, object],
    feature_card_dir: Path,
    feature_family_file: Path,
) -> dict[str, object]:
    """Write proposed paper feature cards/family metadata after a passing text screen."""
    if not bool(screen.get("passed")):
        return {"written": False, "reason": "screen did not pass"}
    feature_set_version = str(
        screen.get("promoted_feature_set_version", "paper-alpha-catalyst-v10")
    )
    selected = _selected_or_passing(screen)
    candidate_set = str(screen.get("candidate_set", "v10-alpha-quality"))
    specs = text_candidate_specs_by_name(candidate_set)
    feature_card_dir.mkdir(parents=True, exist_ok=True)
    families: dict[str, list[str]] = {}
    written_cards: list[str] = []
    for feature_name in selected:
        spec = specs.get(feature_name)
        if spec is None:
            continue
        card_path = feature_card_dir / f"{feature_name}.json"
        card_path.write_text(
            _json_dump(_feature_card_payload(spec, feature_set_version)),
            encoding="utf-8",
        )
        families[f"sec_primary_text_{feature_name.removesuffix('_21d')}"] = [feature_name]
        written_cards.append(str(card_path))
    feature_family_file.parent.mkdir(parents=True, exist_ok=True)
    feature_family_file.write_text(
        _json_dump({"feature_set_version": feature_set_version, "families": families}),
        encoding="utf-8",
    )
    return {
        "written": True,
        "feature_card_dir": str(feature_card_dir),
        "feature_family_file": str(feature_family_file),
        "feature_cards": written_cards,
        "selected_candidates": list(selected),
    }


def _feature_card_payload(
    spec: TextCandidateSpec,
    feature_set_version: str,
) -> dict[str, object]:
    return {
        "name": spec.name,
        "version": feature_set_version,
        "owner": "research",
        "economic_thesis": spec.thesis
        or "Primary SEC filing text should contain slow-moving alpha information after audit.",
        "source_datasets": ["sec_primary_filings", "llm_text_v5_features", "daily_adjusted_ohlcv"],
        "required_lags": [
            "Use only manifest-listed primary SEC filing text, deterministic compacted excerpts, "
            "and LLM extraction artifacts with available_at <= the daily as_of timestamp."
        ],
        "valid_universe": "15-name liquid U.S. equity validation universe",
        "expected_sign": "positive",
        "horizon_days": 21,
        "expected_turnover": "low",
        "state": "paper",
        "failure_modes": [
            "LLM extraction drift",
            "primary filing text is already priced",
            "source density weakens outside the validation universe",
        ],
        "risk_exposures": [
            "single-name event risk",
            "earnings season crowding",
            "sector outlook-cycle exposure",
        ],
    }


def _json_dump(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"

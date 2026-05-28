"""Report and metadata writers for intraday candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.campaigns.screening.reports import (
    render_candidate_screen_report,
)
from quant_platform.services.research_service.intraday.candidates.screening.candidates import (
    INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES,
    INTRADAY_MICROSTRUCTURE_V2_CANDIDATES,
)
from quant_platform.services.research_service.intraday.candidates.screening.types import (
    INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION,
    IntradayCandidateSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def render_intraday_candidate_screen_report(screen: Mapping[str, object]) -> str:
    return render_candidate_screen_report(
        screen,
        title="Intraday Candidate Screen",
        feature_set_key="intraday_feature_set_version",
        feature_set_label="Intraday feature set",
        next_action=(
            "Only write intraday microstructure feature cards and family metadata when at "
            "least three candidates pass this screen without threshold relaxation."
        ),
    )


def write_intraday_candidate_family_artifacts(
    *,
    screen: Mapping[str, object],
    feature_card_dir: Path,
    feature_family_file: Path,
) -> dict[str, object]:
    """Write proposed diagnostics-only paper feature metadata after a passing screen."""
    if not bool(screen.get("passed")):
        return {"written": False, "reason": "screen did not pass"}
    feature_set_version = str(
        screen.get("intraday_feature_set_version", INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION)
    )
    passing = {str(name) for name in cast("Sequence[object]", screen.get("passing_candidates", ()))}
    specs = {
        candidate.name: candidate
        for candidate in (
            *INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES,
            *INTRADAY_MICROSTRUCTURE_V2_CANDIDATES,
        )
    }
    feature_card_dir.mkdir(parents=True, exist_ok=True)
    families: dict[str, list[str]] = {}
    written_cards: list[str] = []
    for feature_name in sorted(passing):
        spec = specs.get(feature_name)
        if spec is None:
            continue
        card_path = feature_card_dir / f"{feature_name}.json"
        card_path.write_text(_json_dump(_feature_card_payload(spec, feature_set_version)))
        families[f"intraday_microstructure_{feature_name.removesuffix('_decay')}"] = [feature_name]
        written_cards.append(str(card_path))
    feature_family_file.parent.mkdir(parents=True, exist_ok=True)
    feature_family_file.write_text(
        _json_dump({"feature_set_version": feature_set_version, "families": families})
    )
    return {
        "written": True,
        "feature_card_dir": str(feature_card_dir),
        "feature_family_file": str(feature_family_file),
        "feature_cards": written_cards,
    }


def _feature_card_payload(
    spec: IntradayCandidateSpec,
    feature_set_version: str,
) -> dict[str, object]:
    return {
        "name": spec.name,
        "version": feature_set_version,
        "owner": "research",
        "economic_thesis": spec.thesis,
        "source_datasets": ["canonical_1m_intraday_bars", "daily_adjusted_ohlcv"],
        "required_lags": [
            "Use only complete 1-minute sessions with end timestamps strictly before sample as_of."
        ],
        "valid_universe": "15-name liquid U.S. equity validation universe",
        "expected_sign": "positive",
        "horizon_days": 21,
        "expected_turnover": "high",
        "state": "paper",
        "failure_modes": [
            "auction prints distort open or close pressure",
            "minute-bar vendor gaps create false microstructure signals",
            "intraday pressure proxy is unstable across volatility regimes",
        ],
        "risk_exposures": ["liquidity regime", "opening/closing auction concentration"],
    }


def _json_dump(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"

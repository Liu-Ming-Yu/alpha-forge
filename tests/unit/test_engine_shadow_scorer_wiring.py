"""Unit tests for optional shadow scorer wiring helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_platform.config import BoostingSettings, PlatformSettings
from quant_platform.engines.framework.types import RunMode
from quant_platform.engines.shadow.scorer_wiring import (
    build_shadow_boosting_scorer,
    build_shadow_text_scorer,
)


def test_shadow_text_scorer_wiring_noops_when_disabled() -> None:
    settings = PlatformSettings(_env_file=None)

    assert build_shadow_text_scorer(settings=settings, session=object()) is None


def test_shadow_boosting_wiring_noops_when_disabled() -> None:
    settings = PlatformSettings(_env_file=None)

    assert (
        build_shadow_boosting_scorer(
            settings=settings,
            run_mode=RunMode.SHADOW,
            session=object(),
        )
        is None
    )


def test_shadow_boosting_wiring_noops_for_non_shadow_modes(tmp_path) -> None:
    settings = PlatformSettings(
        _env_file=None,
        boosting=BoostingSettings(
            enabled=True,
            artifact_manifest=str(tmp_path / "manifest.json"),
            shadow_artifact_root=str(tmp_path / "shadow"),
        ),
    )

    assert (
        build_shadow_boosting_scorer(
            settings=settings,
            run_mode=RunMode.PAPER,
            session=object(),
        )
        is None
    )


def test_shadow_boosting_wiring_requires_manifest_when_enabled() -> None:
    settings = PlatformSettings(
        _env_file=None,
        boosting=BoostingSettings(enabled=True, artifact_manifest=""),
    )

    with pytest.raises(ValueError, match="ARTIFACT_MANIFEST"):
        build_shadow_boosting_scorer(
            settings=settings,
            run_mode=RunMode.SHADOW,
            session=SimpleNamespace(signal_contribution_repo=object()),
        )

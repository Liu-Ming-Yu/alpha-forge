"""Campaign-fitted classical weights load into the live signal model (WS3)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from quant_platform.bootstrap.signal_models import build_default_signal_model
from quant_platform.bootstrap.signal_models.classical_manifest import (
    load_classical_signal_model,
)
from quant_platform.config import FactorSettings, PlatformSettings

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "model_version": "u300-classical-v1",
        "feature_set_version": "1.1.0",
        "passed": True,
        "selected_weights": {"momentum_3m": 0.4, "low_vol_63d": 0.35, "reversal_21d": 0.25},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_fitted_weights_and_pins_feature_set_version(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "campaign_manifest.json")
    model = load_classical_signal_model(manifest)
    assert set(model.feature_names) == {"momentum_3m", "low_vol_63d", "reversal_21d"}
    assert model.model_version == "u300-classical-v1"
    assert model.expected_feature_set_version == "1.1.0"


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classical_signal_model(tmp_path / "does_not_exist.json")


def test_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "campaign_manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        load_classical_signal_model(bad)


def test_no_selected_weights_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "campaign_manifest.json", selected_weights={})
    with pytest.raises(ValueError, match="selected_weights"):
        load_classical_signal_model(manifest)


def test_all_zero_weights_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "campaign_manifest.json",
        selected_weights={"momentum_3m": 0.0},
    )
    with pytest.raises(ValueError, match="zero weights"):
        load_classical_signal_model(manifest)


def test_ineligible_manifest_still_loads(tmp_path: Path) -> None:
    """passed:false manifests load (governance gates promotion, not loading)."""
    manifest = _write_manifest(tmp_path / "campaign_manifest.json", passed=False)
    model = load_classical_signal_model(manifest)
    assert len(model.feature_names) == 3


def test_build_default_signal_model_uses_manifest_when_set(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "campaign_manifest.json")
    settings = PlatformSettings(
        _env_file=None,
        factors=FactorSettings(fitted_weights_manifest=str(manifest)),
    )
    model = build_default_signal_model(settings)
    assert set(model.feature_names) == {"momentum_3m", "low_vol_63d", "reversal_21d"}
    assert model.expected_feature_set_version == "1.1.0"


def test_build_default_signal_model_falls_back_to_hand_picked() -> None:
    settings = PlatformSettings(_env_file=None)
    model = build_default_signal_model(settings)
    # Hand-picked momentum-heavy defaults remain when no manifest is configured.
    assert "momentum_12m_1m" in model.feature_names
    assert model.expected_feature_set_version is None

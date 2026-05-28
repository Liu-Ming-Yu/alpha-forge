"""Unit tests for FamilyManifest + family-level registry indexing."""

from __future__ import annotations

import pytest

from quant_platform.research.features import (
    FamilyManifest,
    FeatureRegistry,
    FeatureSpec,
    bootstrap_default_families,
    get_global_registry,
)


def _spec(name: str, *, family: str = "price_volume", version: str = "test-v1") -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family=family,  # type: ignore[arg-type]
        description=f"{name} test spec",
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=1,
        version=version,
    )


# ---------------------------------------------------------------------------
# FamilyManifest validation
# ---------------------------------------------------------------------------


def test_manifest_requires_non_empty_version() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FamilyManifest(
            name="price_volume",
            version="   ",
            feature_specs=(_spec("foo"),),
            required_input_columns=("close",),
            key_columns=("instrument_id", "date"),
            default_training_feature_names=("foo",),
        )


def test_manifest_requires_at_least_one_spec() -> None:
    with pytest.raises(ValueError, match="at least one spec"):
        FamilyManifest(
            name="price_volume",
            version="v1",
            feature_specs=(),
            required_input_columns=("close",),
            key_columns=("instrument_id", "date"),
            default_training_feature_names=(),
        )


def test_manifest_rejects_version_mismatch() -> None:
    with pytest.raises(ValueError, match="spec.version must equal"):
        FamilyManifest(
            name="price_volume",
            version="v2",
            feature_specs=(_spec("foo", version="v1"),),
            required_input_columns=("close",),
            key_columns=("instrument_id", "date"),
            default_training_feature_names=("foo",),
        )


def test_manifest_rejects_family_mismatch() -> None:
    with pytest.raises(ValueError, match="spec.family must equal"):
        FamilyManifest(
            name="price_volume",
            version="test-v1",
            feature_specs=(_spec("foo", family="fundamentals"),),
            required_input_columns=("close",),
            key_columns=("instrument_id", "date"),
            default_training_feature_names=("foo",),
        )


def test_manifest_rejects_training_names_not_in_specs() -> None:
    with pytest.raises(ValueError, match="default_training_feature_names"):
        FamilyManifest(
            name="price_volume",
            version="test-v1",
            feature_specs=(_spec("foo"),),
            required_input_columns=("close",),
            key_columns=("instrument_id", "date"),
            default_training_feature_names=("foo", "not_a_real_spec"),
        )


def test_manifest_rejects_orphan_alias() -> None:
    """An alias spec whose ``canonical_name`` is not also in
    ``feature_specs`` is rejected: aliases need a canonical to point at."""
    alias_spec = FeatureSpec(
        name="alias_name",
        family="price_volume",
        description="alias",
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=1,
        version="test-v1",
        canonical_name="missing_canonical",
    )
    with pytest.raises(ValueError, match="not present in feature_specs"):
        FamilyManifest(
            name="price_volume",
            version="test-v1",
            feature_specs=(alias_spec,),
            required_input_columns=("close",),
            key_columns=("instrument_id", "date"),
            default_training_feature_names=(),
        )


def test_manifest_accepts_alias_pointing_at_present_canonical() -> None:
    """The bidirectional alias rule: canonical lists aliases, alias
    points at canonical via canonical_name. Both must be in
    feature_specs."""
    canonical = _spec("the_canonical")
    alias = FeatureSpec(
        name="the_alias",
        family="price_volume",
        description="alias",
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=1,
        version="test-v1",
        canonical_name="the_canonical",
    )
    # Must not raise.
    manifest = FamilyManifest(
        name="price_volume",
        version="test-v1",
        feature_specs=(canonical, alias),
        required_input_columns=("close",),
        key_columns=("instrument_id", "date"),
        default_training_feature_names=("the_canonical",),
    )
    assert manifest.feature_names == ("the_canonical", "the_alias")


def test_manifest_feature_names_property() -> None:
    manifest = FamilyManifest(
        name="price_volume",
        version="test-v1",
        feature_specs=(_spec("foo"), _spec("bar")),
        required_input_columns=("close",),
        key_columns=("instrument_id", "date"),
        default_training_feature_names=("foo", "bar"),
    )
    assert manifest.feature_names == ("foo", "bar")


# ---------------------------------------------------------------------------
# Registry family indexing
# ---------------------------------------------------------------------------


def test_local_registry_round_trip() -> None:
    registry = FeatureRegistry()
    manifest = FamilyManifest(
        name="price_volume",
        version="test-v1",
        feature_specs=(_spec("foo"),),
        required_input_columns=("close",),
        key_columns=("instrument_id", "date"),
        default_training_feature_names=("foo",),
    )
    registry.register_family(manifest)
    assert registry.has_family("price_volume")
    assert registry.has_family("price_volume", "test-v1")
    assert registry.get_family("price_volume") == manifest
    # The specs were also registered through register_many.
    assert registry.has("foo", "test-v1")


def test_register_family_is_idempotent_for_identical_manifest() -> None:
    registry = FeatureRegistry()
    manifest = FamilyManifest(
        name="price_volume",
        version="test-v1",
        feature_specs=(_spec("foo"),),
        required_input_columns=("close",),
        key_columns=("instrument_id", "date"),
        default_training_feature_names=("foo",),
    )
    registry.register_family(manifest)
    registry.register_family(manifest)  # no error
    assert registry.families() == ("price_volume",)


def test_register_family_rejects_conflicting_manifest() -> None:
    registry = FeatureRegistry()
    m1 = FamilyManifest(
        name="price_volume",
        version="test-v1",
        feature_specs=(_spec("foo"),),
        required_input_columns=("close",),
        key_columns=("instrument_id", "date"),
        default_training_feature_names=("foo",),
    )
    m2 = FamilyManifest(
        name="price_volume",
        version="test-v1",
        feature_specs=(_spec("bar"),),
        required_input_columns=("close",),
        key_columns=("instrument_id", "date"),
        default_training_feature_names=("bar",),
    )
    registry.register_family(m1)
    with pytest.raises(ValueError, match="different manifest"):
        registry.register_family(m2)


def test_get_family_unknown_raises_with_known_list() -> None:
    registry = FeatureRegistry()
    with pytest.raises(KeyError, match="no family registered"):
        registry.get_family("nope")


# ---------------------------------------------------------------------------
# Global bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_registers_every_default_family() -> None:
    bootstrap_default_families()
    registry = get_global_registry()
    assert registry.has_family("price_volume")
    assert registry.has_family("fundamentals")
    assert registry.has_family("formulaic")


def test_bootstrap_is_idempotent() -> None:
    bootstrap_default_families()
    bootstrap_default_families()  # second call must not raise
    registry = get_global_registry()
    assert set(registry.families()) >= {"price_volume", "fundamentals", "formulaic"}


def test_importing_features_package_auto_populates_registry() -> None:
    """``import quant_platform.research.features`` should be enough to
    query the registry; explicit family imports are not required."""
    # The package's __init__.py calls bootstrap_default_families() at
    # the bottom of the module, so by the time this test runs the
    # registry has every shipped family — regardless of whether the
    # test directly imported the family packages.
    registry = get_global_registry()
    families = set(registry.families())
    assert {"price_volume", "fundamentals", "formulaic"} <= families

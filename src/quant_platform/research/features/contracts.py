"""Shared contracts for the multi-family alpha feature factory.

Feature families return a :class:`FeatureFrame` rather than a bare
``DataFrame`` so downstream governance code can keep feature values, specs,
versions, signal timing, and coverage together.

All price-volume starter features are end-of-day signals: they are computed
after the close of ``date`` and may only be used for forward returns strictly
after that date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

FeatureDirection = Literal["+", "-", "unknown"]
FeatureFamily = Literal[
    "price_volume",
    "fundamentals",
    "formulaic",
    "text",
    "microstructure",
    "options",
    "macro",
    "ownership",
    "estimates",
    "learned",
    "regime",
]
SignalTimestamp = Literal["eod_after_close", "bod_before_open", "intraday"]

# Derived from the Literal so the two cannot drift apart.
_SIGNAL_TIMESTAMPS: frozenset[str] = frozenset(get_args(SignalTimestamp))


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one exported feature column.

    The :class:`FeatureSpec` is the single source of truth for every
    feature shipping through the pipeline. Governance, sign-audits,
    neutralisation, walk-forward gating, and the feature catalog all
    read from this object — not from naming conventions.

    Attributes
    ----------
    name:
        Versioned feature column name. Stable for the lifetime of a
        feature version; rename = new version.
    family:
        Which :data:`FeatureFamily` this feature belongs to.
    description:
        Human-readable one-paragraph description. The platform
        convention is ``"formula. Direction interpretation."`` — see
        either family for examples.
    expected_direction:
        Empirically-validated sign of the IC for the *exported*
        feature value. ``"+"`` for positive-oriented features (the
        default); ``"-"`` is only allowed when ``larger_is_better=False``;
        ``"unknown"`` for evidence-uncertain features whose direction
        will be confirmed by IC tests.
    required_inputs:
        Tuple of raw input column names this feature depends on. Used
        by the dependency graph to detect missing inputs before
        compute.
    point_in_time:
        ``True`` if the computation is guaranteed not to use future
        data at ``(instrument_id, date)``. Required to be ``True`` for
        any spec admitted to a production feature set.
    lookback_days:
        Maximum number of calendar/trading days of history the
        feature needs before it produces a non-NaN value. The precise
        boundary is enforced by the family preparator via
        ``min_periods``; this field is the estimate the walk-forward
        analyzer uses to pre-warm history at fold boundaries.
    version:
        Family-level feature-set version, e.g.
        ``"price-volume-starter-v1"``. Bumped when the formula,
        inputs, or normalisation change.
    signal_timestamp:
        When in the trading day the signal becomes available. Default
        ``"eod_after_close"`` — fundamentals are filed mid-day but
        consumed in next-day rebalances; price-volume features close
        after the close. ``"bod_before_open"`` and ``"intraday"`` are
        reserved for future families.
    neutralization_supported:
        ``True`` if the feature is meaningful after sector / industry
        / size residualisation. Most cross-sectional features are; a
        few (e.g. market-wide regime scores) are not.
    larger_is_better:
        Whether the platform's positive-orientation contract applies.
        Non-negative-weight evaluators depend on this; uncertain-
        direction features should set this to ``False``.
    canonical_name:
        When this spec names an alias view of another feature
        (identical formula, different exported name), point at the
        canonical name. Aliases are emitted into the FeatureFrame for
        provenance but
        :meth:`FeatureFrame.training_feature_names` excludes them.
    aliases:
        On the **canonical** spec, list the other names this feature
        is known by. On an **alias** spec, leave this empty — the
        alias points at the canonical via ``canonical_name``. This
        is the only bidirectional rule for aliases; getting it wrong
        gives correct compute and incorrect catalog metadata.
    """

    name: str
    family: FeatureFamily
    description: str
    expected_direction: FeatureDirection
    required_inputs: tuple[str, ...]
    point_in_time: bool
    lookback_days: int | None
    version: str
    signal_timestamp: SignalTimestamp = "eod_after_close"
    neutralization_supported: bool = True
    larger_is_better: bool = True
    canonical_name: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FeatureSpec.name must be non-empty")
        if not self.version.strip():
            raise ValueError(f"FeatureSpec({self.name!r}).version must be non-empty")
        if self.expected_direction == "-" and self.larger_is_better:
            raise ValueError(
                f"FeatureSpec({self.name!r}) has expected_direction='-' but "
                "larger_is_better=True; invert and rename the feature, or set "
                "larger_is_better=False."
            )
        if self.signal_timestamp not in _SIGNAL_TIMESTAMPS:
            raise ValueError(
                f"FeatureSpec({self.name!r}).signal_timestamp is invalid: {self.signal_timestamp!r}"
            )
        if self.canonical_name is not None and not self.canonical_name.strip():
            raise ValueError(
                f"FeatureSpec({self.name!r}).canonical_name must be non-empty when set"
            )
        if self.name in self.aliases:
            raise ValueError(f"FeatureSpec({self.name!r}) cannot alias itself")

    @property
    def is_alias(self) -> bool:
        """Whether this spec names an alternate view of another feature."""
        return self.canonical_name is not None and self.canonical_name != self.name


@dataclass(frozen=True)
class BaseFamilyConfig:
    """Common fields every family ``Config`` carries.

    Family configs subclass this dataclass and add family-specific
    knobs (lookback windows, min-periods policies, missing-input
    fallback). The base only requires what's universal: the family
    version string that gets stamped into every produced
    :class:`FeatureSpec`.

    Subclasses must also be ``@dataclass(frozen=True)``. All fields
    here have defaults, so subclasses can introduce required (no-
    default) fields without MRO ordering pain — but the convention
    is to keep every config field optional with a sensible default
    so a contributor can instantiate ``FooConfig()`` and get the
    canonical production setup.
    """

    version: str = ""

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError(f"{type(self).__name__}.version must be non-empty")


@dataclass(frozen=True)
class FamilyManifest:
    """Manifest for one feature family — the contract every family ships.

    Every family package declares one ``MANIFEST`` at its ``__init__.py``
    top level and registers it via
    :func:`~.registry.FeatureRegistry.register_family`. The manifest
    bundles everything downstream code needs to enumerate, validate, or
    materialise a family's output without having to import the family's
    private modules:

    * ``name`` — the :data:`FeatureFamily` enum value.
    * ``version`` — the feature-set version string this manifest pins
      (matches every contained :attr:`FeatureSpec.version`).
    * ``feature_specs`` — the ordered tuple of FeatureSpecs the family
      emits.
    * ``required_input_columns`` — columns the family's compute function
      expects on the input panel/frame; missing columns are a hard
      error.
    * ``key_columns`` — the ``(instrument_id, date-ish)`` pair on the
      produced :class:`FeatureFrame`. ``"date"`` for daily-bar families,
      ``"datekey"`` for fundamentals, etc.
    * ``default_training_feature_names`` — the subset of
      :attr:`feature_specs` names suitable for default training
      matrices (excludes alias specs).

    The :class:`FamilyManifest` lets governance/walk-forward code take
    one object and discover everything about a family. It also gives a
    "you must ship a manifest" checklist to new-family contributors —
    if a field can't be filled in, the family isn't ready to register.
    """

    name: FeatureFamily
    version: str
    feature_specs: tuple[FeatureSpec, ...]
    required_input_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    default_training_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError(f"FamilyManifest({self.name!r}).version must be non-empty")
        if not self.feature_specs:
            raise ValueError(f"FamilyManifest({self.name!r}) must declare at least one spec")
        if not self.key_columns:
            raise ValueError(f"FamilyManifest({self.name!r}).key_columns must be non-empty")

        version_mismatches = [
            spec.name for spec in self.feature_specs if spec.version != self.version
        ]
        if version_mismatches:
            raise ValueError(
                f"FamilyManifest({self.name!r}): every spec.version must equal "
                f"manifest.version={self.version!r}; mismatching specs: "
                f"{version_mismatches!r}"
            )

        family_mismatches = [spec.name for spec in self.feature_specs if spec.family != self.name]
        if family_mismatches:
            raise ValueError(
                f"FamilyManifest({self.name!r}): every spec.family must equal "
                f"manifest.name={self.name!r}; mismatching specs: "
                f"{family_mismatches!r}"
            )

        spec_names = {spec.name for spec in self.feature_specs}
        missing_training = [
            name for name in self.default_training_feature_names if name not in spec_names
        ]
        if missing_training:
            raise ValueError(
                f"FamilyManifest({self.name!r}): default_training_feature_names "
                f"contain names not in feature_specs: {missing_training!r}"
            )

        orphan_aliases = [
            spec.name
            for spec in self.feature_specs
            if spec.canonical_name is not None and spec.canonical_name not in spec_names
        ]
        if orphan_aliases:
            raise ValueError(
                f"FamilyManifest({self.name!r}): alias specs point at canonical "
                f"names not present in feature_specs: {orphan_aliases!r}. Either "
                "include the canonical in feature_specs or drop the alias."
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Tuple of feature names in :attr:`feature_specs` order."""
        return tuple(spec.name for spec in self.feature_specs)


@dataclass(frozen=True)
class FeatureFrame:
    """Result wrapper for a computed family's feature panel.

    ``key_columns`` is **required** — there is no default. Families that
    key by ``(instrument_id, date)`` (price-volume) and ``(instrument_id,
    datekey)`` (fundamentals) both pass theirs explicitly, and future
    families with a third spelling (intraday timestamp, multi-asset key)
    must declare what they use. The old default of ``("instrument_id",
    "date")`` was silently wrong for fundamentals; removing it forces
    each compute site to be honest about its key.
    """

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_specs: Mapping[str, FeatureSpec]
    coverage: Mapping[str, int]
    key_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key_columns:
            raise ValueError("FeatureFrame.key_columns must be non-empty")

        duplicate_names = _duplicates(self.feature_names)
        if duplicate_names:
            raise ValueError(f"FeatureFrame: duplicate feature_names: {duplicate_names!r}")

        missing_key_columns = [name for name in self.key_columns if name not in self.frame.columns]
        if missing_key_columns:
            raise ValueError(f"FeatureFrame: frame missing key columns: {missing_key_columns!r}")

        missing_columns = [name for name in self.feature_names if name not in self.frame.columns]
        if missing_columns:
            raise ValueError(f"FeatureFrame: frame missing feature columns: {missing_columns!r}")

        missing_specs = [name for name in self.feature_names if name not in self.feature_specs]
        if missing_specs:
            raise ValueError(f"FeatureFrame: feature_names without specs: {missing_specs!r}")

        mismatched_specs = [
            name for name in self.feature_names if self.feature_specs[name].name != name
        ]
        if mismatched_specs:
            raise ValueError(
                f"FeatureFrame: feature_specs keys do not match spec.name for {mismatched_specs!r}"
            )

        expected_coverage_keys = set(self.feature_names)
        actual_coverage_keys = set(self.coverage)
        if actual_coverage_keys != expected_coverage_keys:
            missing = sorted(expected_coverage_keys - actual_coverage_keys)
            extra = sorted(actual_coverage_keys - expected_coverage_keys)
            raise ValueError(
                "FeatureFrame: coverage keys must match feature_names; "
                f"missing={missing!r}, extra={extra!r}"
            )

    @property
    def training_feature_names(self) -> tuple[str, ...]:
        """Feature columns suitable for default training matrices.

        Alias specs stay in the catalog for provenance, but a canonical feature
        should appear only once in a default training matrix.
        """
        return tuple(name for name in self.feature_names if not self.feature_specs[name].is_alias)


def _duplicates(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


__all__ = [
    "BaseFamilyConfig",
    "FamilyManifest",
    "FeatureDirection",
    "FeatureFamily",
    "FeatureFrame",
    "FeatureSpec",
    "SignalTimestamp",
]
